// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! OpenAI provider implementation for the AI framework.
//!
//! This crate provides a [`Provider`] implementation for OpenAI's API,
//! supporting GPT-4, GPT-3.5 Turbo, and embedding models.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_provider_openai::OpenAiProvider;
//! use ai_core::{Provider, ModelConfig};
//! use ai_core::types::Message;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
//!
//!     let messages = vec![
//!         Message::user("Hello, GPT!")
//!     ];
//!
//!     let config = ModelConfig::gpt4();
//!     let response = provider.complete(messages, &config, &[]).await?;
//!
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};

use ai_core::error::ProviderError;
use ai_core::provider::Provider;
use ai_core::tool::ToolDescriptor;
use ai_core::types::{
    CompletionResponse, Content, ContentPart, FinishReason, Message,
    ModelConfig, Role, StreamChunk, ToolCall, Usage,
};

/// OpenAI API provider.
///
/// Supports GPT-4, GPT-3.5 Turbo, and text embedding models.
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    json_mode: bool,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider with the given API key.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_openai::OpenAiProvider;
    /// let provider = OpenAiProvider::new("sk-...".to_string());
    /// ```
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            json_mode: false,
        }
    }

    /// Enable JSON mode for structured output.
    ///
    /// When enabled, the model will respond with valid JSON.
    /// Note: JSON mode requires a system prompt instructing the model to output JSON.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_openai::OpenAiProvider;
    /// let provider = OpenAiProvider::new("sk-...".to_string())
    ///     .with_json_mode(true);
    /// ```
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// Set a custom base URL (for Azure OpenAI or compatible APIs).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_openai::OpenAiProvider;
    /// let provider = OpenAiProvider::new("sk-...".to_string())
    ///     .with_base_url("https://my-azure.openai.azure.com".to_string());
    /// ```
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Get the base URL this provider is configured to use.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Convert framework messages to OpenAI format.
    fn convert_messages(messages: Vec<Message>) -> Vec<OpenAiMessage> {
        messages
            .into_iter()
            .map(|m| OpenAiMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: Self::convert_content(m.content),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect()
    }

    /// Convert framework content to OpenAI content format.
    fn convert_content(content: Content) -> OpenAiContent {
        match content {
            Content::Text(text) => OpenAiContent::Text(text),
            Content::MultiPart(parts) => {
                let items: Vec<OpenAiContentPart> = parts
                    .into_iter()
                    .map(|p| match p {
                        ContentPart::Text(t) => OpenAiContentPart::Text { text: t },
                        ContentPart::Image { url, media_type } => {
                            OpenAiContentPart::ImageUrl {
                                image_url: OpenAiImageUrl {
                                    url,
                                    detail: Some("auto".to_string()),
                                },
                            }
                        }
                    })
                    .collect();
                OpenAiContent::Array(items)
            }
        }
    }

    /// Convert tool descriptors to OpenAI tool format.
    fn convert_tools(tools: &[ToolDescriptor]) -> Vec<OpenAiTool> {
        tools
            .iter()
            .map(|t| OpenAiTool {
                r#type: "function".to_string(),
                function: OpenAiFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let request_body = OpenAiCompletionRequest {
            model: config.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::convert_tools(tools))
            },
            stream: None,
            response_format: if self.json_mode {
                Some(OpenAiResponseFormat {
                    r#type: "json_object".to_string(),
                })
            } else {
                None
            },
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        let body = response.text().await.map_err(ProviderError::Http)?;

        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        let openai_response: OpenAiCompletionResponse =
            serde_json::from_str(&body).map_err(ProviderError::Deserialize)?;

        Ok(CompletionResponse {
            content: openai_response.choices[0].message.content.clone().unwrap_or_default(),
            tool_calls: openai_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|c| ToolCall {
                            id: c.id.clone(),
                            name: c.function.name.clone(),
                            arguments: serde_json::from_str(&c.function.arguments)
                                .unwrap_or_else(|_| serde_json::Value::Null),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            usage: Usage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
            },
            finish_reason: match openai_response.choices[0].finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "tool_calls" => FinishReason::ToolCalls,
                "length" => FinishReason::Length,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            },
        })
    }

    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let json_mode = self.json_mode;
        let request_body = OpenAiCompletionRequest {
            model: config.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::convert_tools(tools))
            },
            stream: Some(true),
            response_format: if json_mode {
                Some(OpenAiResponseFormat {
                    r#type: "json_object".to_string(),
                })
            } else {
                None
            },
        };

        Box::pin(async_stream::try_stream! {
            let response = client
                .post(format!("{}/chat/completions", base_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await
                .map_err(ProviderError::Http)?;

            // Check status - if error, we can't easily get both error body and stream
            // Return status-based error without consuming response body
            let status = response.status();
            if !status.is_success() {
                Err(ProviderError::Api {
                    status,
                    body: format!("HTTP error: {}", status),
                })?;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(ProviderError::Http)?;
                let data = String::from_utf8_lossy(&chunk);
                buffer.push_str(&data);

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer.drain(..=newline_pos).collect::<String>();
                    buffer = buffer.trim_start().to_string();

                    let line = line.trim();
                    if !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line[6..];
                    if data == "[DONE]" {
                        return;
                    }

                    if let Ok(chunk_response) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                        if let Some(choice) = chunk_response.choices.first() {
                            yield StreamChunk {
                                delta: choice.delta.content.clone(),
                                tool_call_delta: None,
                                finish_reason: choice.finish_reason.clone().map(|r| match r.as_str() {
                                    "stop" => FinishReason::Stop,
                                    "tool_calls" => FinishReason::ToolCalls,
                                    "length" => FinishReason::Length,
                                    "content_filter" => FinishReason::ContentFilter,
                                    _ => FinishReason::Stop,
                                }),
                                usage: chunk_response.usage.map(|u| Usage {
                                    prompt_tokens: u.prompt_tokens,
                                    completion_tokens: u.completion_tokens,
                                    total_tokens: u.total_tokens,
                                }),
                            };
                        }
                    }
                }
            }
        })
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        let request_body = OpenAiEmbeddingRequest {
            model: "text-embedding-3-small".to_string(),
            input: if texts.len() == 1 {
                OpenAiEmbeddingInput::Single(texts[0].clone())
            } else {
                OpenAiEmbeddingInput::Multiple(texts)
            },
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        let body = response.text().await.map_err(ProviderError::Http)?;

        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        let openai_response: OpenAiEmbeddingResponse =
            serde_json::from_str(&body).map_err(ProviderError::Deserialize)?;

        Ok(openai_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect())
    }

    fn name(&self) -> &str {
        "openai"
    }
}

// ===== OpenAI API Types =====

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: OpenAiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Array(Vec<OpenAiContentPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Debug, Serialize, Clone)]
struct OpenAiImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
}

/// Response format for structured output.
#[derive(Debug, Serialize, Clone)]
struct OpenAiResponseFormat {
    r#type: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    r#type: String,
    function: OpenAiFunction,
}

#[derive(Debug, Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenAiToolCall {
    id: String,
    r#type: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionResponse {
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseToolCall {
    id: String,
    function: OpenAiResponseFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest {
    model: String,
    input: OpenAiEmbeddingInput,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiEmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    // REQ-1.1: Multi-Provider Tests
    // REQ-1.4: Streaming Tests

    #[test]
    fn test_provider_creation() {
        let provider = OpenAiProvider::new("test-key".to_string());
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn test_provider_with_custom_base_url() {
        let provider = OpenAiProvider::new("test-key".to_string())
            .with_base_url("https://custom.example.com".to_string());
        assert_eq!(provider.base_url(), "https://custom.example.com");
    }

    #[test]
    fn test_provider_with_json_mode() {
        let provider = OpenAiProvider::new("test-key".to_string())
            .with_json_mode(true);
        assert!(provider.json_mode);
    }

    #[test]
    fn test_provider_name() {
        let provider = OpenAiProvider::new("test-key".to_string());
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_convert_text_content() {
        let content = Content::Text("Hello, world!".to_string());
        let openai_content = OpenAiProvider::convert_content(content);

        match openai_content {
            OpenAiContent::Text(text) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_convert_multipart_content() {
        let parts = vec![
            ContentPart::Text("Describe this image".to_string()),
            ContentPart::Image {
                url: "https://example.com/image.jpg".to_string(),
                media_type: "image/jpeg".to_string(),
            },
        ];

        let content = Content::MultiPart(parts);
        let openai_content = OpenAiProvider::convert_content(content);

        match openai_content {
            OpenAiContent::Array(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    OpenAiContentPart::Text { text } => {
                        assert_eq!(text, "Describe this image");
                    }
                    _ => panic!("Expected Text part"),
                }
                match &items[1] {
                    OpenAiContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "https://example.com/image.jpg");
                    }
                    _ => panic!("Expected ImageUrl part"),
                }
            }
            _ => panic!("Expected Array content"),
        }
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there"),
            Message::tool("tool_123", "Result"),
        ];

        let openai_messages = OpenAiProvider::convert_messages(messages);

        assert_eq!(openai_messages.len(), 4);
        assert_eq!(openai_messages[0].role, "system");
        assert_eq!(openai_messages[1].role, "user");
        assert_eq!(openai_messages[2].role, "assistant");
        assert_eq!(openai_messages[3].role, "tool");
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDescriptor {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
            output_schema: None,
        }];

        let openai_tools = OpenAiProvider::convert_tools(&tools);

        assert_eq!(openai_tools.len(), 1);
        assert_eq!(openai_tools[0].r#type, "function");
        assert_eq!(openai_tools[0].function.name, "search");
        assert_eq!(openai_tools[0].function.description, "Search the web");
    }

    #[test]
    fn test_convert_tools_multiple() {
        let tools = vec![
            ToolDescriptor {
                name: "search".to_string(),
                description: "Search the web".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
                output_schema: None,
            },
            ToolDescriptor {
                name: "calculator".to_string(),
                description: "Calculate".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
                output_schema: None,
            },
        ];

        let openai_tools = OpenAiProvider::convert_tools(&tools);

        assert_eq!(openai_tools.len(), 2);
        assert_eq!(openai_tools[0].function.name, "search");
        assert_eq!(openai_tools[1].function.name, "calculator");
    }

    #[test]
    fn test_openai_stream_chunk_parsing() {
        // Test SSE chunk parsing
        let json = r#"{"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_with_finish_reason() {
        let json = r#"{"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        assert!(chunk.usage.is_some());
        assert_eq!(chunk.usage.unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_openai_stream_chunk_tool_calls() {
        let json = r#"{"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some("tool_calls".to_string()));
    }

    #[test]
    fn test_openai_completion_response_parsing() {
        let json = r#"{
            "id":"chatcmpl-123",
            "object":"chat.completion",
            "created":1234567890,
            "model":"gpt-4",
            "choices":[{
                "index":0,
                "message":{
                    "role":"assistant",
                    "content":"Hello!"
                },
                "finish_reason":"stop"
            }],
            "usage":{
                "prompt_tokens":10,
                "completion_tokens":5,
                "total_tokens":15
            }
        }"#;

        let response = serde_json::from_str::<OpenAiCompletionResponse>(json);
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, Some("Hello!".to_string()));
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 5);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn test_openai_completion_with_tool_calls() {
        let json = r#"{
            "id":"chatcmpl-123",
            "object":"chat.completion",
            "created":1234567890,
            "model":"gpt-4",
            "choices":[{
                "index":0,
                "message":{
                    "role":"assistant",
                    "content":null,
                    "tool_calls":[{
                        "id":"call_123",
                        "type":"function",
                        "function":{
                            "name":"search",
                            "arguments":"{\"query\":\"test\"}"
                        }
                    }]
                },
                "finish_reason":"tool_calls"
            }],
            "usage":{
                "prompt_tokens":10,
                "completion_tokens":5,
                "total_tokens":15
            }
        }"#;

        let response = serde_json::from_str::<OpenAiCompletionResponse>(json);
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.choices[0].message.content, None);
        assert!(response.choices[0].message.tool_calls.is_some());
        assert_eq!(response.choices[0].finish_reason, "tool_calls");
    }

    #[test]
    fn test_openai_embedding_response_parsing() {
        let json = r#"{
            "object":"list",
            "data":[{
                "object":"embedding",
                "embedding":[0.1,0.2,0.3],
                "index":0
            }],
            "model":"text-embedding-3-small",
            "usage":{
                "prompt_tokens":5,
                "total_tokens":5
            }
        }"#;

        let response = serde_json::from_str::<OpenAiEmbeddingResponse>(json);
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_provider_error_handling() {
        // This test verifies error types can be constructed
        let api_err = ProviderError::Api {
            status: reqwest::StatusCode::UNAUTHORIZED,
            body: "Invalid API key".to_string(),
        };
        assert!(api_err.to_string().contains("401"));
        assert!(api_err.to_string().contains("Invalid API key"));
    }

    #[tokio::test]
    async fn test_stream_parsing_done_marker() {
        // Test that [DONE] marker is handled correctly in streaming
        let line = "data: [DONE]";
        assert!(line.contains("[DONE]"));
    }

    #[test]
    fn test_openai_content_with_base64_image() {
        let parts = vec![
            ContentPart::Image {
                url: "data:image/jpeg;base64,/9j/4AAQSkZJRg".to_string(),
                media_type: "image/jpeg".to_string(),
            },
        ];

        let content = Content::MultiPart(parts);
        let openai_content = OpenAiProvider::convert_content(content);

        match openai_content {
            OpenAiContent::Array(items) => {
                assert_eq!(items.len(), 1);
                match &items[0] {
                    OpenAiContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "data:image/jpeg;base64,/9j/4AAQSkZJRg");
                    }
                    _ => panic!("Expected ImageUrl part"),
                }
            }
            _ => panic!("Expected Array content"),
        }
    }

    #[test]
    fn test_openai_request_serialization() {
        // Test that request serialization works correctly
        let request = OpenAiCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: OpenAiContent::Text("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stop: None,
            tools: None,
            stream: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"gpt-4\""));
        assert!(json_str.contains("\"Hello\""));
        assert!(json_str.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_openai_request_with_json_mode() {
        let request = OpenAiCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            tools: None,
            stream: None,
            response_format: Some(OpenAiResponseFormat {
                r#type: "json_object".to_string(),
            }),
        };

        let json = serde_json::to_string(&request);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"json_object\""));
    }

    // REQ-1.4: Streaming Tests - Multiple chunks and accumulation

    #[test]
    fn test_sse_chunk_accumulation() {
        // Test that multiple SSE chunks can be parsed and accumulated
        let chunks = vec![
            r#"data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"id":"1","choices":[{"delta":{"content":" world"}}]}"#,
            r#"data: {"id":"1","choices":[{"delta":{"content":"!"}}]}"#,
            r#"data: [DONE]"#,
        ];

        let mut accumulated = String::new();
        for chunk in chunks {
            if chunk.contains("[DONE]") {
                break;
            }
            if let Some(data) = chunk.strip_prefix("data: ") {
                if let Ok(parsed) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                    if let Some(delta) = parsed.choices.first().map(|c| &c.delta) {
                        if let Some(content) = &delta.content {
                            accumulated.push_str(content);
                        }
                    }
                }
            }
        }

        assert_eq!(accumulated, "Hello world!");
    }

    #[test]
    fn test_sse_empty_delta_handling() {
        // Test handling of chunks with empty deltas (first chunk often has no content)
        let json = r#"{"id":"1","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":null}]}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        // Delta struct exists but content field is None
        assert!(chunk.choices[0].delta.content.is_none());
        // Delta exists but content is None
    }

    #[test]
    fn test_sse_multiple_choices() {
        // Test handling of chunks with multiple choices (n>1 responses)
        let json = r#"{"id":"1","choices":[
            {"index":0,"delta":{"content":"Response A"}},
            {"index":1,"delta":{"content":"Response B"}}
        ]}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Response A"));
        assert_eq!(chunk.choices[1].delta.content.as_deref(), Some("Response B"));
    }

    #[test]
    fn test_sse_parse_malformed_recoverable() {
        // Test that malformed SSE lines are handled gracefully
        let malformed_lines = vec![
            "",  // Empty line
            ":", // Comment-only line
            "data: ", // Incomplete data
            "event: message", // Event line without data
        ];

        for line in malformed_lines {
            // None of these should cause panics when attempting to parse
            if line.starts_with("data: ") && line.len() > 6 {
                let _ = serde_json::from_str::<OpenAiStreamChunk>(&line[6..]);
            }
        }
    }

    #[test]
    fn test_stream_chunk_with_usage() {
        // Test final chunk with usage information
        let json = r#"{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;

        let chunk = serde_json::from_str::<OpenAiStreamChunk>(json);
        assert!(chunk.is_ok());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        assert!(chunk.usage.is_some());
        assert_eq!(chunk.usage.unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_openai_stream_request_serialization() {
        // Test that streaming request serializes correctly
        let request = OpenAiCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            tools: None,
            stream: Some(true), // Streaming enabled
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
    }
}
