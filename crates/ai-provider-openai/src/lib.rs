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

    #[tokio::test]
    async fn test_provider_error_handling() {
        // This test verifies error types can be constructed
        let http_err = reqwest::Error::from(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        ));
        let provider_err = ProviderError::Http(http_err);
        assert!(provider_err.to_string().contains("HTTP request failed"));

        let api_err = ProviderError::Api {
            status: reqwest::StatusCode::UNAUTHORIZED,
            body: "Invalid API key".to_string(),
        };
        assert!(api_err.to_string().contains("401"));
        assert!(api_err.to_string().contains("Invalid API key"));
    }
}
