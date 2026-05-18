// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Anthropic provider implementation for the AI framework.
//!
//! This crate provides a [`Provider`] implementation for Anthropic's Claude API,
//! supporting Claude 3 Opus, Sonnet, and Haiku models.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_provider_anthropic::AnthropicProvider;
//! use ai_core::{Provider, ModelConfig};
//! use ai_core::types::Message;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?);
//!
//!     let messages = vec![
//!         Message::user("Hello, Claude!")
//!     ];
//!
//!     let config = ModelConfig::claude_opus();
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

/// Anthropic API provider for Claude models.
///
/// Supports Claude 3 Opus, Sonnet, and Haiku with both text and vision capabilities.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    json_mode: bool,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given API key.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_anthropic::AnthropicProvider;
    /// let provider = AnthropicProvider::new("sk-ant-...".to_string());
    /// ```
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: "https://api.anthropic.com/v1".to_string(),
            json_mode: false,
        }
    }

    /// Enable JSON mode for structured output.
    ///
    /// When enabled, adds instructions for the model to respond with valid JSON.
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Convert framework messages to Anthropic format.
    fn convert_messages(messages: Vec<Message>) -> Vec<AnthropicMessage> {
        messages
            .into_iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    _ => "user".to_string(), // System messages are handled separately
                },
                content: Self::convert_content(m.content),
            })
            .collect()
    }

    /// Convert framework content to Anthropic content format.
    fn convert_content(content: Content) -> AnthropicContent {
        match content {
            Content::Text(text) => AnthropicContent::Text(text),
            Content::MultiPart(parts) => {
                let items: Vec<AnthropicContentPart> = parts
                    .into_iter()
                    .map(|p| match p {
                        ContentPart::Text(t) => {
                            AnthropicContentPart::Text { type_: "text".to_string(), text: t }
                        }
                        ContentPart::Image { url, .. } => {
                            // Extract base64 data if present
                            let (source, media_type) = if url.starts_with("data:") {
                                let parts: Vec<&str> = url.splitn(2, ':').collect();
                                let media_part = parts.get(1).unwrap_or(&"");
                                let (media_type, base64_data) = media_part
                                    .split_once(';')
                                    .unwrap_or(("image/jpeg", ""));
                                let base64_data = base64_data.strip_prefix("base64,").unwrap_or("");
                                (
                                    AnthropicSource {
                                        type_: "base64".to_string(),
                                        media_type: media_type.to_string(),
                                        data: base64_data.to_string(),
                                    },
                                    media_type.to_string(),
                                )
                            } else {
                                (
                                    AnthropicSource {
                                        type_: "url".to_string(),
                                        media_type: "image/jpeg".to_string(),
                                        data: url,
                                    },
                                    "image/jpeg".to_string(),
                                )
                            };
                            AnthropicContentPart::Image { type_: "image".to_string(), source }
                        }
                    })
                    .collect();
                AnthropicContent::Blocks(items)
            }
        }
    }

    /// Convert tool descriptors to Anthropic tool format.
    fn convert_tools(tools: &[ToolDescriptor]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        // Extract system message if present
        let (system_messages, messages): (Vec<Message>, Vec<Message>) = messages
            .into_iter()
            .partition(|m| matches!(m.role, Role::System));

        let base_system = system_messages
            .first()
            .map(|m| m.content.as_text().unwrap_or_default().to_string());

        // Add JSON mode instructions if enabled
        let system = if self.json_mode {
            let json_instruction = "\n\nYou must respond with valid JSON only. Do not include any additional text or explanation outside the JSON structure.";
            if let Some(base) = base_system {
                Some(base + json_instruction)
            } else {
                Some(json_instruction.to_string())
            }
        } else {
            base_system
        };

        let request_body = AnthropicMessageRequest {
            model: config.model.clone(),
            messages: Self::convert_messages(messages),
            system,
            max_tokens: config.max_tokens.unwrap_or(4096),
            temperature: config.temperature,
            top_p: config.top_p,
            stop_sequences: config.stop_sequences.clone(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::convert_tools(tools))
            },
            stream: None,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
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

        let anthropic_response: AnthropicMessageResponse =
            serde_json::from_str(&body).map_err(ProviderError::Deserialize)?;

        let content = &anthropic_response.content;

        let (text, tool_calls) = Self::extract_content_and_tools(content);

        Ok(CompletionResponse {
            content: text,
            tool_calls,
            usage: Usage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            },
            finish_reason: match anthropic_response.stop_reason.as_str() {
                "end_turn" => FinishReason::Stop,
                "max_tokens" => FinishReason::Length,
                "tool_use" => FinishReason::ToolCalls,
                "stop_sequence" => FinishReason::Stop,
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

        let (system_messages, messages): (Vec<Message>, Vec<Message>) = messages
            .into_iter()
            .partition(|m| matches!(m.role, Role::System));

        let base_system = system_messages
            .first()
            .map(|m| m.content.as_text().unwrap_or_default().to_string());

        // Add JSON mode instructions if enabled
        let system = if json_mode {
            let json_instruction = "\n\nYou must respond with valid JSON only. Do not include any additional text or explanation outside the JSON structure.";
            if let Some(base) = base_system {
                Some(base + json_instruction)
            } else {
                Some(json_instruction.to_string())
            }
        } else {
            base_system
        };

        let request_body = AnthropicMessageRequest {
            model: config.model.clone(),
            messages: Self::convert_messages(messages),
            system,
            max_tokens: config.max_tokens.unwrap_or(4096),
            temperature: config.temperature,
            top_p: config.top_p,
            stop_sequences: config.stop_sequences.clone(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::convert_tools(tools))
            },
            stream: Some(true),
        };

        Box::pin(async_stream::try_stream! {
            let response = client
                .post(format!("{}/messages", base_url))
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await?;

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

                    if let Ok(chunk_response) = serde_json::from_str::<AnthropicStreamChunk>(data) {
                        if let Some(AnthropicStreamDelta::Text { text }) =
                            &chunk_response.delta
                        {
                            yield StreamChunk {
                                delta: Some(text.clone()),
                                tool_call_delta: None,
                                finish_reason: None,
                                usage: None,
                            };
                        }

                        if chunk_response.type_ == "message_stop" {
                            if let Some(usage) = chunk_response.message_usage {
                                yield StreamChunk {
                                    delta: None,
                                    tool_call_delta: None,
                                    finish_reason: Some(FinishReason::Stop),
                                    usage: Some(Usage {
                                        prompt_tokens: usage.input_tokens,
                                        completion_tokens: usage.output_tokens,
                                        total_tokens: usage.input_tokens + usage.output_tokens,
                                    }),
                                };
                            }
                        }
                    }
                }
            }
        })
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        // Anthropic doesn't currently have an embeddings API
        // This will need to use a different provider or local model
        Err(ProviderError::Api {
            status: reqwest::StatusCode::NOT_IMPLEMENTED,
            body: "Anthropic does not provide an embeddings API. Please use OpenAI, Cohere, or a local model.".to_string(),
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

impl AnthropicProvider {
    fn extract_content_and_tools(
        content: &[AnthropicResponseContent],
    ) -> (String, Vec<ToolCall>) {
        let mut text = String::new();
        let mut tool_calls = Vec::new();

        for item in content {
            match item {
                AnthropicResponseContent::Text { text: t } => {
                    text.push_str(t);
                }
                AnthropicResponseContent::ToolUse {
                    id, name, input, ..
                } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
                _ => {}
            }
        }

        (text, tool_calls)
    }
}

// ===== Anthropic API Types =====

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentPart {
    Text { type_: String, text: String },
    Image { type_: String, source: AnthropicSource },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicSource {
    type_: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageResponse {
    content: Vec<AnthropicResponseContent>,
    usage: AnthropicUsage,
    stop_reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseContent {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
    Image { source: AnthropicSource },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamChunk {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<AnthropicStreamDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message_usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicStreamDelta {
    Text { text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AnthropicProvider::new("sk-ant-test".to_string());
        assert_eq!(provider.name(), "anthropic");
    }

    #[test]
    fn test_provider_with_custom_base_url() {
        let provider = AnthropicProvider::new("sk-ant-test".to_string())
            .with_base_url("https://custom.anthropic.com".to_string());
        assert_eq!(provider.base_url, "https://custom.anthropic.com");
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("sk-ant-test".to_string());
        assert_eq!(provider.name(), "anthropic");
    }

    #[test]
    fn test_convert_text_content() {
        let content = Content::Text("Hello, Claude!".to_string());
        let anthropic_content = AnthropicProvider::convert_content(content);

        match anthropic_content {
            AnthropicContent::Text(text) => {
                assert_eq!(text, "Hello, Claude!");
            }
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_convert_multipart_content() {
        let parts = vec![
            ContentPart::Text("What's in this image?".to_string()),
            ContentPart::Image {
                url: "data:image/jpeg;base64,/9j/4AAQ".to_string(),
                media_type: "image/jpeg".to_string(),
            },
        ];

        let content = Content::MultiPart(parts);
        let anthropic_content = AnthropicProvider::convert_content(content);

        match anthropic_content {
            AnthropicContent::Blocks(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    AnthropicContentPart::Text { type_: t, text } => {
                        assert_eq!(t, "text");
                        assert_eq!(text, "What's in this image?");
                    }
                    _ => panic!("Expected Text part"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there"),
        ];

        let anthropic_messages = AnthropicProvider::convert_messages(messages);

        assert_eq!(anthropic_messages.len(), 2);
        assert_eq!(anthropic_messages[0].role, "user");
        assert_eq!(anthropic_messages[1].role, "assistant");
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![ToolDescriptor {
            name: "calculator".to_string(),
            description: "Perform calculations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }),
            output_schema: None,
        }];

        let anthropic_tools = AnthropicProvider::convert_tools(&tools);

        assert_eq!(anthropic_tools.len(), 1);
        assert_eq!(anthropic_tools[0].name, "calculator");
        assert_eq!(anthropic_tools[0].description, "Perform calculations");
    }

    #[test]
    fn test_extract_content_and_tools() {
        let content = vec![
            AnthropicResponseContent::Text {
                text: "I'll search for that.".to_string(),
            },
            AnthropicResponseContent::ToolUse {
                id: "tool_123".to_string(),
                name: "search".to_string(),
                input: serde_json::json!({"query": "test"}),
            },
        ];

        let (text, tool_calls) = AnthropicProvider::extract_content_and_tools(&content);

        assert_eq!(text, "I'll search for that.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tool_123");
        assert_eq!(tool_calls[0].name, "search");
    }

    #[tokio::test]
    async fn test_embed_not_implemented() {
        let provider = AnthropicProvider::new("sk-ant-test".to_string());
        let result = provider.embed(vec!["test".to_string()]).await;

        match result {
            Err(ProviderError::Api { status, .. }) => {
                assert_eq!(status, reqwest::StatusCode::NOT_IMPLEMENTED);
            }
            _ => panic!("Expected NOT_IMPLEMENTED error"),
        }
    }
}
