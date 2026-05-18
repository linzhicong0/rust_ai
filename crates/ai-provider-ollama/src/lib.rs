// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Ollama provider implementation for the AI framework.
//!
//! This crate provides a [`Provider`] implementation for Ollama,
//! enabling use of local LLMs like Llama 3, Mistral, and others.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_provider_ollama::OllamaProvider;
//! use ai_core::{Provider, ModelConfig};
//! use ai_core::types::Message;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = OllamaProvider::new();
//!
//!     let messages = vec![
//!         Message::user("Hello!")
//!     ];
//!
//!     let config = ModelConfig::new("llama3:8b");
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
    CompletionResponse, FinishReason, Message, ModelConfig, Role, StreamChunk, Usage,
};

/// Ollama provider for local LLMs.
///
/// Ollama runs models locally and provides an OpenAI-compatible API.
/// This provider supports chat completions, streaming, and embeddings.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    json_mode: bool,
}

impl OllamaProvider {
    /// Create a new Ollama provider connecting to localhost:11434.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_ollama::OllamaProvider;
    /// let provider = OllamaProvider::new();
    /// ```
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "http://localhost:11434/api".to_string(),
            json_mode: false,
        }
    }

    /// Enable JSON mode for structured output.
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// Set a custom base URL (useful for remote Ollama instances).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_provider_ollama::OllamaProvider;
    /// let provider = OllamaProvider::new()
    ///     .with_base_url("http://ollama-server:11434/api".to_string());
    /// ```
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Convert framework messages to Ollama format.
    fn convert_messages(messages: Vec<Message>) -> Vec<OllamaMessage> {
        messages
            .into_iter()
            .filter_map(|m| {
                // Ollama doesn't use system messages the same way
                // Include system messages as user messages for now
                if matches!(m.role, Role::Tool) {
                    None
                } else {
                    Some(OllamaMessage {
                        role: match m.role {
                            Role::System => "user".to_string(),
                            Role::User => "user".to_string(),
                            Role::Assistant => "assistant".to_string(),
                            Role::Tool => return None,
                        },
                        content: m.content.as_text().unwrap_or_default().to_string(),
                    })
                }
            })
            .collect()
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let mut messages = Self::convert_messages(messages);

        // Add JSON mode instructions if enabled
        if self.json_mode {
            messages.insert(0, OllamaMessage {
                role: "system".to_string(),
                content: "You must respond with valid JSON only. Do not include any additional text or explanation outside the JSON structure.".to_string(),
            });
        }

        let request_body = OllamaChatRequest {
            model: config.model.clone(),
            messages,
            stream: Some(false),
            options: Some(OllamaOptions {
                temperature: config.temperature,
                num_predict: config.max_tokens,
                top_p: config.top_p,
                stop: config.stop_sequences.clone(),
            }),
        };

        let response = self
            .client
            .post(format!("{}/chat", self.base_url))
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

        let ollama_response: OllamaChatResponse =
            serde_json::from_str(&body).map_err(ProviderError::Deserialize)?;

        Ok(CompletionResponse {
            content: ollama_response.message.content,
            tool_calls: Vec::new(), // Ollama doesn't support tool calling in the same way
            usage: Usage {
                prompt_tokens: ollama_response.prompt_eval_count.unwrap_or(0),
                completion_tokens: ollama_response.eval_count.unwrap_or(0),
                total_tokens: ollama_response
                    .prompt_eval_count
                    .unwrap_or(0)
                    + ollama_response.eval_count.unwrap_or(0),
            },
            finish_reason: FinishReason::Stop,
        })
    }

    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let json_mode = self.json_mode;

        let mut messages = Self::convert_messages(messages);

        // Add JSON mode instructions if enabled
        if json_mode {
            messages.insert(0, OllamaMessage {
                role: "system".to_string(),
                content: "You must respond with valid JSON only. Do not include any additional text or explanation outside the JSON structure.".to_string(),
            });
        }

        let request_body = OllamaChatRequest {
            model: config.model.clone(),
            messages,
            stream: Some(true),
            options: Some(OllamaOptions {
                temperature: config.temperature,
                num_predict: config.max_tokens,
                top_p: config.top_p,
                stop: config.stop_sequences.clone(),
            }),
        };

        Box::pin(async_stream::try_stream! {
            let response = client
                .post(format!("{}/chat", base_url))
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

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(ProviderError::Http)?;
                let data = String::from_utf8_lossy(&chunk);

                // Ollama sends NDJSON (newline-delimited JSON)
                for line in data.lines() {
                    if let Ok(chunk_response) = serde_json::from_str::<OllamaStreamChunk>(line) {
                        if chunk_response.done {
                            yield StreamChunk {
                                delta: None,
                                tool_call_delta: None,
                                finish_reason: Some(FinishReason::Stop),
                                usage: chunk_response.usage.map(|u| Usage {
                                    prompt_tokens: u.prompt_eval_count.unwrap_or(0),
                                    completion_tokens: u.eval_count.unwrap_or(0),
                                    total_tokens: u.prompt_eval_count.unwrap_or(0) + u.eval_count.unwrap_or(0),
                                }),
                            };
                        } else if !chunk_response.message.content.is_empty() {
                            yield StreamChunk {
                                delta: Some(chunk_response.message.content.clone()),
                                tool_call_delta: None,
                                finish_reason: None,
                                usage: None,
                            };
                        }
                    }
                }
            }
        })
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        // For multiple texts, we need to make multiple requests (Ollama limitation)
        let mut embeddings = Vec::new();

        for text in texts {
            let request_body = OllamaEmbedRequest {
                model: "nomic-embed-text".to_string(),
                input: text,
            };

            let response = self
                .client
                .post(format!("{}/embed", self.base_url))
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

            let ollama_response: OllamaEmbedResponse =
                serde_json::from_str(&body).map_err(ProviderError::Deserialize)?;

            embeddings.push(ollama_response.embedding);
        }

        Ok(embeddings)
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// ===== Ollama API Types =====

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: OllamaResponseMessage,
    done: bool,
    usage: Option<OllamaUsage>,
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_provider_default() {
        let provider = OllamaProvider::default();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_provider_with_custom_base_url() {
        let provider = OllamaProvider::new()
            .with_base_url("http://remote-ollama:11434/api".to_string());
        assert_eq!(provider.base_url, "http://remote-ollama:11434/api");
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there"),
        ];

        let ollama_messages = OllamaProvider::convert_messages(messages);

        // System and user messages are both converted to "user" role
        // Tool messages are filtered out
        assert_eq!(ollama_messages.len(), 3);
        assert_eq!(ollama_messages[0].role, "user");
        assert_eq!(ollama_messages[1].role, "user");
        assert_eq!(ollama_messages[2].role, "assistant");
    }

    #[test]
    fn test_convert_messages_filters_tool_messages() {
        let messages = vec![
            Message::user("Hello"),
            Message::tool("tool_123", "Result"),
            Message::assistant("Done"),
        ];

        let ollama_messages = OllamaProvider::convert_messages(messages);

        // Tool message should be filtered out
        assert_eq!(ollama_messages.len(), 2);
        assert_eq!(ollama_messages[0].role, "user");
        assert_eq!(ollama_messages[1].role, "assistant");
    }

    #[test]
    fn test_provider_name() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.name(), "ollama");
    }
}
