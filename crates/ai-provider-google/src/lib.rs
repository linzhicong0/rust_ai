// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Google Gemini provider implementation for the AI framework.
//!
//! Supports the Google Gemini family of models (`gemini-1.5-pro`,
//! `gemini-1.5-flash`, `gemini-pro`, etc.) via the Gemini REST API.
//!
//! ## Authentication
//!
//! Set the `GOOGLE_API_KEY` environment variable (or `GEMINI_API_KEY`).
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_provider_google::GoogleProvider;
//! use ai_core::types::{Message, ModelConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = GoogleProvider::new("YOUR_API_KEY");
//!     let config = ModelConfig::new("gemini-1.5-flash");
//!     let messages = vec![Message::user("What is Rust?")];
//!
//!     use ai_core::provider::Provider;
//!     let response = provider.complete(messages, &config, &[]).await?;
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use ai_core::error::ProviderError;
use ai_core::provider::Provider;
use ai_core::tool::ToolDescriptor;
use ai_core::types::{
    CompletionResponse, Content, FinishReason, Message, ModelConfig, Role, StreamChunk, Usage,
};

const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google Gemini provider.
///
/// Supports all Gemini models via the `generateContent` and
/// `streamGenerateContent` REST endpoints.
pub struct GoogleProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GoogleProvider {
    /// Create a provider using the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: GEMINI_BASE_URL.to_string(),
        }
    }

    /// Create a provider reading the API key from the environment.
    ///
    /// Tries `GOOGLE_API_KEY` first, then `GEMINI_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns `ProviderError::Auth` if neither variable is set.
    pub fn from_env() -> Result<Self, ProviderError> {
        let key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| ProviderError::Api {
                status: StatusCode::UNAUTHORIZED,
                body: "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable".to_string(),
            })?;
        Ok(Self::new(key))
    }

    /// Override the base URL (useful for testing or proxies).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the endpoint URL for a given model and method.
    fn endpoint(&self, model: &str, method: &str) -> String {
        format!(
            "{}/models/{}:{}?key={}",
            self.base_url, model, method, self.api_key
        )
    }

    /// Convert framework messages to Gemini `contents` format.
    fn convert_messages(messages: Vec<Message>) -> (Option<String>, Vec<GeminiContent>) {
        let mut system_instruction: Option<String> = None;
        let mut contents: Vec<GeminiContent> = Vec::new();

        for msg in messages {
            let text = match &msg.content {
                Content::Text(t) => t.clone(),
                Content::MultiPart(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        ai_core::types::ContentPart::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };

            match msg.role {
                Role::System => {
                    // Gemini handles system instructions separately
                    system_instruction = Some(text);
                }
                Role::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text }],
                    });
                }
                Role::Assistant => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart::Text { text }],
                    });
                }
                Role::Tool => {
                    // Represent tool results as user messages
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text { text }],
                    });
                }
            }
        }

        (system_instruction, contents)
    }

    /// Convert framework tools to Gemini function declarations.
    fn convert_tools(tools: &[ToolDescriptor]) -> Option<Vec<GeminiTool>> {
        if tools.is_empty() {
            return None;
        }
        let declarations: Vec<GeminiFunctionDeclaration> = tools
            .iter()
            .map(|t| GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            })
            .collect();
        Some(vec![GeminiTool {
            function_declarations: declarations,
        }])
    }

    /// Map a Gemini finish reason string to the framework's enum.
    fn map_finish_reason(reason: Option<&str>) -> FinishReason {
        match reason {
            Some("STOP") => FinishReason::Stop,
            Some("MAX_TOKENS") => FinishReason::Length,
            Some("SAFETY") => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        }
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    /// Generate a completion using `generateContent`.
    #[instrument(skip(self, messages, config), fields(provider = "google", model = %config.model))]
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let (system_instruction, contents) = Self::convert_messages(messages);
        let gemini_tools = Self::convert_tools(tools);

        let generation_config = GeminiGenerationConfig {
            temperature: config.temperature.map(|t| t as f32),
            max_output_tokens: config.max_tokens,
            top_p: config.top_p.map(|p| p as f32),
            stop_sequences: config.stop_sequences.clone(),
        };

        let req_body = GeminiRequest {
            contents,
            system_instruction: system_instruction.map(|text| GeminiSystemInstruction {
                parts: vec![GeminiPart::Text { text }],
            }),
            generation_config: Some(generation_config),
            tools: gemini_tools,
        };

        let url = self.endpoint(&config.model, "generateContent");
        debug!(url = %url, "Sending Gemini generateContent request");

        let response = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api { status, body });
        }

        let gemini_resp: GeminiResponse = response.json().await.map_err(ProviderError::Http)?;

        // Extract the first candidate
        let candidate =
            gemini_resp
                .candidates
                .into_iter()
                .next()
                .ok_or_else(|| ProviderError::Api {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    body: "No candidates in Gemini response".to_string(),
                })?;

        // Collect text from all parts
        let content = candidate
            .content
            .parts
            .into_iter()
            .filter_map(|p| match p {
                GeminiPart::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        let finish_reason = Self::map_finish_reason(candidate.finish_reason.as_deref());

        let usage = gemini_resp
            .usage_metadata
            .map(|m| Usage {
                prompt_tokens: m.prompt_token_count,
                completion_tokens: m.candidates_token_count,
                total_tokens: m.total_token_count,
            })
            .unwrap_or(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });

        Ok(CompletionResponse {
            content,
            tool_calls: vec![],
            usage,
            finish_reason,
        })
    }

    /// Stream tokens using `streamGenerateContent`.
    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        let (system_instruction, contents) = Self::convert_messages(messages);

        let generation_config = GeminiGenerationConfig {
            temperature: config.temperature.map(|t| t as f32),
            max_output_tokens: config.max_tokens,
            top_p: config.top_p.map(|p| p as f32),
            stop_sequences: config.stop_sequences.clone(),
        };

        let req_body = GeminiRequest {
            contents,
            system_instruction: system_instruction.map(|text| GeminiSystemInstruction {
                parts: vec![GeminiPart::Text { text }],
            }),
            generation_config: Some(generation_config),
            tools: None,
        };

        let url = format!(
            "{}/models/{}:streamGenerateContent?key={}&alt=sse",
            self.base_url, config.model, self.api_key
        );
        let client = self.client.clone();

        Box::pin(stream! {
            let response = match client.post(&url).json(&req_body).send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(ProviderError::Http(e));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                yield Err(ProviderError::Api { status, body });
                return;
            }

            // Parse SSE stream: each event is a JSON GeminiResponse
            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(ProviderError::Http(e));
                        return;
                    }
                };

                let text = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        yield Err(ProviderError::Api {
                            status: StatusCode::INTERNAL_SERVER_ERROR,
                            body: e.to_string(),
                        });
                        return;
                    }
                };

                buffer.push_str(&text);

                // Process complete SSE events (separated by double newline)
                while let Some(event_end) = buffer.find("\n\n") {
                    let event = buffer[..event_end].to_string();
                    buffer = buffer[event_end + 2..].to_string();

                    // SSE data lines start with "data: "
                    for line in event.lines() {
                        let data = if let Some(d) = line.strip_prefix("data: ") {
                            d
                        } else {
                            continue;
                        };

                        if data == "[DONE]" {
                            return;
                        }

                        match serde_json::from_str::<GeminiResponse>(data) {
                            Ok(resp) => {
                                if let Some(candidate) = resp.candidates.into_iter().next() {
                                    let delta: String = candidate
                                        .content
                                        .parts
                                        .into_iter()
                                        .filter_map(|p| match p {
                                            GeminiPart::Text { text } => Some(text),
                                            _ => None,
                                        })
                                        .collect();

                                    let finish_reason = candidate
                                        .finish_reason
                                        .as_deref()
                                        .map(|r| Self::map_finish_reason(Some(r)));

                                    yield Ok(StreamChunk {
                                        delta: if delta.is_empty() { None } else { Some(delta) },
                                        tool_call_delta: None,
                                        finish_reason,
                                        usage: resp.usage_metadata.map(|m| Usage {
                                            prompt_tokens: m.prompt_token_count,
                                            completion_tokens: m.candidates_token_count,
                                            total_tokens: m.total_token_count,
                                        }),
                                    });
                                }
                            }
                            Err(_) => {
                                // Skip unparseable lines (comments, etc.)
                            }
                        }
                    }
                }
            }
        })
    }

    /// Generate text embeddings using the `embedContent` endpoint.
    #[instrument(skip(self, texts), fields(provider = "google", count = texts.len()))]
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        let model = "text-embedding-004";

        for text in texts {
            let url = self.endpoint(model, "embedContent");
            let body = GeminiEmbedRequest {
                model: format!("models/{model}"),
                content: GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart::Text { text }],
                },
            };

            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(ProviderError::Http)?;

            let status = response.status();
            if !status.is_success() {
                let err = response.text().await.unwrap_or_default();
                return Err(ProviderError::Api { status, body: err });
            }

            let embed_resp: GeminiEmbedResponse =
                response.json().await.map_err(ProviderError::Http)?;

            embeddings.push(embed_resp.embedding.values);
        }

        Ok(embeddings)
    }

    fn name(&self) -> &str {
        "google"
    }
}

// ─── Gemini API types ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiEmbedRequest {
    model: String,
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-1.1: Multi-Provider Tests — Google

    #[test]
    fn test_provider_name() {
        let provider = GoogleProvider::new("test-key");
        assert_eq!(provider.name(), "google");
    }

    #[test]
    fn test_endpoint_url() {
        let provider = GoogleProvider::new("my-key");
        let url = provider.endpoint("gemini-1.5-flash", "generateContent");
        assert!(url.contains("gemini-1.5-flash"));
        assert!(url.contains("generateContent"));
        assert!(url.contains("my-key"));
    }

    #[test]
    fn test_convert_messages_separates_system() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];
        let (system, contents) = GoogleProvider::convert_messages(messages);
        assert_eq!(system.as_deref(), Some("You are helpful."));
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[1].role, "model");
    }

    #[test]
    fn test_convert_messages_user_only() {
        let messages = vec![Message::user("What is Rust?")];
        let (system, contents) = GoogleProvider::convert_messages(messages);
        assert!(system.is_none());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
    }

    #[test]
    fn test_map_finish_reason() {
        assert!(matches!(
            GoogleProvider::map_finish_reason(Some("STOP")),
            FinishReason::Stop
        ));
        assert!(matches!(
            GoogleProvider::map_finish_reason(Some("MAX_TOKENS")),
            FinishReason::Length
        ));
        assert!(matches!(
            GoogleProvider::map_finish_reason(Some("SAFETY")),
            FinishReason::ContentFilter
        ));
        assert!(matches!(
            GoogleProvider::map_finish_reason(None),
            FinishReason::Stop
        ));
    }

    #[test]
    fn test_convert_tools_empty() {
        assert!(GoogleProvider::convert_tools(&[]).is_none());
    }

    #[test]
    fn test_convert_tools_non_empty() {
        use ai_core::tool::ToolDescriptor;
        use serde_json::json;

        let tools = vec![ToolDescriptor::new(
            "search",
            "Search the web",
            json!({"type": "object"}),
        )];
        let converted = GoogleProvider::convert_tools(&tools);
        assert!(converted.is_some());
        let tool_list = converted.unwrap();
        assert_eq!(tool_list.len(), 1);
        assert_eq!(tool_list[0].function_declarations[0].name, "search");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Ensure neither key is set for this test
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        let result = GoogleProvider::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_json_serialization_request() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text {
                    text: "hello".to_string(),
                }],
            }],
            system_instruction: None,
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: Some(1024),
                top_p: None,
                stop_sequences: None,
            }),
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_json_deserialization_response() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello, world!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }"#;

        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.candidates.len(), 1);
        assert_eq!(resp.candidates[0].finish_reason.as_deref(), Some("STOP"));
        let usage = resp.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 5);
    }
}
