// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! DeepSeek provider implementation for the AI framework.
//!
//! This crate provides a [`Provider`] implementation for DeepSeek's OpenAI-compatible
//! chat API and exposes provider-specific methods for DeepSeek-only features, including:
//! thinking mode, reasoning effort, JSON output, tool calls, chat prefix completion,
//! FIM completion, model listing, and balance lookup.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_core::{ModelConfig, Provider};
//! use ai_core::types::Message;
//! use ai_provider_deepseek::DeepSeekProvider;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = DeepSeekProvider::new(std::env::var("DEEPSEEK_API_KEY")?)
//!         .with_thinking_enabled(true)
//!         .with_reasoning_effort("high");
//!
//!     let response = provider
//!         .complete(vec![Message::user("Explain Rust ownership briefly.")], &ModelConfig::new("deepseek-v4-pro"), &[])
//!         .await?;
//!
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use ai_core::error::ProviderError;
use ai_core::provider::Provider;
use ai_core::tool::ToolDescriptor;
use ai_core::types::{
    CompletionResponse, Content, ContentPart, FinishReason, Message, ModelConfig, Role,
    StreamChunk, ToolCall, ToolCallDelta, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_BETA_BASE_URL: &str = "https://api.deepseek.com/beta";
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const FIM_COMPLETIONS_PATH: &str = "/completions";
const MODELS_PATH: &str = "/models";
const BALANCE_PATH: &str = "/user/balance";

/// DeepSeek API provider.
///
/// The generic [`Provider`] implementation covers the framework's portable surface.
/// For DeepSeek-specific features such as thinking mode transcripts, beta chat prefix
/// completion, FIM completion, model listing, and balance lookup, use the provider's
/// dedicated methods.
pub struct DeepSeekProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    beta_base_url: String,
    json_mode: bool,
    thinking: Option<DeepSeekThinking>,
    reasoning_effort: Option<String>,
    stream_options: Option<DeepSeekStreamOptions>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    user_id: Option<String>,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            beta_base_url: DEFAULT_BETA_BASE_URL.to_string(),
            json_mode: false,
            thinking: None,
            reasoning_effort: None,
            stream_options: None,
            logprobs: None,
            top_logprobs: None,
            user_id: None,
        }
    }

    /// Enable or disable JSON mode for generic provider calls.
    pub fn with_json_mode(mut self, enabled: bool) -> Self {
        self.json_mode = enabled;
        self
    }

    /// Set the standard DeepSeek base URL.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Set the DeepSeek beta base URL.
    pub fn with_beta_base_url(mut self, url: String) -> Self {
        self.beta_base_url = url;
        self
    }

    /// Configure the default thinking toggle for generic provider calls.
    pub fn with_thinking_enabled(mut self, enabled: bool) -> Self {
        self.thinking = Some(if enabled {
            DeepSeekThinking::enabled()
        } else {
            DeepSeekThinking::disabled()
        });
        self
    }

    /// Configure the default reasoning effort for generic provider calls.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Configure a default user identifier for generic provider calls.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Configure log probabilities for generic provider calls.
    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Configure top log probabilities for generic provider calls.
    pub fn with_top_logprobs(mut self, top_logprobs: u8) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }

    /// Ask DeepSeek to include usage in the final streamed chunk on generic provider calls.
    pub fn with_stream_include_usage(mut self, include_usage: bool) -> Self {
        let mut stream_options = self.stream_options.unwrap_or_default();
        stream_options.include_usage = Some(include_usage);
        self.stream_options = Some(stream_options);
        self
    }

    /// Returns the configured standard base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the configured beta base URL.
    pub fn beta_base_url(&self) -> &str {
        &self.beta_base_url
    }

    /// Execute a raw DeepSeek chat completion request.
    pub async fn chat_complete(
        &self,
        request: DeepSeekChatRequest,
    ) -> Result<DeepSeekChatCompletionResponse, ProviderError> {
        self.post_json(&self.url(&self.base_url, CHAT_COMPLETIONS_PATH), &request)
            .await
    }

    /// Execute a raw DeepSeek beta chat completion request.
    ///
    /// Use this for beta chat features such as chat prefix completion.
    pub async fn chat_complete_beta(
        &self,
        request: DeepSeekChatRequest,
    ) -> Result<DeepSeekChatCompletionResponse, ProviderError> {
        self.post_json(
            &self.url(&self.beta_base_url, CHAT_COMPLETIONS_PATH),
            &request,
        )
        .await
    }

    /// Stream a raw DeepSeek chat completion response.
    pub fn chat_stream_raw(
        &self,
        mut request: DeepSeekChatRequest,
    ) -> BoxStream<'static, Result<DeepSeekChatStreamChunk, ProviderError>> {
        request.stream = Some(true);
        self.post_sse_json(&self.url(&self.base_url, CHAT_COMPLETIONS_PATH), request)
    }

    /// Stream a raw beta DeepSeek chat completion response.
    pub fn chat_stream_beta_raw(
        &self,
        mut request: DeepSeekChatRequest,
    ) -> BoxStream<'static, Result<DeepSeekChatStreamChunk, ProviderError>> {
        request.stream = Some(true);
        self.post_sse_json(
            &self.url(&self.beta_base_url, CHAT_COMPLETIONS_PATH),
            request,
        )
    }

    /// Execute a DeepSeek beta FIM completion request.
    pub async fn fim_complete(
        &self,
        request: DeepSeekFimRequest,
    ) -> Result<DeepSeekFimCompletionResponse, ProviderError> {
        self.post_json(
            &self.url(&self.beta_base_url, FIM_COMPLETIONS_PATH),
            &request,
        )
        .await
    }

    /// Stream a DeepSeek beta FIM completion response.
    pub fn fim_stream(
        &self,
        mut request: DeepSeekFimRequest,
    ) -> BoxStream<'static, Result<DeepSeekFimStreamChunk, ProviderError>> {
        request.stream = Some(true);
        self.post_sse_json(
            &self.url(&self.beta_base_url, FIM_COMPLETIONS_PATH),
            request,
        )
    }

    /// List models available to the current API key.
    pub async fn list_models(&self) -> Result<DeepSeekModelListResponse, ProviderError> {
        self.get_json(&self.url(&self.base_url, MODELS_PATH)).await
    }

    /// Retrieve current account balance information.
    pub async fn get_user_balance(&self) -> Result<DeepSeekUserBalanceResponse, ProviderError> {
        self.get_json(&self.url(&self.base_url, BALANCE_PATH)).await
    }

    fn build_chat_request(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
        stream: Option<bool>,
    ) -> DeepSeekChatRequest {
        DeepSeekChatRequest {
            model: config.model.clone(),
            messages: Self::convert_messages(messages),
            thinking: self.thinking.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            max_tokens: config.max_tokens,
            response_format: if self.json_mode {
                Some(DeepSeekResponseFormat::json_object())
            } else {
                None
            },
            stop: config.stop_sequences.clone(),
            stream,
            stream_options: self.stream_options.clone(),
            temperature: config.temperature,
            top_p: config.top_p,
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::convert_tools(tools))
            },
            tool_choice: None,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            user_id: self.user_id.clone(),
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
        }
    }

    fn convert_messages(messages: Vec<Message>) -> Vec<DeepSeekMessage> {
        messages
            .into_iter()
            .map(|message| DeepSeekMessage {
                role: match message.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: Self::convert_content(message.content),
                prefix: None,
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            })
            .collect()
    }

    fn convert_content(content: Content) -> DeepSeekContent {
        match content {
            Content::Text(text) => DeepSeekContent::Text(text),
            Content::MultiPart(parts) => DeepSeekContent::Array(
                parts
                    .into_iter()
                    .map(|part| match part {
                        ContentPart::Text(text) => DeepSeekContentPart::Text { text },
                        ContentPart::Image { url, .. } => DeepSeekContentPart::ImageUrl {
                            image_url: DeepSeekImageUrl {
                                url,
                                detail: Some("auto".to_string()),
                            },
                        },
                    })
                    .collect(),
            ),
        }
    }

    fn convert_tools(tools: &[ToolDescriptor]) -> Vec<DeepSeekToolDefinition> {
        tools
            .iter()
            .map(|tool| DeepSeekToolDefinition {
                r#type: "function".to_string(),
                function: DeepSeekFunctionDefinition {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.input_schema.clone(),
                },
            })
            .collect()
    }

    fn completion_from_chat_response(
        response: DeepSeekChatCompletionResponse,
    ) -> Result<CompletionResponse, ProviderError> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Api {
                status: reqwest::StatusCode::BAD_GATEWAY,
                body: "DeepSeek response did not include any choices".to_string(),
            })?;

        Ok(CompletionResponse {
            content: choice.message.content.unwrap_or_default(),
            tool_calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: serde_json::from_str(&call.function.arguments)
                        .unwrap_or(serde_json::Value::Null),
                })
                .collect(),
            usage: response.usage.unwrap_or_default().into_usage(),
            finish_reason: map_finish_reason(choice.finish_reason.as_deref()),
        })
    }

    fn url(&self, base_url: &str, path: &str) -> String {
        format!("{}{}", base_url.trim_end_matches('/'), path)
    }

    async fn get_json<T>(&self, url: &str) -> Result<T, ProviderError>
    where
        T: DeserializeOwned,
    {
        let response = self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        let body = response.text().await.map_err(ProviderError::Http)?;
        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        serde_json::from_str(&body).map_err(ProviderError::Deserialize)
    }

    async fn post_json<T, B>(&self, url: &str, body: &B) -> Result<T, ProviderError>
    where
        T: DeserializeOwned,
        B: Serialize + ?Sized,
    {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(body)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        let body = response.text().await.map_err(ProviderError::Http)?;
        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        serde_json::from_str(&body).map_err(ProviderError::Deserialize)
    }

    fn post_sse_json<T, B>(
        &self,
        url: &str,
        body: B,
    ) -> BoxStream<'static, Result<T, ProviderError>>
    where
        T: DeserializeOwned + Send + 'static,
        B: Serialize + Send + Sync + 'static,
    {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = url.to_string();

        Box::pin(async_stream::try_stream! {
            let response = client
                .post(url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(ProviderError::Http)?;

            let status = response.status();
            let response = if status.is_success() {
                response
            } else {
                let body = response.text().await.map_err(ProviderError::Http)?;
                Err(ProviderError::Api { status, body })?
            };

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(ProviderError::Http)?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer.drain(..=newline_pos).collect::<String>();
                    let trimmed = line.trim();
                    if !trimmed.starts_with("data:") {
                        continue;
                    }

                    let data = trimmed.trim_start_matches("data:").trim();
                    if data == "[DONE]" {
                        return;
                    }

                    if data.is_empty() {
                        continue;
                    }

                    yield serde_json::from_str::<T>(data).map_err(ProviderError::Deserialize)?;
                }
            }
        })
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let response = self
            .chat_complete(self.build_chat_request(messages, config, tools, None))
            .await?;
        Self::completion_from_chat_response(response)
    }

    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        let stream =
            self.chat_stream_raw(self.build_chat_request(messages, config, tools, Some(true)));

        Box::pin(stream.map(|result| {
            result.map(|chunk| {
                let mut delta = None;
                let mut tool_call_delta = None;
                let mut finish_reason = None;

                if let Some(choice) = chunk.choices.first() {
                    delta = choice.delta.content.clone();
                    tool_call_delta = choice
                        .delta
                        .tool_calls
                        .as_ref()
                        .and_then(|calls| calls.first())
                        .map(|call| ToolCallDelta {
                            id: call.id.clone(),
                            name: call
                                .function
                                .as_ref()
                                .and_then(|function| function.name.clone()),
                            arguments_delta: call
                                .function
                                .as_ref()
                                .and_then(|function| function.arguments.clone()),
                        });
                    finish_reason = choice
                        .finish_reason
                        .as_deref()
                        .map(|reason| map_finish_reason(Some(reason)));
                }

                StreamChunk {
                    delta,
                    tool_call_delta,
                    finish_reason,
                    usage: chunk.usage.map(DeepSeekUsage::into_usage),
                }
            })
        }))
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        Err(ProviderError::Api {
            status: reqwest::StatusCode::NOT_IMPLEMENTED,
            body: "DeepSeek does not currently expose an embeddings endpoint in this provider. Use another embedding provider or extend this crate when DeepSeek adds one.".to_string(),
        })
    }

    fn name(&self) -> &str {
        "deepseek"
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason.unwrap_or("stop") {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

/// DeepSeek chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekChatRequest {
    pub model: String,
    pub messages: Vec<DeepSeekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<DeepSeekThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<DeepSeekResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<DeepSeekStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<DeepSeekToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<DeepSeekToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
}

impl DeepSeekChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<DeepSeekMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            thinking: None,
            reasoning_effort: None,
            max_tokens: None,
            response_format: None,
            stop: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            user_id: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

/// DeepSeek beta FIM completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekFimRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<DeepSeekStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
}

impl DeepSeekFimRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            echo: None,
            logprobs: None,
            max_tokens: None,
            stop: None,
            stream: None,
            stream_options: None,
            suffix: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

/// DeepSeek thinking mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekThinking {
    #[serde(rename = "type")]
    pub type_: String,
}

impl DeepSeekThinking {
    pub fn enabled() -> Self {
        Self {
            type_: "enabled".to_string(),
        }
    }

    pub fn disabled() -> Self {
        Self {
            type_: "disabled".to_string(),
        }
    }
}

/// DeepSeek response format selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekResponseFormat {
    #[serde(rename = "type")]
    pub type_: String,
}

impl DeepSeekResponseFormat {
    pub fn text() -> Self {
        Self {
            type_: "text".to_string(),
        }
    }

    pub fn json_object() -> Self {
        Self {
            type_: "json_object".to_string(),
        }
    }
}

/// DeepSeek stream options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeepSeekStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A DeepSeek chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekMessage {
    pub role: String,
    pub content: DeepSeekContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeepSeekToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl DeepSeekMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: DeepSeekContent::Text(content.into()),
            prefix: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: DeepSeekContent::Text(content.into()),
            prefix: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: DeepSeekContent::Text(content.into()),
            prefix: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_prefix(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: DeepSeekContent::Text(content.into()),
            prefix: Some(true),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: DeepSeekContent::Text(content.into()),
            prefix: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    pub fn with_reasoning_content(mut self, reasoning_content: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning_content.into());
        self
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<DeepSeekToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekContent {
    Text(String),
    Array(Vec<DeepSeekContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DeepSeekContentPart {
    Text { text: String },
    ImageUrl { image_url: DeepSeekImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// DeepSeek tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekToolDefinition {
    pub r#type: String,
    pub function: DeepSeekFunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// DeepSeek tool choice. This accepts either string modes such as `none` / `auto`
/// or a provider-specific object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekToolChoice {
    Mode(String),
    Object(serde_json::Value),
}

/// A tool call included in assistant messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekToolCall {
    pub id: String,
    pub r#type: String,
    pub function: DeepSeekFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// DeepSeek chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<DeepSeekChatChoice>,
    #[serde(default)]
    pub usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatChoice {
    #[serde(default)]
    pub index: Option<u32>,
    pub message: DeepSeekChatResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatResponseMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeepSeekToolCall>>,
}

/// Raw streamed chat completion chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatStreamChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<DeepSeekChatStreamChoice>,
    #[serde(default)]
    pub usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatStreamChoice {
    #[serde(default)]
    pub index: Option<u32>,
    pub delta: DeepSeekChatStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekChatStreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeepSeekToolCallDelta>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekToolCallDelta {
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<DeepSeekFunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekFunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// DeepSeek beta FIM completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekFimCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<DeepSeekFimChoice>,
    #[serde(default)]
    pub usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekFimChoice {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

/// Raw streamed FIM chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekFimStreamChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<DeepSeekFimStreamChoice>,
    #[serde(default)]
    pub usage: Option<DeepSeekUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekFimStreamChoice {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

/// DeepSeek model list response.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekModelListResponse {
    pub object: String,
    pub data: Vec<DeepSeekModel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekModel {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// DeepSeek user balance response.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekUserBalanceResponse {
    pub is_available: bool,
    #[serde(default)]
    pub balance_infos: Vec<DeepSeekBalanceInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekBalanceInfo {
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeepSeekUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

impl DeepSeekUsage {
    fn into_usage(self) -> Usage {
        let total_tokens = if self.total_tokens == 0 {
            self.prompt_tokens + self.completion_tokens
        } else {
            self.total_tokens
        };

        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_defaults() {
        let provider = DeepSeekProvider::new("test-key".to_string());
        assert_eq!(provider.base_url(), DEFAULT_BASE_URL);
        assert_eq!(provider.beta_base_url(), DEFAULT_BETA_BASE_URL);
        assert_eq!(provider.name(), "deepseek");
    }

    #[test]
    fn test_convert_text_content() {
        let content = DeepSeekProvider::convert_content(Content::Text("hello".to_string()));
        match content {
            DeepSeekContent::Text(text) => assert_eq!(text, "hello"),
            DeepSeekContent::Array(_) => panic!("expected text content"),
        }
    }

    #[test]
    fn test_assistant_prefix_message() {
        let message = DeepSeekMessage::assistant_prefix("```python\n");
        assert_eq!(message.role, "assistant");
        assert_eq!(message.prefix, Some(true));
    }

    #[test]
    fn test_build_chat_request_uses_provider_defaults() {
        let provider = DeepSeekProvider::new("test-key".to_string())
            .with_thinking_enabled(true)
            .with_reasoning_effort("max")
            .with_json_mode(true)
            .with_user_id("user-123")
            .with_logprobs(true)
            .with_top_logprobs(5)
            .with_stream_include_usage(true);

        let config = ModelConfig::new("deepseek-v4-pro")
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_top_p(0.9);

        let request =
            provider.build_chat_request(vec![Message::user("hello")], &config, &[], Some(true));

        assert_eq!(request.model, "deepseek-v4-pro");
        assert_eq!(request.reasoning_effort.as_deref(), Some("max"));
        assert_eq!(request.user_id.as_deref(), Some("user-123"));
        assert_eq!(request.logprobs, Some(true));
        assert_eq!(request.top_logprobs, Some(5));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stream_options.and_then(|opts| opts.include_usage),
            Some(true)
        );
        assert_eq!(request.response_format.unwrap().type_, "json_object");
    }

    #[test]
    fn test_usage_defaults_total_when_missing() {
        let usage = DeepSeekUsage {
            prompt_tokens: 11,
            completion_tokens: 7,
            total_tokens: 0,
        }
        .into_usage();

        assert_eq!(usage.total_tokens, 18);
    }
}
