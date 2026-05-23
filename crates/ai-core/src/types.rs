use serde::{Deserialize, Serialize};

/// Configuration for model generation parameters.
///
/// This struct defines per-request settings that control LLM output behavior.
/// Use the builder methods to construct configuration, or start from defaults.
///
/// ## Example
///
/// ```rust
/// use ai_core::ModelConfig;
///
/// let config = ModelConfig::new("gpt-4")
///     .with_temperature(0.7)
///     .with_max_tokens(1000)
///     .with_stop_sequences(vec!["END".to_string()]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The model identifier (e.g., "gpt-4", "claude-3-opus-20240229").
    pub model: String,

    /// Sampling temperature (0.0 to 2.0).
    ///
    /// Lower values make output more deterministic; higher values more random.
    pub temperature: Option<f64>,

    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,

    /// Nucleus sampling threshold (0.0 to 1.0).
    ///
    /// Controls the cumulative probability cutoff for token selection.
    pub top_p: Option<f64>,

    /// Penalty for repeating tokens (-2.0 to 2.0).
    ///
    /// Positive values reduce repetition; negative values encourage it.
    pub frequency_penalty: Option<f64>,

    /// Penalty for using tokens already in context (-2.0 to 2.0).
    ///
    /// Positive values encourage talking about new topics.
    pub presence_penalty: Option<f64>,

    /// Sequences that will stop generation when encountered.
    pub stop_sequences: Option<Vec<String>>,
}

impl ModelConfig {
    /// Create a new model config with just the model name.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
        }
    }

    /// Create a config for GPT-4 with sensible defaults.
    pub fn gpt4() -> Self {
        Self::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(4096)
    }

    /// Create a config for GPT-3.5 Turbo with sensible defaults.
    pub fn gpt35_turbo() -> Self {
        Self::new("gpt-3.5-turbo")
            .with_temperature(0.7)
            .with_max_tokens(4096)
    }

    /// Create a config for Claude 3 Opus with sensible defaults.
    pub fn claude_opus() -> Self {
        Self::new("claude-3-opus-20240229")
            .with_temperature(0.7)
            .with_max_tokens(4096)
    }

    /// Create a config for Claude 3 Sonnet with sensible defaults.
    pub fn claude_sonnet() -> Self {
        Self::new("claude-3-sonnet-20240229")
            .with_temperature(0.7)
            .with_max_tokens(4096)
    }

    /// Set the temperature.
    ///
    /// # Panics
    ///
    /// Panics if temperature is outside [0.0, 2.0].
    pub fn with_temperature(mut self, temp: f64) -> Self {
        assert!(
            (0.0..=2.0).contains(&temp),
            "temperature must be between 0.0 and 2.0"
        );
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum tokens to generate.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the top-p (nucleus sampling) threshold.
    ///
    /// # Panics
    ///
    /// Panics if top_p is outside [0.0, 1.0].
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&top_p),
            "top_p must be between 0.0 and 1.0"
        );
        self.top_p = Some(top_p);
        self
    }

    /// Set the frequency penalty.
    ///
    /// # Panics
    ///
    /// Panics if penalty is outside [-2.0, 2.0].
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        assert!(
            (-2.0..=2.0).contains(&penalty),
            "frequency_penalty must be between -2.0 and 2.0"
        );
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    ///
    /// # Panics
    ///
    /// Panics if penalty is outside [-2.0, 2.0].
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        assert!(
            (-2.0..=2.0).contains(&penalty),
            "presence_penalty must be between -2.0 and 2.0"
        );
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set stop sequences that will end generation.
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Create a new config that merges this config with overrides.
    ///
    /// Values from `other` take precedence over `self`. This is useful
    /// for applying per-request overrides to a base config.
    pub fn merge_with(&self, other: &ModelConfig) -> ModelConfig {
        ModelConfig {
            model: other.model.clone(),
            temperature: other.temperature.or(self.temperature),
            max_tokens: other.max_tokens.or(self.max_tokens),
            top_p: other.top_p.or(self.top_p),
            frequency_penalty: other.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            stop_sequences: other
                .stop_sequences
                .clone()
                .or_else(|| self.stop_sequences.clone()),
        }
    }

    /// Get a reference to this config with only the model name.
    ///
    /// Useful when you want to discard parameters and use provider defaults.
    pub fn with_model_only(&self) -> ModelConfig {
        ModelConfig::new(&self.model)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new("gpt-4")
    }
}

impl From<&str> for ModelConfig {
    fn from(model: &str) -> Self {
        Self::new(model)
    }
}

impl From<String> for ModelConfig {
    fn from(model: String) -> Self {
        Self::new(model)
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,

    /// The message content.
    pub content: Content,
}

impl Message {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Content::Text(content.into()),
        }
    }

    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Content::Text(content.into()),
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::Text(content.into()),
        }
    }

    /// Create a new tool message (result of tool execution).
    pub fn tool(_tool_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Content::Text(content.into()),
        }
    }
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    /// System message (sets agent behavior).
    System,

    /// User message (human input).
    User,

    /// Assistant message (LLM response).
    Assistant,

    /// Tool message (function call result).
    Tool,
}

/// Content of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    /// Plain text content.
    Text(String),

    /// Multi-modal content (text + images, etc.).
    MultiPart(Vec<ContentPart>),
}

impl Content {
    /// Create text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create multi-part content from parts.
    pub fn multi(parts: Vec<ContentPart>) -> Self {
        Self::MultiPart(parts)
    }

    /// Get the text content, if present.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get all text parts from multi-part content.
    pub fn get_text_parts(&self) -> Vec<&str> {
        match self {
            Content::Text(s) => vec![s.as_str()],
            Content::MultiPart(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Check if this content contains images.
    pub fn has_images(&self) -> bool {
        match self {
            Content::MultiPart(parts) => parts.iter().any(|p| p.is_image()),
            _ => false,
        }
    }

    /// Get all image parts from this content.
    pub fn get_images(&self) -> Vec<&ContentPart> {
        match self {
            Content::MultiPart(parts) => parts.iter().filter(|p| p.is_image()).collect(),
            _ => vec![],
        }
    }

    /// Convert image bytes to base64 data URL.
    pub fn encode_image_base64(data: &[u8], media_type: &str) -> String {
        use base64::prelude::*;
        format!(
            "data:{};base64,{}",
            media_type,
            BASE64_STANDARD.encode(data)
        )
    }

    /// Decode base64 data URL to bytes and media type.
    pub fn decode_image_data_url(url: &str) -> Result<(Vec<u8>, String), String> {
        if !url.starts_with("data:") {
            return Err("Not a data URL".to_string());
        }

        let parts = url
            .strip_prefix("data:")
            .ok_or("Invalid data URL")?
            .split_once(';')
            .ok_or("Invalid data URL: missing semicolon")?;

        let media_type = parts.0.to_string();
        let rest = parts.1;

        if !rest.starts_with("base64,") {
            return Err("Invalid data URL: not base64".to_string());
        }

        let base64_data = rest
            .strip_prefix("base64,")
            .ok_or("Invalid data URL: missing base64 data")?;

        use base64::prelude::*;
        let bytes = BASE64_STANDARD
            .decode(base64_data)
            .map_err(|e| format!("Failed to decode base64: {}", e))?;

        Ok((bytes, media_type))
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// A part of multi-part content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    /// Text part.
    Text(String),

    /// Image part.
    Image {
        /// URL or base64 data URL of the image.
        url: String,

        /// Media type (e.g., "image/jpeg", "image/png").
        media_type: String,

        /// Optional detail level for vision models ("low", "high", "auto").
        /// "low" = resized to fit within 512x512, "high" = full resolution
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    /// Image from bytes (base64-encoded internally).
    ImageBytes {
        /// Raw image bytes (will be base64-encoded).
        #[serde(skip_serializing)]
        data: Vec<u8>,

        /// Media type (e.g., "image/jpeg", "image/png").
        media_type: String,

        /// Optional detail level for vision models.
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

impl ContentPart {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create an image part from a URL.
    pub fn image_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Image {
            url: url.into(),
            media_type: media_type.into(),
            detail: None,
        }
    }

    /// Create an image part from a URL with detail level.
    pub fn image_url_with_detail(
        url: impl Into<String>,
        media_type: impl Into<String>,
        detail: ImageDetail,
    ) -> Self {
        Self::Image {
            url: url.into(),
            media_type: media_type.into(),
            detail: Some(detail.as_str().to_string()),
        }
    }

    /// Create an image part from base64 data.
    pub fn image_base64(base64_data: impl Into<String>, media_type: impl Into<String>) -> Self {
        let media_type = media_type.into();
        let data_url = format!("data:{};base64,{}", media_type, base64_data.into());
        Self::Image {
            url: data_url,
            media_type,
            detail: None,
        }
    }

    /// Create an image part from raw bytes.
    pub fn image_bytes(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::ImageBytes {
            data,
            media_type: media_type.into(),
            detail: None,
        }
    }

    /// Create an image part from raw bytes with detail level.
    pub fn image_bytes_with_detail(
        data: Vec<u8>,
        media_type: impl Into<String>,
        detail: ImageDetail,
    ) -> Self {
        Self::ImageBytes {
            data,
            media_type: media_type.into(),
            detail: Some(detail.as_str().to_string()),
        }
    }

    /// Check if this part is an image.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. } | Self::ImageBytes { .. })
    }

    /// Get the image URL if this is an image part.
    pub fn as_image_url(&self) -> Option<&str> {
        match self {
            Self::Image { url, .. } => Some(url),
            Self::ImageBytes { .. } => None,
            _ => None,
        }
    }
}

/// Detail level for vision models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDetail {
    /// Low detail (image resized to fit within 512x512).
    Low,

    /// High detail (full resolution, uses more tokens).
    High,

    /// Auto detail (model chooses based on image size).
    Auto,
}

impl ImageDetail {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Auto => "auto",
        }
    }
}

impl Message {
    /// Create a user message with multi-part content (text + images).
    pub fn user_multi(parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: Content::MultiPart(parts),
        }
    }

    /// Create a user message with text and a single image from URL.
    pub fn user_with_image(
        text: impl Into<String>,
        image_url: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self::user_multi(vec![
            ContentPart::text(text.into()),
            ContentPart::image_url(image_url, media_type),
        ])
    }

    /// Create a user message with text and multiple images.
    pub fn user_with_images(
        text: impl Into<String>,
        images: Vec<(String, String)>, // (url, media_type) pairs
    ) -> Self {
        let mut parts = vec![ContentPart::text(text.into())];
        for (url, media_type) in images {
            parts.push(ContentPart::image_url(url, media_type));
        }
        Self::user_multi(parts)
    }

    /// Create a user message with text and an image from bytes.
    pub fn user_with_image_bytes(
        text: impl Into<String>,
        image_data: Vec<u8>,
        media_type: impl Into<String>,
    ) -> Self {
        Self::user_multi(vec![
            ContentPart::text(text.into()),
            ContentPart::image_bytes(image_data, media_type),
        ])
    }

    /// Check if this message contains images.
    pub fn has_images(&self) -> bool {
        match &self.content {
            Content::MultiPart(parts) => parts.iter().any(|p| p.is_image()),
            _ => false,
        }
    }

    /// Get all image parts from this message.
    pub fn get_images(&self) -> Vec<&ContentPart> {
        match &self.content {
            Content::MultiPart(parts) => parts.iter().filter(|p| p.is_image()).collect(),
            _ => vec![],
        }
    }
}

/// Token usage information from an LLM response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,

    /// Number of tokens in the completion.
    pub completion_tokens: u32,

    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
}

impl Usage {
    /// Calculate estimated cost based on per-million-token pricing.
    ///
    /// # Arguments
    ///
    /// * `prompt_price_per_m` — Price per million prompt tokens
    /// * `completion_price_per_m` — Price per million completion tokens
    pub fn estimated_cost(&self, prompt_price_per_m: f64, completion_price_per_m: f64) -> f64 {
        let prompt_cost = (self.prompt_tokens as f64 / 1_000_000.0) * prompt_price_per_m;
        let completion_cost =
            (self.completion_tokens as f64 / 1_000_000.0) * completion_price_per_m;
        prompt_cost + completion_cost
    }
}

/// A complete LLM response.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The generated text content.
    pub content: String,

    /// Tool calls requested by the LLM.
    pub tool_calls: Vec<ToolCall>,

    /// Token usage information.
    pub usage: Usage,

    /// Why the generation ended.
    pub finish_reason: FinishReason,
}

/// A single tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Name of the tool to call.
    pub name: String,

    /// Arguments to pass to the tool (JSON object).
    pub arguments: serde_json::Value,
}

/// A chunk of a streaming response.
#[derive(Debug)]
pub struct StreamChunk {
    /// Delta text for this chunk.
    pub delta: Option<String>,

    /// Tool call delta (partial tool call).
    pub tool_call_delta: Option<ToolCallDelta>,

    /// Finish reason, if the stream is ending.
    pub finish_reason: Option<FinishReason>,

    /// Usage information (only in final chunk).
    pub usage: Option<Usage>,
}

/// Delta for a streaming tool call.
#[derive(Debug)]
pub struct ToolCallDelta {
    /// Tool call ID.
    pub id: Option<String>,

    /// Tool name.
    pub name: Option<String>,

    /// Partial arguments JSON.
    pub arguments_delta: Option<String>,
}

/// Why a generation finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    /// Natural stop (model chose to end).
    Stop,

    /// Tool calls were requested.
    ToolCalls,

    /// Max tokens limit reached.
    Length,

    /// Content filter triggered.
    ContentFilter,
}

/// Output from an agent execution.
#[derive(Debug)]
pub struct AgentOutput {
    /// The final content produced by the agent.
    pub content: String,

    /// Total token usage accumulated across the agent run.
    pub usage: Usage,

    /// Estimated total cost in USD for the agent run.
    pub estimated_cost: f64,

    /// Scopes updated while tracking this run.
    pub tracked_scopes: Vec<String>,
}

/// An event during agent streaming.
#[derive(Debug)]
pub enum AgentEvent {
    /// Text content was generated.
    Text(String),

    /// A tool call was initiated.
    ToolCall(ToolCall),

    /// A tool result was received.
    ToolResult {
        /// The ID of the tool call this result is for.
        call_id: String,

        /// The result content.
        content: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-1.2: ModelConfig Tests

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("gpt-4")
            .with_temperature(0.5)
            .with_max_tokens(2000)
            .with_top_p(0.9);

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(2000));
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn test_model_config_merge() {
        let base = ModelConfig::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000);

        let override_config = ModelConfig::new("gpt-4").with_temperature(0.5);

        let merged = base.merge_with(&override_config);

        // Override takes precedence for temperature
        assert_eq!(merged.temperature, Some(0.5));
        // Base value is preserved for max_tokens
        assert_eq!(merged.max_tokens, Some(1000));
    }

    #[test]
    fn test_model_config_merge_with_model_override() {
        let base = ModelConfig::new("gpt-4").with_temperature(0.7);

        let override_config = ModelConfig::new("claude-3-opus-20240229");

        let merged = base.merge_with(&override_config);

        // Model from override is always used
        assert_eq!(merged.model, "claude-3-opus-20240229");
        // Base temperature is preserved
        assert_eq!(merged.temperature, Some(0.7));
    }

    #[test]
    fn test_model_config_merge_preserves_all_fields() {
        let base = ModelConfig::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_top_p(0.9)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(0.3)
            .with_stop_sequences(vec!["END".to_string()]);

        let override_config = ModelConfig::new("gpt-4").with_temperature(0.5);

        let merged = base.merge_with(&override_config);

        assert_eq!(merged.temperature, Some(0.5));
        assert_eq!(merged.max_tokens, Some(1000));
        assert_eq!(merged.top_p, Some(0.9));
        assert_eq!(merged.frequency_penalty, Some(0.5));
        assert_eq!(merged.presence_penalty, Some(0.3));
        assert_eq!(merged.stop_sequences, Some(vec!["END".to_string()]));
    }

    #[test]
    fn test_model_config_presets_gpt4() {
        let config = ModelConfig::gpt4();
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(4096));
    }

    #[test]
    fn test_model_config_presets_gpt35_turbo() {
        let config = ModelConfig::gpt35_turbo();
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(4096));
    }

    #[test]
    fn test_model_config_presets_claude_opus() {
        let config = ModelConfig::claude_opus();
        assert_eq!(config.model, "claude-3-opus-20240229");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(4096));
    }

    #[test]
    fn test_model_config_presets_claude_sonnet() {
        let config = ModelConfig::claude_sonnet();
        assert_eq!(config.model, "claude-3-sonnet-20240229");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(4096));
    }

    #[test]
    fn test_model_config_with_model_only() {
        let config = ModelConfig::new("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_top_p(0.9);

        let model_only = config.with_model_only();

        assert_eq!(model_only.model, "gpt-4");
        assert_eq!(model_only.temperature, None);
        assert_eq!(model_only.max_tokens, None);
        assert_eq!(model_only.top_p, None);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.model, "gpt-4");
    }

    #[test]
    fn test_model_config_from_str() {
        let config: ModelConfig = "gpt-4".into();
        assert_eq!(config.model, "gpt-4");
    }

    #[test]
    fn test_model_config_from_string() {
        let config: ModelConfig = String::from("claude-3-opus").into();
        assert_eq!(config.model, "claude-3-opus");
    }

    #[test]
    #[should_panic(expected = "temperature must be between 0.0 and 2.0")]
    fn test_model_config_temperature_too_high() {
        ModelConfig::new("gpt-4").with_temperature(2.5);
    }

    #[test]
    #[should_panic(expected = "temperature must be between 0.0 and 2.0")]
    fn test_model_config_temperature_too_low() {
        ModelConfig::new("gpt-4").with_temperature(-0.1);
    }

    #[test]
    #[should_panic(expected = "top_p must be between 0.0 and 1.0")]
    fn test_model_config_top_p_too_high() {
        ModelConfig::new("gpt-4").with_top_p(1.5);
    }

    #[test]
    #[should_panic(expected = "top_p must be between 0.0 and 1.0")]
    fn test_model_config_top_p_negative() {
        ModelConfig::new("gpt-4").with_top_p(-0.1);
    }

    #[test]
    #[should_panic(expected = "frequency_penalty must be between -2.0 and 2.0")]
    fn test_model_config_frequency_penalty_too_high() {
        ModelConfig::new("gpt-4").with_frequency_penalty(2.5);
    }

    #[test]
    #[should_panic(expected = "frequency_penalty must be between -2.0 and 2.0")]
    fn test_model_config_frequency_penalty_too_low() {
        ModelConfig::new("gpt-4").with_frequency_penalty(-2.5);
    }

    #[test]
    #[should_panic(expected = "presence_penalty must be between -2.0 and 2.0")]
    fn test_model_config_presence_penalty_too_high() {
        ModelConfig::new("gpt-4").with_presence_penalty(2.5);
    }

    #[test]
    #[should_panic(expected = "presence_penalty must be between -2.0 and 2.0")]
    fn test_model_config_presence_penalty_too_low() {
        ModelConfig::new("gpt-4").with_presence_penalty(-2.5);
    }

    #[test]
    fn test_model_config_boundary_values() {
        // Test boundary values that should NOT panic
        let _ = ModelConfig::new("gpt-4")
            .with_temperature(0.0)
            .with_temperature(2.0)
            .with_top_p(0.0)
            .with_top_p(1.0)
            .with_frequency_penalty(-2.0)
            .with_frequency_penalty(2.0)
            .with_presence_penalty(-2.0)
            .with_presence_penalty(2.0);
    }

    #[test]
    fn test_message_constructors() {
        let user_msg = Message::user("hello");
        assert!(matches!(user_msg.role, Role::User));
        assert_eq!(user_msg.content.as_text(), Some("hello"));

        let system_msg = Message::system("You are helpful");
        assert!(matches!(system_msg.role, Role::System));
        assert_eq!(system_msg.content.as_text(), Some("You are helpful"));

        let assistant_msg = Message::assistant("Hello!");
        assert!(matches!(assistant_msg.role, Role::Assistant));
        assert_eq!(assistant_msg.content.as_text(), Some("Hello!"));

        let tool_msg = Message::tool("call_123", "Result");
        assert!(matches!(tool_msg.role, Role::Tool));
        assert_eq!(tool_msg.content.as_text(), Some("Result"));
    }

    #[test]
    fn test_content_text() {
        let content = Content::text("hello");
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn test_content_multi() {
        let parts = vec![
            ContentPart::Text("Hello".to_string()),
            ContentPart::Image {
                url: "https://example.com/image.jpg".to_string(),
                media_type: "image/jpeg".to_string(),
                detail: None,
            },
        ];
        let content = Content::multi(parts);
        assert!(matches!(content, Content::MultiPart(_)));
    }

    #[test]
    fn test_content_from_string() {
        let content: Content = String::from("hello").into();
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn test_content_from_str() {
        let content: Content = "hello".into();
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn test_usage_cost_estimation() {
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        // GPT-4 pricing (example)
        let cost = usage.estimated_cost(30.0, 60.0);
        assert!((cost - 0.06).abs() < 0.001); // ~6 cents
    }

    #[test]
    fn test_usage_cost_estimation_zero_tokens() {
        let usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        let cost = usage.estimated_cost(30.0, 60.0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_usage_cost_estimation_large_tokens() {
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 500_000,
            total_tokens: 1_500_000,
        };

        let cost = usage.estimated_cost(10.0, 20.0);
        assert_eq!(cost, 20.0); // $10 for prompts + $10 for completions
    }

    // REQ-8.1: Image Understanding Tests

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::text("Hello, world!");
        assert!(matches!(part, ContentPart::Text(_)));
        if let ContentPart::Text(s) = part {
            assert_eq!(s, "Hello, world!");
        }
    }

    #[test]
    fn test_content_part_image_url() {
        let part = ContentPart::image_url("https://example.com/image.jpg", "image/jpeg");

        assert!(part.is_image());
        assert_eq!(part.as_image_url(), Some("https://example.com/image.jpg"));

        if let ContentPart::Image {
            url,
            media_type,
            detail,
        } = part
        {
            assert_eq!(url, "https://example.com/image.jpg");
            assert_eq!(media_type, "image/jpeg");
            assert!(detail.is_none());
        } else {
            panic!("Expected Image variant");
        }
    }

    #[test]
    fn test_content_part_image_url_with_detail() {
        let part = ContentPart::image_url_with_detail(
            "https://example.com/image.jpg",
            "image/jpeg",
            ImageDetail::High,
        );

        assert!(part.is_image());

        if let ContentPart::Image {
            url,
            media_type,
            detail,
        } = part
        {
            assert_eq!(url, "https://example.com/image.jpg");
            assert_eq!(media_type, "image/jpeg");
            assert_eq!(detail, Some("high".to_string()));
        } else {
            panic!("Expected Image variant");
        }
    }

    #[test]
    fn test_content_part_image_base64() {
        let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let part = ContentPart::image_base64(base64_data, "image/png");

        assert!(part.is_image());

        if let ContentPart::Image {
            url, media_type, ..
        } = part
        {
            assert!(url.starts_with("data:image/png;base64,"));
            assert!(url.contains(base64_data));
            assert_eq!(media_type, "image/png");
        } else {
            panic!("Expected Image variant");
        }
    }

    #[test]
    fn test_content_part_image_bytes() {
        let bytes = vec![0x89, 0x50, 0x4E, 0x47]; // PNG signature
        let part = ContentPart::image_bytes(bytes.clone(), "image/png");

        assert!(part.is_image());

        if let ContentPart::ImageBytes {
            data, media_type, ..
        } = part
        {
            assert_eq!(data, bytes);
            assert_eq!(media_type, "image/png");
        } else {
            panic!("Expected ImageBytes variant");
        }
    }

    #[test]
    fn test_image_detail_as_str() {
        assert_eq!(ImageDetail::Low.as_str(), "low");
        assert_eq!(ImageDetail::High.as_str(), "high");
        assert_eq!(ImageDetail::Auto.as_str(), "auto");
    }

    #[test]
    fn test_message_user_with_image() {
        let msg = Message::user_with_image(
            "What's in this image?",
            "https://example.com/image.jpg",
            "image/jpeg",
        );

        assert!(matches!(msg.role, Role::User));
        assert!(msg.has_images());

        let images = msg.get_images();
        assert_eq!(images.len(), 1);
    }

    #[test]
    fn test_message_user_with_images() {
        let images = vec![
            (
                "https://example.com/img1.jpg".to_string(),
                "image/jpeg".to_string(),
            ),
            (
                "https://example.com/img2.png".to_string(),
                "image/png".to_string(),
            ),
        ];

        let msg = Message::user_with_images("Compare these images", images);

        assert!(matches!(msg.role, Role::User));
        assert!(msg.has_images());

        let images = msg.get_images();
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_message_user_with_image_bytes() {
        let bytes = vec![0x89, 0x50, 0x4E, 0x47];
        let msg = Message::user_with_image_bytes("What's in this image?", bytes, "image/png");

        assert!(matches!(msg.role, Role::User));
        assert!(msg.has_images());
    }

    #[test]
    fn test_content_has_images() {
        let text_content = Content::text("Just text");
        assert!(!text_content.has_images());

        let multi_content = Content::multi(vec![
            ContentPart::text("Text"),
            ContentPart::image_url("https://example.com/img.jpg", "image/jpeg"),
        ]);
        assert!(multi_content.has_images());
    }

    #[test]
    fn test_content_get_images() {
        let content = Content::multi(vec![
            ContentPart::text("Text"),
            ContentPart::image_url("https://example.com/img1.jpg", "image/jpeg"),
            ContentPart::image_url("https://example.com/img2.png", "image/png"),
        ]);

        let images = content.get_images();
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_content_get_text_parts() {
        let content = Content::multi(vec![
            ContentPart::text("First text"),
            ContentPart::image_url("https://example.com/img.jpg", "image/jpeg"),
            ContentPart::text("Second text"),
        ]);

        let texts = content.get_text_parts();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "First text");
        assert_eq!(texts[1], "Second text");
    }

    #[test]
    fn test_content_encode_image_base64() {
        let data = vec![0x89, 0x50, 0x4E, 0x47];
        let data_url = Content::encode_image_base64(&data, "image/png");

        assert!(data_url.starts_with("data:image/png;base64,"));
        assert!(data_url.contains("iVBORw"));
    }

    #[test]
    fn test_content_decode_image_data_url() {
        let data = vec![0x89, 0x50, 0x4E, 0x47];
        let data_url = Content::encode_image_base64(&data, "image/png");

        let (decoded, media_type) = Content::decode_image_data_url(&data_url).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(media_type, "image/png");
    }

    #[test]
    fn test_content_decode_invalid_data_url() {
        assert!(Content::decode_image_data_url("not a data url").is_err());
        assert!(Content::decode_image_data_url("data:text/plain").is_err());
        assert!(Content::decode_image_data_url("data:image/png;base64,invalid!@#").is_err());
    }

    #[test]
    fn test_message_multi_part_with_images() {
        let parts = vec![
            ContentPart::Text("Describe these images:".to_string()),
            ContentPart::Image {
                url: "https://example.com/img1.jpg".to_string(),
                media_type: "image/jpeg".to_string(),
                detail: Some("high".to_string()),
            },
            ContentPart::Image {
                url: "https://example.com/img2.jpg".to_string(),
                media_type: "image/jpeg".to_string(),
                detail: Some("high".to_string()),
            },
        ];

        let msg = Message::user_multi(parts);

        assert!(matches!(msg.role, Role::User));
        assert_eq!(msg.get_images().len(), 2);
        assert!(msg.has_images());
    }

    #[test]
    fn test_message_text_only_has_no_images() {
        let msg = Message::user("Just text, no images");
        assert!(!msg.has_images());
        assert_eq!(msg.get_images().len(), 0);
    }

    #[test]
    fn test_content_part_is_image() {
        assert!(!ContentPart::Text("text".to_string()).is_image());
        assert!(ContentPart::image_url("http://example.com/img.jpg", "image/jpeg").is_image());
        assert!(ContentPart::image_bytes(vec![1, 2, 3], "image/png").is_image());
    }
}
