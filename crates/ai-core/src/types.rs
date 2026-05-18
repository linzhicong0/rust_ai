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
            stop_sequences: other.stop_sequences.clone().or_else(|| self.stop_sequences.clone()),
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
    },
}

/// Token usage information from an LLM response.
#[derive(Debug)]
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
        let completion_cost = (self.completion_tokens as f64 / 1_000_000.0) * completion_price_per_m;
        prompt_cost + completion_cost
    }
}

/// A complete LLM response.
#[derive(Debug)]
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

        let override_config = ModelConfig::new("gpt-4")
            .with_temperature(0.5);

        let merged = base.merge_with(&override_config);

        // Override takes precedence for temperature
        assert_eq!(merged.temperature, Some(0.5));
        // Base value is preserved for max_tokens
        assert_eq!(merged.max_tokens, Some(1000));
    }

    #[test]
    fn test_message_constructors() {
        let user_msg = Message::user("hello");
        assert_eq!(user_msg.role, Role::User);

        let system_msg = Message::system("You are helpful");
        assert_eq!(system_msg.role, Role::System);
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
}
