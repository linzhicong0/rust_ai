//! Model registry for managing known AI models and their metadata.
//!
//! The model registry provides a centralized way to:
//! - Query model capabilities and limits
//! - Discover models by features
//! - Register custom/local models
//! - Access model metadata (context window, costs, etc.)
//!
//! ## Example
//!
//! ```rust,no_run
//! # use ai_core::model_registry::{ModelRegistry, ModelCapability};
//! # use ai_core::ModelConfig;
//!
//! // Get model info
//! let model = ModelRegistry::get("gpt-4").unwrap();
//! println!("Context window: {}", model.context_window);
//!
//! // Find models with streaming support
//! let streaming_models = ModelRegistry::find_by_capability(ModelCapability::Streaming);
//!
//! // Register a custom model
//! ModelRegistry::register_custom(
//!     "my-local-model",
//!     8192,
//!     vec![ModelCapability::Chat, ModelCapability::Streaming],
//! );
//! ```

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

/// Capabilities that a model may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    /// Text generation/chat
    Chat,
    /// Streaming responses
    Streaming,
    /// Function calling/tool use
    FunctionCalling,
    /// Image/vision input
    Vision,
    /// JSON output mode
    JsonMode,
    /// Embeddings generation
    Embeddings,
    /// Code generation/specialization
    Code,
}

/// Cost information for a model.
#[derive(Debug, Clone)]
pub struct ModelCost {
    /// Cost per million input tokens in USD
    pub input_per_million: f64,
    /// Cost per million output tokens in USD
    pub output_per_million: f64,
}

impl ModelCost {
    /// Create a new cost structure.
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
        }
    }

    /// Calculate cost for given token counts.
    pub fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }
}

/// Metadata about a specific model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
    pub id: String,
    /// Human-readable display name
    pub name: String,
    /// Provider that hosts this model
    pub provider: String,
    /// Maximum context window size in tokens
    pub context_window: u32,
    /// Supported capabilities
    pub capabilities: Vec<ModelCapability>,
    /// Cost information (if available)
    pub cost: Option<ModelCost>,
    /// Maximum output tokens
    pub max_output_tokens: u32,
    /// Whether this model is a default/recommended choice
    pub is_recommended: bool,
}

impl ModelInfo {
    /// Create new model info.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        provider: impl Into<String>,
        context_window: u32,
        capabilities: Vec<ModelCapability>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            provider: provider.into(),
            context_window,
            capabilities,
            cost: None,
            max_output_tokens: context_window / 2, // Default to half context window
            is_recommended: false,
        }
    }

    /// Add cost information.
    pub fn with_cost(mut self, cost: ModelCost) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Set maximum output tokens.
    pub fn with_max_output(mut self, max_output: u32) -> Self {
        self.max_output_tokens = max_output;
        self
    }

    /// Mark as recommended model.
    pub fn recommended(mut self) -> Self {
        self.is_recommended = true;
        self
    }

    /// Check if model supports a specific capability.
    pub fn has_capability(&self, capability: ModelCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// Calculate cost for a given usage.
    pub fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> Option<f64> {
        self.cost
            .as_ref()
            .map(|cost| cost.calculate_cost(input_tokens, output_tokens))
    }
}

/// Global model registry.
///
/// The registry maintains metadata about known models and allows
/// querying by capability, provider, or other criteria.
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Get the global registry instance.
    pub fn global() -> &'static RwLock<ModelRegistry> {
        static REGISTRY: Lazy<RwLock<ModelRegistry>> = Lazy::new(|| {
            let mut registry = ModelRegistry {
                models: HashMap::new(),
            };

            // Register known models
            registry.register_known_models();

            RwLock::new(registry)
        });

        &REGISTRY
    }

    /// Get model information by ID.
    pub fn get(model_id: &str) -> Option<ModelInfo> {
        Self::global()
            .read()
            .ok()
            .and_then(|registry| registry.models.get(model_id).cloned())
    }

    /// Find all models that support a specific capability.
    pub fn find_by_capability(capability: ModelCapability) -> Vec<ModelInfo> {
        Self::global()
            .read()
            .ok()
            .map(|registry| {
                registry
                    .models
                    .values()
                    .filter(|model| model.has_capability(capability))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all models from a specific provider.
    pub fn find_by_provider(provider: &str) -> Vec<ModelInfo> {
        Self::global()
            .read()
            .ok()
            .map(|registry| {
                registry
                    .models
                    .values()
                    .filter(|model| model.provider == provider)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all recommended models.
    pub fn recommended() -> Vec<ModelInfo> {
        Self::global()
            .read()
            .ok()
            .map(|registry| {
                registry
                    .models
                    .values()
                    .filter(|model| model.is_recommended)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all registered models.
    pub fn all() -> Vec<ModelInfo> {
        Self::global()
            .read()
            .ok()
            .map(|registry| registry.models.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Register a custom model.
    pub fn register_custom(
        id: impl Into<String>,
        name: impl Into<String>,
        provider: impl Into<String>,
        context_window: u32,
        capabilities: Vec<ModelCapability>,
    ) -> Result<(), ModelRegistryError> {
        Self::global()
            .write()
            .map_err(|_| ModelRegistryError::LockError)
            .map(|mut registry| {
                let model_info = ModelInfo::new(id, name, provider, context_window, capabilities);
                registry.models.insert(model_info.id.clone(), model_info);
            })
    }

    /// Check if a model ID is registered.
    pub fn is_registered(model_id: &str) -> bool {
        Self::global()
            .read()
            .ok()
            .map(|registry| registry.models.contains_key(model_id))
            .unwrap_or(false)
    }

    /// Register all known models.
    fn register_known_models(&mut self) {
        // OpenAI Models
        self.register_model(
            ModelInfo::new(
                "gpt-4",
                "GPT-4",
                "openai",
                8192,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::FunctionCalling,
                    ModelCapability::Streaming,
                ],
            )
            .with_cost(ModelCost::new(30.0, 60.0))
            .with_max_output(4096)
            .recommended(),
        );

        self.register_model(
            ModelInfo::new(
                "gpt-4-turbo",
                "GPT-4 Turbo",
                "openai",
                128000,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::FunctionCalling,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                    ModelCapability::JsonMode,
                ],
            )
            .with_cost(ModelCost::new(10.0, 30.0))
            .with_max_output(4096)
            .recommended(),
        );

        self.register_model(
            ModelInfo::new(
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo",
                "openai",
                16385,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::FunctionCalling,
                    ModelCapability::Streaming,
                ],
            )
            .with_cost(ModelCost::new(0.5, 1.5))
            .with_max_output(4096),
        );

        self.register_model(
            ModelInfo::new(
                "text-embedding-ada-002",
                "Text Embedding Ada 002",
                "openai",
                8191,
                vec![ModelCapability::Embeddings],
            )
            .with_cost(ModelCost::new(0.1, 0.0))
            .with_max_output(0), // Embeddings don't generate tokens
        );

        self.register_model(
            ModelInfo::new(
                "text-embedding-3-small",
                "Text Embedding 3 Small",
                "openai",
                8191,
                vec![ModelCapability::Embeddings],
            )
            .with_cost(ModelCost::new(0.02, 0.0))
            .with_max_output(0),
        );

        self.register_model(
            ModelInfo::new(
                "text-embedding-3-large",
                "Text Embedding 3 Large",
                "openai",
                8191,
                vec![ModelCapability::Embeddings],
            )
            .with_cost(ModelCost::new(0.13, 0.0))
            .with_max_output(0),
        );

        // Anthropic Models
        self.register_model(
            ModelInfo::new(
                "claude-3-opus-20240229",
                "Claude 3 Opus",
                "anthropic",
                200000,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                ],
            )
            .with_cost(ModelCost::new(15.0, 75.0))
            .with_max_output(4096)
            .recommended(),
        );

        self.register_model(
            ModelInfo::new(
                "claude-3-sonnet-20240229",
                "Claude 3 Sonnet",
                "anthropic",
                200000,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                ],
            )
            .with_cost(ModelCost::new(3.0, 15.0))
            .with_max_output(4096)
            .recommended(),
        );

        self.register_model(
            ModelInfo::new(
                "claude-3-haiku-20240307",
                "Claude 3 Haiku",
                "anthropic",
                200000,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                ],
            )
            .with_cost(ModelCost::new(0.25, 1.25))
            .with_max_output(4096),
        );

        // Google Models
        self.register_model(
            ModelInfo::new(
                "gemini-pro",
                "Gemini Pro",
                "google",
                91728,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::FunctionCalling,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                ],
            )
            .with_cost(ModelCost::new(0.5, 1.5))
            .with_max_output(2048),
        );

        self.register_model(
            ModelInfo::new(
                "gemini-ultra",
                "Gemini Ultra",
                "google",
                32768,
                vec![
                    ModelCapability::Chat,
                    ModelCapability::FunctionCalling,
                    ModelCapability::Streaming,
                    ModelCapability::Vision,
                ],
            )
            .with_cost(ModelCost::new(2.0, 8.0))
            .with_max_output(2048),
        );

        // Local/Custom Models placeholder
        self.register_model(
            ModelInfo::new(
                "llama-2-13b",
                "Llama 2 13B",
                "local",
                4096,
                vec![ModelCapability::Chat, ModelCapability::Streaming],
            )
            .with_max_output(2048),
        );
    }

    /// Internal method to register a model.
    fn register_model(&mut self, model: ModelInfo) {
        self.models.insert(model.id.clone(), model);
    }
}

/// Errors that can occur in the model registry.
#[derive(Debug, thiserror::Error)]
pub enum ModelRegistryError {
    /// Failed to acquire registry lock
    #[error("Failed to acquire registry lock")]
    LockError,

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid model configuration
    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_known_model() {
        let gpt4 = ModelRegistry::get("gpt-4").unwrap();
        assert_eq!(gpt4.id, "gpt-4");
        assert_eq!(gpt4.name, "GPT-4");
        assert_eq!(gpt4.provider, "openai");
        assert_eq!(gpt4.context_window, 8192);
        assert!(gpt4.has_capability(ModelCapability::Chat));
        assert!(gpt4.has_capability(ModelCapability::FunctionCalling));
    }

    #[test]
    fn test_get_unknown_model() {
        let unknown = ModelRegistry::get("unknown-model");
        assert!(unknown.is_none());
    }

    #[test]
    fn test_find_by_capability() {
        let chat_models = ModelRegistry::find_by_capability(ModelCapability::Chat);
        assert!(!chat_models.is_empty());

        let streaming_models = ModelRegistry::find_by_capability(ModelCapability::Streaming);
        assert!(!streaming_models.is_empty());

        let embedding_models = ModelRegistry::find_by_capability(ModelCapability::Embeddings);
        assert!(!embedding_models.is_empty());
    }

    #[test]
    fn test_find_by_provider() {
        let openai_models = ModelRegistry::find_by_provider("openai");
        assert!(!openai_models.is_empty());
        assert!(openai_models.iter().all(|m| m.provider == "openai"));

        let anthropic_models = ModelRegistry::find_by_provider("anthropic");
        assert!(!anthropic_models.is_empty());
        assert!(anthropic_models.iter().all(|m| m.provider == "anthropic"));
    }

    #[test]
    fn test_recommended_models() {
        let recommended = ModelRegistry::recommended();
        assert!(!recommended.is_empty());
        assert!(recommended.iter().all(|m| m.is_recommended));
    }

    #[test]
    fn test_is_registered() {
        assert!(ModelRegistry::is_registered("gpt-4"));
        assert!(ModelRegistry::is_registered("claude-3-opus-20240229"));
        assert!(!ModelRegistry::is_registered("unknown-model"));
    }

    #[test]
    fn test_register_custom_model() {
        let result = ModelRegistry::register_custom(
            "my-custom-model",
            "My Custom Model",
            "custom",
            16384,
            vec![ModelCapability::Chat, ModelCapability::Streaming],
        );

        assert!(result.is_ok());
        assert!(ModelRegistry::is_registered("my-custom-model"));

        let model = ModelRegistry::get("my-custom-model").unwrap();
        assert_eq!(model.id, "my-custom-model");
        assert_eq!(model.name, "My Custom Model");
        assert_eq!(model.provider, "custom");
        assert_eq!(model.context_window, 16384);
    }

    #[test]
    fn test_model_cost_calculation() {
        let cost = ModelCost::new(10.0, 30.0);
        let total = cost.calculate_cost(1000, 500);

        // 1000 input tokens * (10.0 / 1_000_000) = 0.01
        // 500 output tokens * (30.0 / 1_000_000) = 0.015
        // Total = 0.025
        assert!((total - 0.025).abs() < 0.0001);
    }

    #[test]
    fn test_model_info_cost() {
        let gpt4 = ModelRegistry::get("gpt-4").unwrap();
        let cost = gpt4.calculate_cost(1000, 500);

        assert!(cost.is_some());
        assert!(cost.unwrap() > 0.0);

        let embedding = ModelRegistry::get("text-embedding-ada-002").unwrap();
        let embedding_cost = embedding.calculate_cost(1000, 0);

        assert!(embedding_cost.is_some());
        assert!(embedding_cost.unwrap() > 0.0);
    }

    #[test]
    fn test_vision_capability() {
        let vision_models = ModelRegistry::find_by_capability(ModelCapability::Vision);
        assert!(!vision_models.is_empty());

        let gpt4_turbo = ModelRegistry::get("gpt-4-turbo").unwrap();
        assert!(gpt4_turbo.has_capability(ModelCapability::Vision));

        let claude = ModelRegistry::get("claude-3-opus-20240229").unwrap();
        assert!(claude.has_capability(ModelCapability::Vision));
    }

    #[test]
    fn test_all_models() {
        let all = ModelRegistry::all();
        assert!(!all.is_empty());
        assert!(all.len() >= 10); // Should have at least our known models
    }

    #[test]
    fn test_json_mode_capability() {
        let json_models = ModelRegistry::find_by_capability(ModelCapability::JsonMode);
        assert!(!json_models.is_empty());

        let gpt4_turbo = ModelRegistry::get("gpt-4-turbo").unwrap();
        assert!(gpt4_turbo.has_capability(ModelCapability::JsonMode));
    }
}
