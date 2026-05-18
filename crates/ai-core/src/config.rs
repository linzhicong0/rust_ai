//! Framework configuration management.
//!
//! The [`FrameworkConfig`] struct provides centralized configuration for
//! the AI framework, with support for layered loading (defaults → file →
//! environment variables → code overrides).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum FrameworkConfigError {
    /// Error from the underlying config crate.
    #[error("Configuration loading error: {0}")]
    Config(#[from] config::ConfigError),

    /// A referenced provider is not configured.
    #[error("Provider '{name}' is not configured")]
    MissingProvider {
        name: String,
    },

    /// The API key for a provider is missing.
    #[error("API key for provider '{provider}' not found. Set the {env_var} environment variable.")]
    MissingApiKey {
        provider: String,
        env_var: String,
    },
}

/// Global framework configuration.
///
/// This struct defines configuration for all framework components:
/// - Provider settings (API keys, base URLs, default models)
/// - Agent defaults (max iterations, temperature)
/// - Server settings (host, port for REST API)
///
/// ## Loading Configuration
///
/// Configuration is loaded in layers with later layers overriding earlier ones:
///
/// 1. **Defaults** — Built-in sensible defaults
/// 2. **File** — From `ai_framework.toml` or `.env`
/// 3. **Environment** — From `AI_*` environment variables
/// 4. **Code** — Programmatic overrides
///
/// ## Example
///
/// ```rust,no_run
/// use ai_core::config::FrameworkConfig;
///
/// // Load from file with environment overrides
/// let config = FrameworkConfig::load()
///     .with_default_provider("anthropic")
///     .with_default_model("claude-3-opus-20240229");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    /// Default provider to use when none is specified.
    pub default_provider: String,

    /// Default model to use when none is specified.
    pub default_model: String,

    /// Configuration for each provider (indexed by provider name).
    pub providers: HashMap<String, ProviderConfig>,

    /// Default agent behavior settings.
    pub agent: AgentConfig,

    /// REST API server settings.
    pub server: ServerConfig,
}

impl FrameworkConfig {
    /// Load configuration from environment and config files.
    ///
    /// This method attempts to load configuration from:
    /// - `ai_framework.toml` or `ai_framework.yaml` in the current directory
    /// - `.env` file for environment variables
    /// - Environment variables prefixed with `AI_`
    ///
    /// Configuration is loaded in layers with later layers overriding earlier ones:
    /// 1. **Defaults** — Built-in sensible defaults
    /// 2. **File** — From `ai_framework.toml` or `ai_framework.yaml`
    /// 3. **Environment** — From `.env` file and `AI_*` environment variables
    ///
    /// # Environment Variable Syntax
    ///
    /// Nested configuration uses double underscore separation:
    /// ```text
    /// AI_DEFAULT_PROVIDER=anthropic
    /// AI_AGENT__MAX_ITERATIONS=20
    /// AI_SERVER__PORT=3000
    /// AI_PROVIDERS__OPENAI__BASE_URL=https://api.openai.com/v1
    /// ```
    ///
    /// # Example config file (ai_framework.toml)
    ///
    /// ```toml
    /// default_provider = "anthropic"
    /// default_model = "claude-3-opus-20240229"
    ///
    /// [agent]
    /// max_iterations = 15
    /// default_temperature = 0.8
    ///
    /// [server]
    /// host = "0.0.0.0"
    /// port = 8080
    ///
    /// [providers.openai]
    /// api_key_env = "OPENAI_API_KEY"
    /// base_url = "https://api.openai.com/v1"
    ///
    /// [providers.anthropic]
    /// api_key_env = "ANTHROPIC_API_KEY"
    /// base_url = "https://api.anthropic.com"
    /// default_model = "claude-3-opus-20240229"
    /// ```
    pub fn load() -> Result<Self, FrameworkConfigError> {
        // Load .env file first (if exists) - this populates std::env
        dotenvy::dotenv().ok();

        // Build layered configuration
        let settings = config::Config::builder()
            // Layer 1: Start with defaults (via serde defaults)
            // Layer 2: Load from ai_framework.toml or ai_framework.yaml
            .add_source(config::File::with_name("ai_framework").required(false))
            // Layer 3: Environment variables with AI_ prefix
            // Use "__" to separate nested keys (e.g., AI_AGENT__MAX_ITERATIONS)
            .add_source(
                config::Environment::with_prefix("AI")
                    .prefix_separator("__")
                    .separator("__")
                    .try_parsing(true)
            )
            .build()?;

        // Deserialize into FrameworkConfig
        // serde defaults will apply if values are missing
        settings.try_deserialize()
            .map_err(FrameworkConfigError::from)
    }

    /// Load configuration from a specific file path.
    ///
    /// This allows loading configuration from a custom location.
    /// Environment variables and `.env` files will still be applied as overrides.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_core::config::FrameworkConfig;
    ///
    /// let config = FrameworkConfig::load_from_path("/etc/myapp/config.toml")?;
    /// ```
    pub fn load_from_path<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, FrameworkConfigError> {
        // Load .env file first (if exists)
        dotenvy::dotenv().ok();

        let settings = config::Config::builder()
            .add_source(config::File::from(path.as_ref()).required(false))
            .add_source(
                config::Environment::with_prefix("AI")
                    .prefix_separator("__")
                    .separator("__")
                    .try_parsing(true)
            )
            .build()?;

        settings.try_deserialize()
            .map_err(FrameworkConfigError::from)
    }

    /// Create a new config with defaults, ignoring any external files or environment.
    ///
    /// This is useful for testing or when you want complete control over configuration.
    pub fn from_defaults() -> Self {
        Self::default()
    }

    /// Set the default provider.
    pub fn with_default_provider(mut self, provider: impl Into<String>) -> Self {
        self.default_provider = provider.into();
        self
    }

    /// Set the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Add or update a provider configuration.
    pub fn with_provider(mut self, name: impl Into<String>, config: ProviderConfig) -> Self {
        self.providers.insert(name.into(), config);
        self
    }

    /// Update agent settings.
    pub fn with_agent_config(mut self, config: AgentConfig) -> Self {
        self.agent = config;
        self
    }

    /// Get provider configuration by name.
    ///
    /// Returns `None` if the provider is not configured.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_core::config::FrameworkConfig;
    ///
    /// let config = FrameworkConfig::load()?;
    /// if let Some(openai) = config.get_provider("openai") {
    ///     if let Some(api_key) = openai.api_key() {
    ///         println!("OpenAI API key found");
    ///     }
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    /// Check if a provider has a valid API key configured.
    ///
    /// Returns `true` if the provider exists and its API key environment
    /// variable is set and non-empty.
    pub fn has_provider_api_key(&self, name: &str) -> bool {
        self.get_provider(name)
            .and_then(|p| p.api_key())
            .map(|k| !k.is_empty())
            .unwrap_or(false)
    }

    /// Validate that the configuration is usable.
    ///
    /// Checks that:
    /// - The default provider is configured
    /// - The default provider has an API key set
    ///
    /// Returns `Ok(())` if valid, or an error describing what's missing.
    pub fn validate(&self) -> Result<(), FrameworkConfigError> {
        if !self.providers.contains_key(&self.default_provider) {
            return Err(FrameworkConfigError::MissingProvider {
                name: self.default_provider.clone(),
            });
        }

        if !self.has_provider_api_key(&self.default_provider) {
            return Err(FrameworkConfigError::MissingApiKey {
                provider: self.default_provider.clone(),
                env_var: self.providers[&self.default_provider].api_key_env.clone(),
            });
        }

        Ok(())
    }
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            default_provider: "openai".to_string(),
            default_model: "gpt-4".to_string(),
            providers: HashMap::new(),
            agent: AgentConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

/// Configuration for a specific LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Environment variable name containing the API key.
    pub api_key_env: String,

    /// Base URL for API requests.
    pub base_url: String,

    /// Default model for this provider (overrides framework default).
    pub default_model: Option<String>,
}

impl ProviderConfig {
    /// Create a new provider configuration.
    pub fn new(
        api_key_env: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            api_key_env: api_key_env.into(),
            base_url: base_url.into(),
            default_model: None,
        }
    }

    /// Set the default model for this provider.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Retrieve the API key from the environment.
    ///
    /// Returns `None` if the environment variable is not set.
    pub fn api_key(&self) -> Option<String> {
        std::env::var(&self.api_key_env).ok()
    }
}

/// Agent behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum iterations before an agent gives up.
    pub max_iterations: u32,

    /// Default temperature for model generation (0.0 to 2.0).
    pub default_temperature: f64,
}

impl AgentConfig {
    /// Create a new agent configuration.
    pub fn new(max_iterations: u32, default_temperature: f64) -> Self {
        Self {
            max_iterations,
            default_temperature,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            default_temperature: 0.7,
        }
    }
}

/// REST API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to.
    pub host: String,

    /// Port number to listen on.
    pub port: u16,
}

impl ServerConfig {
    /// Create a new server configuration.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    /// Get the full bind address.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FrameworkConfig::default();
        assert_eq!(config.default_provider, "openai");
        assert_eq!(config.default_model, "gpt-4");
        assert_eq!(config.agent.max_iterations, 10);
    }

    #[test]
    fn test_config_builder() {
        let config = FrameworkConfig::default()
            .with_default_provider("anthropic")
            .with_default_model("claude-3-opus-20240229");

        assert_eq!(config.default_provider, "anthropic");
        assert_eq!(config.default_model, "claude-3-opus-20240229");
    }

    #[test]
    fn test_provider_config() {
        let provider = ProviderConfig::new("OPENAI_API_KEY", "https://api.openai.com/v1")
            .with_default_model("gpt-4-turbo");

        assert_eq!(provider.api_key_env, "OPENAI_API_KEY");
        assert_eq!(provider.base_url, "https://api.openai.com/v1");
        assert_eq!(provider.default_model, Some("gpt-4-turbo".to_string()));
    }
}
