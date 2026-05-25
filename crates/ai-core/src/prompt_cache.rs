//! Prompt caching support for LLM providers (REQ-12.3).
//!
//! This module provides mechanisms to leverage provider-side prompt caching
//! (e.g., Anthropic prompt caching) to reduce costs and latency.
//!
//! ## Features
//!
//! - Detect provider caching support
//! - Mark cacheable prompt prefixes with provider-specific markers
//! - Report cache hit/miss status in response metadata

use std::collections::HashMap;

/// Supported caching providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachingProvider {
    /// Anthropic provider with native prompt caching.
    Anthropic,
    /// OpenAI provider (automatic caching, no explicit markers needed).
    OpenAi,
    /// Provider that does not support caching.
    None,
}

/// Metadata about caching behavior in a response.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheMetadata {
    /// Number of tokens written to the cache for future use.
    pub cache_creation_input_tokens: u64,

    /// Number of tokens read from the cache (cache hit).
    pub cache_read_input_tokens: u64,

    /// Additional provider-specific metadata.
    pub extra: HashMap<String, String>,
}

impl CacheMetadata {
    /// Create empty cache metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata indicating a cache hit.
    pub fn cache_hit(read_tokens: u64) -> Self {
        Self {
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: read_tokens,
            extra: HashMap::new(),
        }
    }

    /// Create metadata indicating a cache miss (new cache creation).
    pub fn cache_miss(creation_tokens: u64) -> Self {
        Self {
            cache_creation_input_tokens: creation_tokens,
            cache_read_input_tokens: 0,
            extra: HashMap::new(),
        }
    }

    /// Returns true if there was any cache read (hit).
    pub fn is_cache_hit(&self) -> bool {
        self.cache_read_input_tokens > 0
    }

    /// Returns true if new cache was created (miss).
    pub fn is_cache_miss(&self) -> bool {
        self.cache_creation_input_tokens > 0 && self.cache_read_input_tokens == 0
    }

    /// Add extra metadata.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Configuration for prompt caching behavior.
#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    /// Whether caching is enabled.
    pub enabled: bool,

    /// The provider to use for caching.
    pub provider: CachingProvider,

    /// Minimum token count for a prefix to be cacheable.
    /// Anthropic requires at least 1024 tokens for caching.
    pub min_cacheable_tokens: u64,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: CachingProvider::None,
            min_cacheable_tokens: 1024,
        }
    }
}

impl PromptCacheConfig {
    /// Create a config for Anthropic caching.
    pub fn anthropic() -> Self {
        Self {
            enabled: true,
            provider: CachingProvider::Anthropic,
            min_cacheable_tokens: 1024,
        }
    }

    /// Create a config for OpenAI caching.
    pub fn openai() -> Self {
        Self {
            enabled: true,
            provider: CachingProvider::OpenAi,
            min_cacheable_tokens: 0, // OpenAI handles this automatically
        }
    }
}

/// Detects whether a provider supports prompt caching.
pub fn detect_caching_support(provider_name: &str) -> CachingProvider {
    match provider_name.to_lowercase().as_str() {
        "anthropic" | "claude" => CachingProvider::Anthropic,
        "openai" | "gpt" => CachingProvider::OpenAi,
        _ => CachingProvider::None,
    }
}

/// Marker for cacheable prompt content.
///
/// Anthropic uses a `cache_control` field with `type: "ephemeral"` to mark
/// content blocks that should be cached.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheMarker {
    /// The type of cache control (e.g., "ephemeral" for Anthropic).
    pub cache_type: String,
    /// Index of the content block to mark as cacheable.
    pub block_index: usize,
}

impl CacheMarker {
    /// Create an Anthropic ephemeral cache marker.
    pub fn anthropic_ephemeral(block_index: usize) -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            block_index,
        }
    }
}

/// A message content block with optional cache control.
#[derive(Debug, Clone)]
pub struct CacheableContent {
    /// The text content.
    pub text: String,
    /// Whether this block should be cached.
    pub cacheable: bool,
    /// The cache control marker if cacheable.
    pub cache_control: Option<CacheMarker>,
}

impl CacheableContent {
    /// Create a non-cacheable content block.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            text: content.into(),
            cacheable: false,
            cache_control: None,
        }
    }

    /// Create a cacheable content block for a specific provider.
    pub fn cacheable(content: impl Into<String>, provider: CachingProvider) -> Self {
        let text = content.into();
        let cache_control = match provider {
            CachingProvider::Anthropic => Some(CacheMarker::anthropic_ephemeral(0)),
            _ => None,
        };
        Self {
            text,
            cacheable: true,
            cache_control,
        }
    }
}

/// Marks the system prompt as cacheable for the given provider.
///
/// For Anthropic, this adds `cache_control: {"type": "ephemeral"}` metadata.
/// For OpenAI, no explicit marking is needed (automatic).
pub fn mark_system_prompt_cacheable(
    system_prompt: &str,
    provider: CachingProvider,
) -> CacheableContent {
    match provider {
        CachingProvider::Anthropic => CacheableContent::cacheable(system_prompt, provider),
        CachingProvider::OpenAi => {
            // OpenAI handles caching automatically, but we mark it as cacheable for tracking
            CacheableContent {
                text: system_prompt.to_string(),
                cacheable: true,
                cache_control: None,
            }
        }
        CachingProvider::None => CacheableContent::text(system_prompt),
    }
}

/// Parse cache metadata from an Anthropic API response.
///
/// Anthropic responses include `cache_creation_input_tokens` and
/// `cache_read_input_tokens` in the usage block.
pub fn parse_anthropic_cache_metadata(usage: &serde_json::Value) -> CacheMetadata {
    let creation = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let read = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    CacheMetadata {
        cache_creation_input_tokens: creation,
        cache_read_input_tokens: read,
        extra: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-12.3: Unit: Anthropic provider detects prompt caching capability
    #[test]
    fn test_detect_anthropic_caching_support() {
        assert_eq!(
            detect_caching_support("anthropic"),
            CachingProvider::Anthropic
        );
        assert_eq!(
            detect_caching_support("Anthropic"),
            CachingProvider::Anthropic
        );
        assert_eq!(detect_caching_support("claude"), CachingProvider::Anthropic);
    }

    #[test]
    fn test_detect_openai_caching_support() {
        assert_eq!(detect_caching_support("openai"), CachingProvider::OpenAi);
        assert_eq!(detect_caching_support("gpt"), CachingProvider::OpenAi);
    }

    #[test]
    fn test_detect_no_caching_support() {
        assert_eq!(detect_caching_support("ollama"), CachingProvider::None);
        assert_eq!(detect_caching_support("unknown"), CachingProvider::None);
    }

    // REQ-12.3: Unit: cacheable prefix marker is added to system prompt for Anthropic
    #[test]
    fn test_mark_system_prompt_cacheable_anthropic() {
        let prompt = "You are a helpful assistant specialized in Rust programming.";
        let result = mark_system_prompt_cacheable(prompt, CachingProvider::Anthropic);

        assert!(result.cacheable);
        assert_eq!(result.text, prompt);
        assert!(result.cache_control.is_some());

        let marker = result.cache_control.unwrap();
        assert_eq!(marker.cache_type, "ephemeral");
        assert_eq!(marker.block_index, 0);
    }

    #[test]
    fn test_mark_system_prompt_no_caching_provider() {
        let prompt = "You are a helpful assistant.";
        let result = mark_system_prompt_cacheable(prompt, CachingProvider::None);

        assert!(!result.cacheable);
        assert!(result.cache_control.is_none());
    }

    // REQ-12.3: Unit: response metadata includes cache_creation_input_tokens and cache_read_input_tokens
    #[test]
    fn test_parse_anthropic_cache_metadata_cache_miss() {
        let usage = serde_json::json!({
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 1500,
            "cache_read_input_tokens": 0
        });

        let metadata = parse_anthropic_cache_metadata(&usage);
        assert_eq!(metadata.cache_creation_input_tokens, 1500);
        assert_eq!(metadata.cache_read_input_tokens, 0);
        assert!(metadata.is_cache_miss());
        assert!(!metadata.is_cache_hit());
    }

    #[test]
    fn test_parse_anthropic_cache_metadata_cache_hit() {
        let usage = serde_json::json!({
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1500
        });

        let metadata = parse_anthropic_cache_metadata(&usage);
        assert_eq!(metadata.cache_creation_input_tokens, 0);
        assert_eq!(metadata.cache_read_input_tokens, 1500);
        assert!(metadata.is_cache_hit());
        assert!(!metadata.is_cache_miss());
    }

    // REQ-12.3: Integration: repeated requests with same prefix show cache_read hits
    #[test]
    fn test_repeated_requests_show_cache_hits() {
        // Simulate first request: cache miss (creation)
        let first_response_usage = serde_json::json!({
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 1800,
            "cache_read_input_tokens": 0
        });

        let first_metadata = parse_anthropic_cache_metadata(&first_response_usage);
        assert!(first_metadata.is_cache_miss());
        assert!(!first_metadata.is_cache_hit());

        // Simulate second request: cache hit (read)
        let second_response_usage = serde_json::json!({
            "input_tokens": 200,
            "output_tokens": 500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1800
        });

        let second_metadata = parse_anthropic_cache_metadata(&second_response_usage);
        assert!(second_metadata.is_cache_hit());
        assert!(!second_metadata.is_cache_miss());
        assert_eq!(second_metadata.cache_read_input_tokens, 1800);
    }

    #[test]
    fn test_cache_metadata_default() {
        let metadata = CacheMetadata::new();
        assert_eq!(metadata.cache_creation_input_tokens, 0);
        assert_eq!(metadata.cache_read_input_tokens, 0);
        assert!(!metadata.is_cache_hit());
        assert!(!metadata.is_cache_miss());
    }

    #[test]
    fn test_prompt_cache_config_anthropic() {
        let config = PromptCacheConfig::anthropic();
        assert!(config.enabled);
        assert_eq!(config.provider, CachingProvider::Anthropic);
        assert_eq!(config.min_cacheable_tokens, 1024);
    }

    #[test]
    fn test_cacheable_content_creation() {
        let content = CacheableContent::cacheable("test content", CachingProvider::Anthropic);
        assert!(content.cacheable);
        assert!(content.cache_control.is_some());

        let content = CacheableContent::text("plain text");
        assert!(!content.cacheable);
        assert!(content.cache_control.is_none());
    }
}
