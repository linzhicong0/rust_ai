// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Token Optimization (REQ-18.2)
//!
//! Optimizes token usage through prompt compression, context pruning,
//! and response length control.

use serde::{Deserialize, Serialize};

/// Errors that can occur during token optimization.
#[derive(Debug, thiserror::Error)]
pub enum TokenOptimizationError {
    #[error("Compression error: {0}")]
    Compression(String),
    #[error("Pruning error: {0}")]
    Pruning(String),
    #[error("Token limit exceeded: used {used}, max {max}")]
    LimitExceeded { used: usize, max: usize },
}

/// Configuration for token optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOptimizationConfig {
    /// Maximum tokens allowed in the prompt.
    pub max_prompt_tokens: usize,
    /// Maximum tokens allowed in the response.
    pub max_response_tokens: usize,
    /// Whether to enable prompt compression.
    pub enable_compression: bool,
    /// Whether to enable context pruning.
    pub enable_pruning: bool,
    /// Minimum relevance score for keeping context entries (0.0 to 1.0).
    pub min_relevance_score: f64,
    /// Whether to remove redundant whitespace.
    pub compress_whitespace: bool,
    /// Whether to shorten common instruction patterns.
    pub shorten_instructions: bool,
}

impl Default for TokenOptimizationConfig {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 4096,
            max_response_tokens: 2048,
            enable_compression: true,
            enable_pruning: true,
            min_relevance_score: 0.3,
            compress_whitespace: true,
            shorten_instructions: true,
        }
    }
}

impl TokenOptimizationConfig {
    /// Create a new config with specified token limits.
    pub fn new(max_prompt: usize, max_response: usize) -> Self {
        Self {
            max_prompt_tokens: max_prompt,
            max_response_tokens: max_response,
            ..Default::default()
        }
    }
}

/// A context entry with a relevance score for pruning decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredContextEntry {
    /// The text content.
    pub content: String,
    /// Relevance score (0.0 to 1.0).
    pub relevance: f64,
    /// Estimated token count.
    pub token_count: usize,
}

impl ScoredContextEntry {
    /// Create a new scored context entry.
    pub fn new(content: impl Into<String>, relevance: f64) -> Self {
        let content = content.into();
        let token_count = estimate_tokens(&content);
        Self {
            content,
            relevance,
            token_count,
        }
    }
}

/// Result of applying token optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// The optimized text.
    pub text: String,
    /// Original token count (estimated).
    pub original_tokens: usize,
    /// Optimized token count (estimated).
    pub optimized_tokens: usize,
    /// Token savings.
    pub tokens_saved: usize,
    /// Compression ratio (0.0 to 1.0, lower is more compressed).
    pub compression_ratio: f64,
}

/// Estimate the number of tokens in a text string.
/// Uses a simple heuristic of ~4 characters per token on average.
pub fn estimate_tokens(text: &str) -> usize {
    // Rough heuristic: 1 token ≈ 4 characters for English text
    (text.len() + 3) / 4
}

/// Compress a prompt by removing redundancy and shortening instructions.
pub fn compress_prompt(text: &str, config: &TokenOptimizationConfig) -> OptimizationResult {
    let original_tokens = estimate_tokens(text);
    let mut result = text.to_string();

    if config.compress_whitespace {
        // Remove excessive whitespace
        result = compress_whitespace(&result);
    }

    if config.shorten_instructions {
        // Shorten common verbose patterns
        result = shorten_instructions(&result);
    }

    let optimized_tokens = estimate_tokens(&result);
    let tokens_saved = original_tokens.saturating_sub(optimized_tokens);
    let compression_ratio = if original_tokens > 0 {
        optimized_tokens as f64 / original_tokens as f64
    } else {
        1.0
    };

    OptimizationResult {
        text: result,
        original_tokens,
        optimized_tokens,
        tokens_saved,
        compression_ratio,
    }
}

/// Remove redundant whitespace from text.
fn compress_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = false;
    let mut prev_was_newline = false;

    for ch in text.chars() {
        if ch == '\n' {
            if !prev_was_newline {
                result.push('\n');
                prev_was_newline = true;
            }
            prev_was_space = false;
        } else if ch.is_whitespace() {
            if !prev_was_space && !prev_was_newline {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(ch);
            prev_was_space = false;
            prev_was_newline = false;
        }
    }

    result.trim().to_string()
}

/// Shorten common verbose instruction patterns.
fn shorten_instructions(text: &str) -> String {
    let mut result = text.to_string();

    // Common verbose patterns -> shorter equivalents
    let replacements = [
        ("Please make sure to", "Ensure"),
        ("please make sure to", "ensure"),
        ("You should always", "Always"),
        ("you should always", "always"),
        ("It is important that you", "You must"),
        ("it is important that you", "you must"),
        ("In order to", "To"),
        ("in order to", "to"),
        ("Please note that", "Note:"),
        ("please note that", "note:"),
        ("Make sure that you", "Ensure you"),
        ("make sure that you", "ensure you"),
        ("As a result of this", "Therefore"),
        ("as a result of this", "therefore"),
        ("Due to the fact that", "Because"),
        ("due to the fact that", "because"),
    ];

    for (verbose, short) in &replacements {
        result = result.replace(verbose, short);
    }

    result
}

/// Prune context entries based on relevance scores and token budget.
pub fn prune_context(
    entries: &[ScoredContextEntry],
    config: &TokenOptimizationConfig,
) -> Vec<ScoredContextEntry> {
    let mut relevant: Vec<&ScoredContextEntry> = entries
        .iter()
        .filter(|e| e.relevance >= config.min_relevance_score)
        .collect();

    // Sort by relevance (highest first)
    relevant.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));

    // Keep entries until we hit the token budget
    let mut total_tokens = 0;
    let mut kept = Vec::new();

    for entry in relevant {
        if total_tokens + entry.token_count > config.max_prompt_tokens {
            break;
        }
        total_tokens += entry.token_count;
        kept.push(entry.clone());
    }

    kept
}

/// Enforce maximum response token limits by truncating a response.
pub fn enforce_response_limit(text: &str, max_tokens: usize) -> String {
    let estimated = estimate_tokens(text);
    if estimated <= max_tokens {
        return text.to_string();
    }

    // Approximate character limit from token limit
    let char_limit = max_tokens * 4;
    if char_limit >= text.len() {
        return text.to_string();
    }

    // Find a good truncation point (sentence or word boundary)
    let truncated = &text[..char_limit];
    if let Some(last_period) = truncated.rfind(". ") {
        format!("{}.", &truncated[..last_period])
    } else if let Some(last_space) = truncated.rfind(' ') {
        truncated[..last_space].to_string() + "..."
    } else {
        truncated.to_string() + "..."
    }
}

/// A token optimizer that combines compression, pruning, and limit enforcement.
pub struct TokenOptimizer {
    config: TokenOptimizationConfig,
}

impl TokenOptimizer {
    /// Create a new token optimizer with the given configuration.
    pub fn new(config: TokenOptimizationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: TokenOptimizationConfig::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TokenOptimizationConfig {
        &self.config
    }

    /// Compress a prompt.
    pub fn compress(&self, text: &str) -> OptimizationResult {
        compress_prompt(text, &self.config)
    }

    /// Prune context entries.
    pub fn prune(&self, entries: &[ScoredContextEntry]) -> Vec<ScoredContextEntry> {
        prune_context(entries, &self.config)
    }

    /// Enforce response token limit.
    pub fn enforce_limit(&self, text: &str) -> String {
        enforce_response_limit(text, self.config.max_response_tokens)
    }

    /// Check if prompt fits within token budget.
    pub fn fits_budget(&self, text: &str) -> bool {
        estimate_tokens(text) <= self.config.max_prompt_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 1); // 2 chars -> (2+3)/4 = 1
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars -> (11+3)/4 = 3
        // Longer text
        let long_text = "a".repeat(100);
        assert_eq!(estimate_tokens(&long_text), 25); // (100+3)/4 = 25
    }

    #[test]
    fn test_compress_whitespace() {
        let input = "Hello   world   this   is   a   test";
        let result = compress_whitespace(input);
        assert_eq!(result, "Hello world this is a test");
    }

    #[test]
    fn test_compress_multiple_newlines() {
        let input = "Line one\n\n\n\nLine two\n\n\nLine three";
        let result = compress_whitespace(input);
        assert_eq!(result, "Line one\nLine two\nLine three");
    }

    #[test]
    fn test_shorten_instructions() {
        let input = "Please make sure to validate all inputs. In order to do this, you should check the schema.";
        let result = shorten_instructions(input);
        assert_eq!(
            result,
            "Ensure validate all inputs. To do this, you should check the schema."
        );
    }

    #[test]
    fn test_compress_prompt() {
        let config = TokenOptimizationConfig::default();
        let input = "Please make sure to   follow   these   instructions   carefully.";
        let result = compress_prompt(input, &config);

        assert!(result.optimized_tokens <= result.original_tokens);
        assert!(result.tokens_saved > 0);
        assert!(result.compression_ratio <= 1.0);
        assert!(!result.text.contains("   "));
        assert!(result.text.contains("Ensure"));
    }

    #[test]
    fn test_compress_prompt_no_compression() {
        let config = TokenOptimizationConfig {
            enable_compression: true,
            compress_whitespace: false,
            shorten_instructions: false,
            ..Default::default()
        };
        let input = "Simple text   with spaces.";
        let result = compress_prompt(input, &config);
        // No transformations since both are disabled
        assert_eq!(result.text, input);
    }

    #[test]
    fn test_prune_context_by_relevance() {
        let config = TokenOptimizationConfig {
            min_relevance_score: 0.5,
            max_prompt_tokens: 10000,
            ..Default::default()
        };

        let entries = vec![
            ScoredContextEntry::new("Highly relevant content", 0.9),
            ScoredContextEntry::new("Low relevance noise", 0.2),
            ScoredContextEntry::new("Medium relevance", 0.6),
            ScoredContextEntry::new("Very low relevance", 0.1),
        ];

        let pruned = prune_context(&entries, &config);
        assert_eq!(pruned.len(), 2);
        assert!(pruned[0].relevance >= pruned[1].relevance);
        assert!(pruned.iter().all(|e| e.relevance >= 0.5));
    }

    #[test]
    fn test_prune_context_by_token_budget() {
        let config = TokenOptimizationConfig {
            min_relevance_score: 0.0,
            max_prompt_tokens: 10, // Very small budget
            ..Default::default()
        };

        let entries = vec![
            ScoredContextEntry::new("First entry with some content", 0.9),
            ScoredContextEntry::new("Second entry with more content", 0.8),
            ScoredContextEntry::new("Third entry overflows budget", 0.7),
        ];

        let pruned = prune_context(&entries, &config);
        // Should keep only entries that fit within 10 tokens
        assert!(pruned.len() < entries.len());
        // Total tokens should be within budget
        let total: usize = pruned.iter().map(|e| e.token_count).sum();
        assert!(total <= 10);
    }

    #[test]
    fn test_enforce_response_limit_short_text() {
        let text = "Short text.";
        let result = enforce_response_limit(text, 100);
        assert_eq!(result, text);
    }

    #[test]
    fn test_enforce_response_limit_long_text() {
        let text = "This is a long sentence. It has multiple parts. And we need to truncate it. Because it exceeds the limit.";
        let result = enforce_response_limit(text, 5); // ~20 chars
        assert!(result.len() < text.len());
        // Should truncate at a sentence boundary
        assert!(result.ends_with('.') || result.ends_with("..."));
    }

    #[test]
    fn test_token_optimizer_compress() {
        let optimizer = TokenOptimizer::default_config();
        let result = optimizer.compress("Please make sure to   check   everything.");
        assert!(result.tokens_saved > 0);
    }

    #[test]
    fn test_token_optimizer_fits_budget() {
        let config = TokenOptimizationConfig::new(10, 5);
        let optimizer = TokenOptimizer::new(config);

        assert!(optimizer.fits_budget("hi")); // 1 token
        assert!(!optimizer.fits_budget(&"x".repeat(100))); // 25 tokens > 10
    }

    #[test]
    fn test_token_optimizer_prune() {
        let optimizer = TokenOptimizer::default_config();
        let entries = vec![
            ScoredContextEntry::new("Relevant", 0.8),
            ScoredContextEntry::new("Irrelevant", 0.1),
        ];
        let pruned = optimizer.prune(&entries);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].content, "Relevant");
    }

    #[test]
    fn test_token_optimizer_enforce_limit() {
        let config = TokenOptimizationConfig::new(4096, 3);
        let optimizer = TokenOptimizer::new(config);
        let long_text = "word ".repeat(50);
        let result = optimizer.enforce_limit(&long_text);
        assert!(result.len() < long_text.len());
    }

    #[test]
    fn test_optimization_config_default() {
        let config = TokenOptimizationConfig::default();
        assert_eq!(config.max_prompt_tokens, 4096);
        assert_eq!(config.max_response_tokens, 2048);
        assert!(config.enable_compression);
        assert!(config.enable_pruning);
        assert!(config.compress_whitespace);
        assert!(config.shorten_instructions);
    }

    #[test]
    fn test_scored_context_entry_new() {
        let entry = ScoredContextEntry::new("Hello world", 0.8);
        assert_eq!(entry.content, "Hello world");
        assert_eq!(entry.relevance, 0.8);
        assert_eq!(entry.token_count, estimate_tokens("Hello world"));
    }
}
