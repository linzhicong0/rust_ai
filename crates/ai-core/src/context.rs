// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Context window management for LLM conversations.
//!
//! This module provides utilities for managing conversation context within
//! model context window limits, including token counting, truncation strategies,
//! and warnings when approaching limits.

use crate::types::{Message, Role};
use std::collections::VecDeque;

/// Default context window size for models (in tokens).
///
/// This is a conservative default. Most modern models support larger contexts:
/// - GPT-4: 8k, 32k, or 128k depending on variant
/// - Claude 3: 200k tokens
/// - GPT-3.5 Turbo: 16k tokens
const DEFAULT_CONTEXT_WINDOW: usize = 4096;

/// Warning threshold as percentage of context window.
///
/// Warn when context usage exceeds this percentage of the limit.
const WARNING_THRESHOLD_PCT: f64 = 0.8; // 80%

/// Truncation strategies for managing context window overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Remove oldest messages first.
    TruncateOldest,

    /// Keep recent N messages, discard the rest.
    SlidingWindow { keep_last_n: usize },

    /// Prioritize system and recent messages, remove older assistant/user messages.
    PrioritizeSystem,

    /// No truncation - return error if context exceeds limit.
    Error,
}

/// Configuration for context window management.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum context window size in tokens.
    pub max_tokens: usize,

    /// Truncation strategy to use when limit is exceeded.
    pub truncation_strategy: TruncationStrategy,

    /// Whether to emit warnings when approaching the limit.
    pub warn_on_approach: bool,

    /// Percentage threshold for warnings (0.0 to 1.0).
    pub warning_threshold: f64,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_CONTEXT_WINDOW,
            truncation_strategy: TruncationStrategy::TruncateOldest,
            warn_on_approach: true,
            warning_threshold: WARNING_THRESHOLD_PCT,
        }
    }
}

impl ContextConfig {
    /// Create a new context config with the specified max tokens.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }

    /// Set the truncation strategy.
    pub fn with_truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self
    }

    /// Enable or disable warnings when approaching the limit.
    pub fn with_warnings(mut self, warn: bool) -> Self {
        self.warn_on_approach = warn;
        self
    }

    /// Set the warning threshold (0.0 to 1.0).
    ///
    /// # Panics
    ///
    /// Panics if threshold is outside [0.0, 1.0].
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "warning threshold must be between 0.0 and 1.0"
        );
        self.warning_threshold = threshold;
        self
    }

    /// Create a config for GPT-4 (8k context).
    pub fn gpt4_8k() -> Self {
        Self::new(8192)
    }

    /// Create a config for GPT-4 (32k context).
    pub fn gpt4_32k() -> Self {
        Self::new(32768)
    }

    /// Create a config for GPT-3.5 Turbo (16k context).
    pub fn gpt35_turbo_16k() -> Self {
        Self::new(16384)
    }

    /// Create a config for Claude 3 (200k context).
    pub fn claude_3_200k() -> Self {
        Self::new(200_000)
    }
}

/// Information about context window usage.
#[derive(Debug, Clone)]
pub struct ContextUsage {
    /// Total tokens in the context.
    pub total_tokens: usize,

    /// Maximum context window size.
    pub max_tokens: usize,

    /// Whether the context exceeds the limit.
    pub exceeds_limit: bool,

    /// Whether the context is approaching the warning threshold.
    pub approaching_warning: bool,

    /// Token count per message (in order).
    pub tokens_per_message: Vec<usize>,
}

impl ContextUsage {
    /// Calculate usage percentage (0.0 to 1.0+).
    pub fn usage_percentage(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / self.max_tokens as f64
    }

    /// Get the number of tokens that can still be added before hitting the limit.
    pub fn remaining_tokens(&self) -> usize {
        if self.total_tokens >= self.max_tokens {
            0
        } else {
            self.max_tokens - self.total_tokens
        }
    }
}

/// Result of context window management.
#[derive(Debug)]
pub enum ContextResult {
    /// Context is within limits and unchanged.
    Ok { messages: Vec<Message>, usage: ContextUsage },

    /// Context was truncated to fit within limits.
    Truncated {
        messages: Vec<Message>,
        usage: ContextUsage,
        original_count: usize,
        removed_count: usize,
    },

    /// Context exceeds limit and truncation strategy is set to Error.
    ExceededLimit { usage: ContextUsage },
}

/// Context window manager.
///
/// Tracks token usage and manages context truncation according to the configured strategy.
#[derive(Debug, Clone)]
pub struct ContextManager {
    config: ContextConfig,
}

impl ContextManager {
    /// Create a new context manager with the given config.
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Create a new context manager with default config.
    pub fn default_config() -> Self {
        Self::with_max_tokens(DEFAULT_CONTEXT_WINDOW)
    }

    /// Create a new context manager with the specified max tokens.
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self::new(ContextConfig::new(max_tokens))
    }

    /// Analyze a set of messages and return usage information.
    pub fn analyze(&self, messages: &[Message]) -> ContextUsage {
        let tokens_per_message: Vec<usize> = messages
            .iter()
            .map(|msg| estimate_tokens(msg))
            .collect();

        let total_tokens: usize = tokens_per_message.iter().sum();

        let usage_percentage = if self.config.max_tokens > 0 {
            total_tokens as f64 / self.config.max_tokens as f64
        } else {
            0.0
        };

        ContextUsage {
            total_tokens,
            max_tokens: self.config.max_tokens,
            exceeds_limit: total_tokens > self.config.max_tokens,
            approaching_warning: self.config.warn_on_approach
                && usage_percentage >= self.config.warning_threshold,
            tokens_per_message,
        }
    }

    /// Manage context window for a set of messages.
    ///
    /// Applies the configured truncation strategy if the context exceeds limits.
    pub fn manage(&self, messages: Vec<Message>) -> ContextResult {
        let usage = self.analyze(&messages);

        if !usage.exceeds_limit {
            return ContextResult::Ok { messages, usage };
        }

        match self.config.truncation_strategy {
            TruncationStrategy::Error => ContextResult::ExceededLimit { usage },
            TruncationStrategy::TruncateOldest => {
                self.truncate_oldest(messages, &usage)
            }
            TruncationStrategy::SlidingWindow { keep_last_n } => {
                self.sliding_window(messages, keep_last_n)
            }
            TruncationStrategy::PrioritizeSystem => {
                self.prioritize_system(messages, &usage)
            }
        }
    }

    /// Truncate oldest messages until the context fits within the limit.
    fn truncate_oldest(&self, messages: Vec<Message>, _usage: &ContextUsage) -> ContextResult {
        let original_count = messages.len();
        let mut truncated = VecDeque::from(messages);

        // Always keep system messages
        let system_messages: Vec<Message> = truncated
            .iter()
            .filter(|msg| matches!(msg.role, Role::System))
            .cloned()
            .collect();

        // Remove oldest non-system messages until we fit
        while Self::estimate_tokens_vec(&truncated) > self.config.max_tokens {
            // Find the oldest non-system message
            let oldest_idx = truncated
                .iter()
                .position(|msg| !matches!(msg.role, Role::System));

            if let Some(idx) = oldest_idx {
                truncated.remove(idx);
            } else {
                break; // Only system messages left, can't truncate more
            }
        }

        let messages: Vec<Message> = system_messages.into_iter().chain(truncated).collect();
        let removed_count = original_count.saturating_sub(messages.len());
        let usage = self.analyze(&messages);

        ContextResult::Truncated {
            messages,
            usage,
            original_count,
            removed_count,
        }
    }

    /// Keep only the last N messages.
    fn sliding_window(&self, messages: Vec<Message>, keep_last_n: usize) -> ContextResult {
        let original_count = messages.len();

        // Always keep system messages
        let system_messages: Vec<Message> = messages
            .iter()
            .filter(|msg| matches!(msg.role, Role::System))
            .cloned()
            .collect();

        // Keep last N non-system messages
        let non_system: Vec<Message> = messages
            .into_iter()
            .filter(|msg| !matches!(msg.role, Role::System))
            .rev()
            .take(keep_last_n)
            .collect();

        let mut result: Vec<Message> = system_messages;
        result.extend(non_system.into_iter().rev());

        let removed_count = original_count.saturating_sub(result.len());
        let usage = self.analyze(&result);

        ContextResult::Truncated {
            messages: result,
            usage,
            original_count,
            removed_count,
        }
    }

    /// Prioritize system messages and recent context, remove older user/assistant messages.
    fn prioritize_system(&self, messages: Vec<Message>, _usage: &ContextUsage) -> ContextResult {
        let original_count = messages.len();

        // Separate messages by role
        let mut system_messages: Vec<Message> = Vec::new();
        let mut other_messages: VecDeque<Message> = VecDeque::new();

        for msg in messages {
            match msg.role {
                Role::System => system_messages.push(msg),
                _ => other_messages.push_back(msg),
            }
        }

        // Remove oldest non-system messages until we fit
        while Self::estimate_messages_tokens(&system_messages, &other_messages) > self.config.max_tokens {
            if other_messages.pop_front().is_none() {
                break;
            }
        }

        let mut result: Vec<Message> = system_messages;
        result.extend(other_messages);

        let removed_count = original_count.saturating_sub(result.len());
        let usage = self.analyze(&result);

        ContextResult::Truncated {
            messages: result,
            usage,
            original_count,
            removed_count,
        }
    }

    /// Estimate tokens for a vector of messages.
    fn estimate_tokens_vec(messages: &VecDeque<Message>) -> usize {
        messages.iter().map(|msg| estimate_tokens(msg)).sum()
    }

    /// Estimate tokens for system messages + a deque of other messages.
    fn estimate_messages_tokens(system: &[Message], other: &VecDeque<Message>) -> usize {
        system.iter().map(|msg| estimate_tokens(msg)).sum::<usize>()
            + other.iter().map(|msg| estimate_tokens(msg)).sum::<usize>()
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Estimate the number of tokens in a message.
///
/// This is a rough approximation based on character and word count.
/// For accurate token counting, you would use a tokenizer specific to the model.
///
/// The approximation uses:
/// - ~4 characters per token for English text (OpenAI's rule of thumb)
/// - Higher multiplier for messages with images (vision tokens)
pub fn estimate_tokens(message: &Message) -> usize {
    match &message.content {
        crate::types::Content::Text(text) => {
            // Rough estimate: ~4 characters per token for English
            // Add a small buffer for overhead
            let char_count = text.chars().count();
            (char_count / 4).max(1)
        }
        crate::types::Content::MultiPart(parts) => {
            let mut tokens = 0;

            for part in parts {
                match part {
                    crate::types::ContentPart::Text(text) => {
                        tokens += (text.chars().count() / 4).max(1);
                    }
                    crate::types::ContentPart::Image { detail, .. } |
                    crate::types::ContentPart::ImageBytes { detail, .. } => {
                        // Vision models use different token accounting for images
                        // GPT-4 Vision: ~85 tokens for low detail, ~170+ for high detail
                        match detail.as_deref() {
                            Some("low") => tokens += 85,
                            Some("high") | None => tokens += 170,
                            _ => tokens += 85, // auto defaults to safe estimate
                        }
                    }
                }
            }

            tokens.max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.max_tokens, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(config.truncation_strategy, TruncationStrategy::TruncateOldest);
        assert!(config.warn_on_approach);
        assert_eq!(config.warning_threshold, WARNING_THRESHOLD_PCT);
    }

    #[test]
    fn test_context_config_builder() {
        let config = ContextConfig::new(8192)
            .with_truncation_strategy(TruncationStrategy::SlidingWindow { keep_last_n: 10 })
            .with_warnings(false)
            .with_warning_threshold(0.9);

        assert_eq!(config.max_tokens, 8192);
        assert_eq!(
            config.truncation_strategy,
            TruncationStrategy::SlidingWindow { keep_last_n: 10 }
        );
        assert!(!config.warn_on_approach);
        assert_eq!(config.warning_threshold, 0.9);
    }

    #[test]
    fn test_context_config_presets() {
        let gpt4_8k = ContextConfig::gpt4_8k();
        assert_eq!(gpt4_8k.max_tokens, 8192);

        let gpt4_32k = ContextConfig::gpt4_32k();
        assert_eq!(gpt4_32k.max_tokens, 32768);

        let gpt35 = ContextConfig::gpt35_turbo_16k();
        assert_eq!(gpt35.max_tokens, 16384);

        let claude = ContextConfig::claude_3_200k();
        assert_eq!(claude.max_tokens, 200_000);
    }

    #[test]
    fn test_context_config_warning_threshold_validation() {
        // Valid thresholds should not panic
        let _ = ContextConfig::new(4096).with_warning_threshold(0.0);
        let _ = ContextConfig::new(4096).with_warning_threshold(0.5);
        let _ = ContextConfig::new(4096).with_warning_threshold(1.0);
    }

    #[test]
    #[should_panic(expected = "warning threshold must be between 0.0 and 1.0")]
    fn test_context_config_warning_threshold_too_high() {
        ContextConfig::new(4096).with_warning_threshold(1.5);
    }

    #[test]
    #[should_panic(expected = "warning threshold must be between 0.0 and 1.0")]
    fn test_context_config_warning_threshold_negative() {
        ContextConfig::new(4096).with_warning_threshold(-0.1);
    }

    #[test]
    fn test_estimate_tokens_text_message() {
        let msg = Message::user("Hello, world!");
        let tokens = estimate_tokens(&msg);
        assert!(tokens > 0);
        assert!(tokens < 100); // Should be reasonable
    }

    #[test]
    fn test_estimate_tokens_longer_text() {
        let long_text = "This is a much longer message that should have more tokens. ".repeat(10);
        let msg = Message::user(&long_text);
        let tokens = estimate_tokens(&msg);

        let short_msg = Message::user("Short");
        let short_tokens = estimate_tokens(&short_msg);

        assert!(tokens > short_tokens);
    }

    #[test]
    fn test_context_manager_analyze() {
        let manager = ContextManager::with_max_tokens(1000);

        let messages = vec![
            Message::system("You are a helpful assistant"),
            Message::user("Hello!"),
            Message::assistant("Hi there!"),
        ];

        let usage = manager.analyze(&messages);

        assert!(usage.total_tokens > 0);
        assert_eq!(usage.max_tokens, 1000);
        assert!(!usage.exceeds_limit);
        assert_eq!(usage.tokens_per_message.len(), 3);
    }

    #[test]
    fn test_context_manager_analyze_exceeds_limit() {
        let manager = ContextManager::with_max_tokens(10);

        let messages = vec![
            Message::user("This is a relatively long message that exceeds the tiny limit"),
        ];

        let usage = manager.analyze(&messages);

        assert!(usage.total_tokens > 10);
        assert!(usage.exceeds_limit);
    }

    #[test]
    fn test_context_usage_percentage() {
        let usage = ContextUsage {
            total_tokens: 500,
            max_tokens: 1000,
            exceeds_limit: false,
            approaching_warning: false,
            tokens_per_message: vec![],
        };

        assert_eq!(usage.usage_percentage(), 0.5);
    }

    #[test]
    fn test_context_usage_remaining_tokens() {
        let usage = ContextUsage {
            total_tokens: 300,
            max_tokens: 1000,
            exceeds_limit: false,
            approaching_warning: false,
            tokens_per_message: vec![],
        };

        assert_eq!(usage.remaining_tokens(), 700);
    }

    #[test]
    fn test_context_usage_remaining_tokens_when_full() {
        let usage = ContextUsage {
            total_tokens: 1000,
            max_tokens: 1000,
            exceeds_limit: false,
            approaching_warning: false,
            tokens_per_message: vec![],
        };

        assert_eq!(usage.remaining_tokens(), 0);
    }

    #[test]
    fn test_context_usage_remaining_tokens_when_exceeded() {
        let usage = ContextUsage {
            total_tokens: 1500,
            max_tokens: 1000,
            exceeds_limit: true,
            approaching_warning: false,
            tokens_per_message: vec![],
        };

        assert_eq!(usage.remaining_tokens(), 0);
    }

    #[test]
    fn test_context_manager_manage_within_limits() {
        let manager = ContextManager::with_max_tokens(10000);

        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi"),
        ];

        match manager.manage(messages) {
            ContextResult::Ok { messages, usage } => {
                assert_eq!(messages.len(), 3);
                assert!(!usage.exceeds_limit);
            }
            _ => panic!("Expected Ok result"),
        }
    }

    #[test]
    fn test_context_manager_truncate_oldest() {
        let manager = ContextManager::new(
            ContextConfig::new(100)
                .with_truncation_strategy(TruncationStrategy::TruncateOldest)
        );

        let messages = vec![
            Message::system("System prompt"),
            Message::user(&"Old message ".repeat(100)),
            Message::assistant(&"Old response ".repeat(100)),
            Message::user("New message"),
        ];

        match manager.manage(messages) {
            ContextResult::Truncated {
                messages,
                original_count,
                removed_count,
                ..
            } => {
                assert_eq!(original_count, 4);
                assert!(removed_count > 0);
                // System message should always be preserved
                assert!(messages.iter().any(|m| matches!(m.role, Role::System)));
            }
            _ => panic!("Expected Truncated result"),
        }
    }

    #[test]
    fn test_context_manager_sliding_window() {
        let manager = ContextManager::new(
            ContextConfig::new(50) // Reduced limit to force truncation
                .with_truncation_strategy(TruncationStrategy::SlidingWindow { keep_last_n: 2 })
        );

        let long_text = "This is a longer message that uses more tokens to ensure truncation occurs. ".repeat(5);
        let messages = vec![
            Message::system("System prompt"),
            Message::user(&format!("Message 1: {}", long_text)),
            Message::assistant(&format!("Response 1: {}", long_text)),
            Message::user(&format!("Message 2: {}", long_text)),
            Message::assistant(&format!("Response 2: {}", long_text)),
            Message::user(&format!("Message 3: {}", long_text)),
        ];

        match manager.manage(messages) {
            ContextResult::Truncated { messages, .. } => {
                // Should have system + 2 most recent messages
                assert!(messages.iter().any(|m| matches!(m.role, Role::System)));
                // Total messages should be <= 3 (1 system + 2 recent)
                assert!(messages.len() <= 3);
            }
            _ => panic!("Expected Truncated result"),
        }
    }

    #[test]
    fn test_context_manager_error_strategy() {
        let manager = ContextManager::new(
            ContextConfig::new(10)
                .with_truncation_strategy(TruncationStrategy::Error)
        );

        let messages = vec![
            Message::user(&"This is a long message that exceeds the limit ".repeat(10)),
        ];

        match manager.manage(messages) {
            ContextResult::ExceededLimit { usage } => {
                assert!(usage.exceeds_limit);
            }
            _ => panic!("Expected ExceededLimit result"),
        }
    }

    #[test]
    fn test_context_manager_warn_on_approach() {
        let manager = ContextManager::new(
            ContextConfig::new(100)
                .with_warnings(true)
                .with_warning_threshold(0.5)
        );

        let messages = vec![
            Message::user(&"This message uses about 60 tokens worth of text content ".repeat(5)),
        ];

        let usage = manager.analyze(&messages);
        assert!(usage.approaching_warning);
    }

    #[test]
    fn test_prioritize_system_preserves_system_messages() {
        let manager = ContextManager::new(
            ContextConfig::new(100)
                .with_truncation_strategy(TruncationStrategy::PrioritizeSystem)
        );

        let messages = vec![
            Message::system("Critical system prompt that must be preserved"),
            Message::user(&"Old user message ".repeat(100)),
            Message::assistant(&"Old assistant response ".repeat(100)),
            Message::user("New user message"),
        ];

        match manager.manage(messages) {
            ContextResult::Truncated { messages, .. } => {
                // System message should always be present
                assert!(messages.iter().any(|m| {
                    matches!(m.role, Role::System)
                        && m.content.as_text().unwrap().contains("Critical")
                }));
            }
            _ => panic!("Expected Truncated result"),
        }
    }

    #[test]
    fn test_truncation_strategies_preserve_system_messages() {
        let messages = vec![
            Message::system("System prompt"),
            Message::user("User 1"),
            Message::assistant("Assistant 1"),
        ];

        // Test all strategies preserve system messages
        for strategy in [
            TruncationStrategy::TruncateOldest,
            TruncationStrategy::SlidingWindow { keep_last_n: 1 },
            TruncationStrategy::PrioritizeSystem,
        ] {
            let manager = ContextManager::new(
                ContextConfig::new(10).with_truncation_strategy(strategy)
            );

            match manager.manage(messages.clone()) {
                ContextResult::Ok { messages, .. } | ContextResult::Truncated { messages, .. } => {
                    assert!(messages.iter().any(|m| matches!(m.role, Role::System)));
                }
                ContextResult::ExceededLimit { .. } => {
                    // Small limit might still exceed, but that's ok for this test
                }
            }
        }
    }

    #[test]
    fn test_estimate_tokens_for_empty_message() {
        let msg = Message::user("");
        let tokens = estimate_tokens(&msg);
        // Empty message should still count as at least 1 token
        assert_eq!(tokens, 1);
    }

    #[test]
    fn test_estimate_tokens_for_multipart_with_images() {
        let msg = Message::user_with_image(
            "What's in this image?",
            "https://example.com/image.jpg",
            "image/jpeg"
        );

        let tokens = estimate_tokens(&msg);
        // Should account for both text and image tokens
        assert!(tokens > 80); // At least the image tokens
    }
}
