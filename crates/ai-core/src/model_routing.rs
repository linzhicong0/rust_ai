// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Model Routing (REQ-18.1)
//!
//! Provides intelligent model routing based on task complexity, directing
//! simple queries to cheaper/faster models and complex queries to more capable ones.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Errors that can occur during model routing.
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("No suitable model found for the given context")]
    NoModelAvailable,
    #[error("Routing rule error: {0}")]
    RuleError(String),
    #[error("Classification error: {0}")]
    ClassificationError(String),
}

/// Task complexity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskComplexity {
    /// Simple tasks: short answers, basic lookups, translations.
    Simple,
    /// Moderate tasks: summarization, standard Q&A.
    Moderate,
    /// Complex tasks: reasoning, code generation, multi-step analysis.
    Complex,
    /// Expert tasks: research, creative writing, advanced math.
    Expert,
}

/// Context provided to the router for making routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingContext {
    /// The user prompt text.
    pub prompt: String,
    /// Estimated complexity of the task.
    pub complexity: Option<TaskComplexity>,
    /// Task type hint (e.g., "code", "chat", "analysis").
    pub task_type: Option<String>,
    /// Maximum acceptable latency in milliseconds.
    pub max_latency_ms: Option<u64>,
    /// Maximum acceptable cost per request.
    pub max_cost: Option<f64>,
    /// Additional metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl RoutingContext {
    /// Create a new routing context with just a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            complexity: None,
            task_type: None,
            max_latency_ms: None,
            max_cost: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the task complexity.
    pub fn with_complexity(mut self, complexity: TaskComplexity) -> Self {
        self.complexity = Some(complexity);
        self
    }

    /// Set the task type hint.
    pub fn with_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Set maximum latency.
    pub fn with_max_latency_ms(mut self, ms: u64) -> Self {
        self.max_latency_ms = Some(ms);
        self
    }

    /// Set maximum cost.
    pub fn with_max_cost(mut self, cost: f64) -> Self {
        self.max_cost = Some(cost);
        self
    }
}

/// A model selection decision from the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The selected model identifier.
    pub model: String,
    /// The provider to use.
    pub provider: String,
    /// Reason for selection.
    pub reason: String,
    /// Estimated complexity that informed the decision.
    pub estimated_complexity: TaskComplexity,
    /// Confidence in the routing decision (0.0 to 1.0).
    pub confidence: f64,
}

/// Trait for intelligent model routing.
#[async_trait]
pub trait Router: Send + Sync {
    /// Route a prompt to the best model given the context.
    async fn route(&self, context: &RoutingContext) -> Result<RoutingDecision, RoutingError>;
}

/// A routing rule for the rule-based router.
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Rule name for debugging.
    pub name: String,
    /// Maximum prompt length for this rule to match.
    pub max_prompt_length: Option<usize>,
    /// Keywords that trigger this rule.
    pub keywords: Vec<String>,
    /// Task types that trigger this rule.
    pub task_types: Vec<String>,
    /// Complexity levels that trigger this rule.
    pub complexity_levels: Vec<TaskComplexity>,
    /// Target model when rule matches.
    pub target_model: String,
    /// Target provider when rule matches.
    pub target_provider: String,
    /// Priority (lower = higher priority).
    pub priority: u32,
}

impl RoutingRule {
    /// Create a new routing rule.
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            max_prompt_length: None,
            keywords: Vec::new(),
            task_types: Vec::new(),
            complexity_levels: Vec::new(),
            target_model: model.into(),
            target_provider: provider.into(),
            priority: 100,
        }
    }

    /// Set max prompt length.
    pub fn with_max_length(mut self, length: usize) -> Self {
        self.max_prompt_length = Some(length);
        self
    }

    /// Add keywords.
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Add task types.
    pub fn with_task_types(mut self, types: Vec<String>) -> Self {
        self.task_types = types;
        self
    }

    /// Add complexity levels.
    pub fn with_complexity(mut self, levels: Vec<TaskComplexity>) -> Self {
        self.complexity_levels = levels;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Check whether this rule matches the given context.
    fn matches(&self, context: &RoutingContext) -> bool {
        // Check prompt length
        if let Some(max_len) = self.max_prompt_length {
            if context.prompt.len() > max_len {
                return false;
            }
        }

        // Check keywords
        if !self.keywords.is_empty() {
            let prompt_lower = context.prompt.to_lowercase();
            let has_keyword = self
                .keywords
                .iter()
                .any(|k| prompt_lower.contains(&k.to_lowercase()));
            if !has_keyword {
                return false;
            }
        }

        // Check task type
        if !self.task_types.is_empty() {
            if let Some(ref task_type) = context.task_type {
                if !self.task_types.contains(task_type) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check complexity
        if !self.complexity_levels.is_empty() {
            if let Some(complexity) = context.complexity {
                if !self.complexity_levels.contains(&complexity) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// Rule-based model router that matches requests against a priority-ordered list of rules.
pub struct RuleBasedRouter {
    rules: Vec<RoutingRule>,
    default_model: String,
    default_provider: String,
}

impl RuleBasedRouter {
    /// Create a new rule-based router.
    pub fn new(default_model: impl Into<String>, default_provider: impl Into<String>) -> Self {
        Self {
            rules: Vec::new(),
            default_model: default_model.into(),
            default_provider: default_provider.into(),
        }
    }

    /// Add a routing rule.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Estimate complexity from prompt characteristics.
    pub fn estimate_complexity(prompt: &str) -> TaskComplexity {
        let word_count = prompt.split_whitespace().count();
        let has_code_markers = prompt.contains("```")
            || prompt.contains("fn ")
            || prompt.contains("def ")
            || prompt.contains("class ");
        let has_reasoning_markers = prompt.contains("explain")
            || prompt.contains("analyze")
            || prompt.contains("compare")
            || prompt.contains("why");

        if word_count < 10 && !has_code_markers && !has_reasoning_markers {
            TaskComplexity::Simple
        } else if has_code_markers || word_count > 100 {
            TaskComplexity::Complex
        } else if has_reasoning_markers || word_count > 50 {
            TaskComplexity::Moderate
        } else {
            TaskComplexity::Simple
        }
    }
}

#[async_trait]
impl Router for RuleBasedRouter {
    async fn route(&self, context: &RoutingContext) -> Result<RoutingDecision, RoutingError> {
        // Find matching rule with highest priority
        for rule in &self.rules {
            if rule.matches(context) {
                let estimated = context
                    .complexity
                    .unwrap_or_else(|| Self::estimate_complexity(&context.prompt));
                return Ok(RoutingDecision {
                    model: rule.target_model.clone(),
                    provider: rule.target_provider.clone(),
                    reason: format!("Matched rule: {}", rule.name),
                    estimated_complexity: estimated,
                    confidence: 0.9,
                });
            }
        }

        // Use default model
        let estimated = context
            .complexity
            .unwrap_or_else(|| Self::estimate_complexity(&context.prompt));
        Ok(RoutingDecision {
            model: self.default_model.clone(),
            provider: self.default_provider.clone(),
            reason: "Default fallback".to_string(),
            estimated_complexity: estimated,
            confidence: 0.5,
        })
    }
}

/// Complexity-based router that selects models based solely on estimated complexity.
pub struct ComplexityRouter {
    /// Model for simple tasks.
    pub simple_model: (String, String),
    /// Model for moderate tasks.
    pub moderate_model: (String, String),
    /// Model for complex tasks.
    pub complex_model: (String, String),
    /// Model for expert tasks.
    pub expert_model: (String, String),
}

impl ComplexityRouter {
    /// Create a new complexity router with models for each level.
    pub fn new(
        simple: (impl Into<String>, impl Into<String>),
        moderate: (impl Into<String>, impl Into<String>),
        complex: (impl Into<String>, impl Into<String>),
        expert: (impl Into<String>, impl Into<String>),
    ) -> Self {
        Self {
            simple_model: (simple.0.into(), simple.1.into()),
            moderate_model: (moderate.0.into(), moderate.1.into()),
            complex_model: (complex.0.into(), complex.1.into()),
            expert_model: (expert.0.into(), expert.1.into()),
        }
    }
}

#[async_trait]
impl Router for ComplexityRouter {
    async fn route(&self, context: &RoutingContext) -> Result<RoutingDecision, RoutingError> {
        let complexity = context
            .complexity
            .unwrap_or_else(|| RuleBasedRouter::estimate_complexity(&context.prompt));

        let (model, provider) = match complexity {
            TaskComplexity::Simple => &self.simple_model,
            TaskComplexity::Moderate => &self.moderate_model,
            TaskComplexity::Complex => &self.complex_model,
            TaskComplexity::Expert => &self.expert_model,
        };

        Ok(RoutingDecision {
            model: model.clone(),
            provider: provider.clone(),
            reason: format!("Complexity-based routing: {:?}", complexity),
            estimated_complexity: complexity,
            confidence: 0.8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rule_based_router_default_fallback() {
        let router = RuleBasedRouter::new("gpt-3.5-turbo", "openai");
        let context = RoutingContext::new("hi");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "gpt-3.5-turbo");
        assert_eq!(decision.provider, "openai");
        assert_eq!(decision.reason, "Default fallback");
    }

    #[tokio::test]
    async fn test_rule_based_router_keyword_match() {
        let mut router = RuleBasedRouter::new("gpt-3.5-turbo", "openai");
        router.add_rule(
            RoutingRule::new("code-rule", "gpt-4", "openai")
                .with_keywords(vec!["code".to_string(), "function".to_string()])
                .with_priority(10),
        );

        let context = RoutingContext::new("Write a function to sort an array");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "gpt-4");
        assert!(decision.reason.contains("code-rule"));
    }

    #[tokio::test]
    async fn test_rule_based_router_complexity_match() {
        let mut router = RuleBasedRouter::new("gpt-3.5-turbo", "openai");
        router.add_rule(
            RoutingRule::new("expert-rule", "claude-3-opus", "anthropic")
                .with_complexity(vec![TaskComplexity::Expert])
                .with_priority(5),
        );

        let context = RoutingContext::new("Explain quantum mechanics")
            .with_complexity(TaskComplexity::Expert);
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "claude-3-opus");
        assert_eq!(decision.provider, "anthropic");
    }

    #[tokio::test]
    async fn test_rule_based_router_task_type_match() {
        let mut router = RuleBasedRouter::new("gpt-3.5-turbo", "openai");
        router.add_rule(
            RoutingRule::new("chat-rule", "gpt-4o-mini", "openai")
                .with_task_types(vec!["chat".to_string()])
                .with_priority(20),
        );

        let context = RoutingContext::new("Hello, how are you?").with_task_type("chat");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "gpt-4o-mini");
    }

    #[tokio::test]
    async fn test_rule_based_router_max_length() {
        let mut router = RuleBasedRouter::new("gpt-4", "openai");
        router.add_rule(
            RoutingRule::new("short-rule", "gpt-3.5-turbo", "openai")
                .with_max_length(20)
                .with_priority(10),
        );

        // Short prompt should match
        let context = RoutingContext::new("hi");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "gpt-3.5-turbo");

        // Long prompt should fall to default
        let context =
            RoutingContext::new("This is a much longer prompt that exceeds twenty characters");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_rule_priority_ordering() {
        let mut router = RuleBasedRouter::new("default", "openai");
        router.add_rule(
            RoutingRule::new("low-priority", "model-b", "openai")
                .with_keywords(vec!["test".to_string()])
                .with_priority(100),
        );
        router.add_rule(
            RoutingRule::new("high-priority", "model-a", "openai")
                .with_keywords(vec!["test".to_string()])
                .with_priority(1),
        );

        let context = RoutingContext::new("this is a test");
        let decision = router.route(&context).await.unwrap();
        assert_eq!(decision.model, "model-a");
    }

    #[tokio::test]
    async fn test_complexity_estimation() {
        assert_eq!(
            RuleBasedRouter::estimate_complexity("hi"),
            TaskComplexity::Simple
        );
        assert_eq!(
            RuleBasedRouter::estimate_complexity(
                "explain why the sky is blue and compare it to other atmospheric phenomena"
            ),
            TaskComplexity::Moderate
        );
        assert_eq!(
            RuleBasedRouter::estimate_complexity(
                "```rust\nfn main() {\n    println!(\"hello\");\n}\n```"
            ),
            TaskComplexity::Complex
        );
    }

    #[tokio::test]
    async fn test_complexity_router() {
        let router = ComplexityRouter::new(
            ("gpt-3.5-turbo", "openai"),
            ("gpt-4o-mini", "openai"),
            ("gpt-4", "openai"),
            ("claude-3-opus", "anthropic"),
        );

        let simple = RoutingContext::new("hi").with_complexity(TaskComplexity::Simple);
        let decision = router.route(&simple).await.unwrap();
        assert_eq!(decision.model, "gpt-3.5-turbo");

        let expert = RoutingContext::new("complex task").with_complexity(TaskComplexity::Expert);
        let decision = router.route(&expert).await.unwrap();
        assert_eq!(decision.model, "claude-3-opus");
        assert_eq!(decision.provider, "anthropic");
    }

    #[tokio::test]
    async fn test_routing_context_builder() {
        let context = RoutingContext::new("test prompt")
            .with_complexity(TaskComplexity::Moderate)
            .with_task_type("code")
            .with_max_latency_ms(500)
            .with_max_cost(0.01);

        assert_eq!(context.prompt, "test prompt");
        assert_eq!(context.complexity, Some(TaskComplexity::Moderate));
        assert_eq!(context.task_type, Some("code".to_string()));
        assert_eq!(context.max_latency_ms, Some(500));
        assert_eq!(context.max_cost, Some(0.01));
    }

    #[tokio::test]
    async fn test_routing_decision_fields() {
        let router = RuleBasedRouter::new("gpt-4", "openai");
        let context = RoutingContext::new("hello").with_complexity(TaskComplexity::Simple);
        let decision = router.route(&context).await.unwrap();

        assert_eq!(decision.estimated_complexity, TaskComplexity::Simple);
        assert!(decision.confidence > 0.0);
        assert!(!decision.reason.is_empty());
    }
}
