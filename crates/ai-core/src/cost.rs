// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Cost tracking for LLM API usage.
//!
//! This module provides utilities for tracking token usage and calculating costs
//! across different scopes: request, agent, workflow, and global.

use crate::types::Usage;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Global aggregate scope name.
pub const GLOBAL_SCOPE: &str = "global";

/// Build a request scope key.
pub fn request_scope(request_id: &str) -> String {
    format!("request:{}", request_id)
}

/// Build an agent scope key.
pub fn agent_scope(agent_name: &str) -> String {
    format!("agent:{}", agent_name)
}

/// Build a workflow scope key.
pub fn workflow_scope(workflow_name: &str) -> String {
    format!("workflow:{}", workflow_name)
}

/// Build a project scope key.
pub fn project_scope(project_name: &str) -> String {
    format!("project:{}", project_name)
}

/// Generate a new request identifier.
pub fn new_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// Pricing information for a specific model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPricing {
    /// Price per million prompt tokens in USD.
    pub prompt_price_per_m: f64,

    /// Price per million completion tokens in USD.
    pub completion_price_per_m: f64,
}

impl ModelPricing {
    /// Create new model pricing.
    pub fn new(prompt_price_per_m: f64, completion_price_per_m: f64) -> Self {
        Self {
            prompt_price_per_m,
            completion_price_per_m,
        }
    }

    /// Calculate cost for a given usage.
    pub fn calculate_cost(&self, usage: &Usage) -> f64 {
        usage.estimated_cost(self.prompt_price_per_m, self.completion_price_per_m)
    }
}

/// Pricing table for common models.
///
/// Prices are approximate and may change. Always check provider documentation
/// for current pricing.
#[derive(Debug, Clone)]
pub struct PricingTable {
    /// Map of model name to pricing information.
    prices: HashMap<String, ModelPricing>,
}

impl Default for PricingTable {
    fn default() -> Self {
        Self::with_standard_pricing()
    }
}

impl PricingTable {
    /// Create a pricing table with standard model prices (as of 2024).
    ///
    /// Prices are in USD per million tokens.
    pub fn with_standard_pricing() -> Self {
        let mut prices = HashMap::new();

        // OpenAI pricing (approximate, as of 2024)
        prices.insert("gpt-4".to_string(), ModelPricing::new(30.0, 60.0));
        prices.insert("gpt-4-32k".to_string(), ModelPricing::new(60.0, 120.0));
        prices.insert("gpt-4-turbo".to_string(), ModelPricing::new(10.0, 30.0));
        prices.insert("gpt-3.5-turbo".to_string(), ModelPricing::new(0.5, 1.5));
        prices.insert("gpt-3.5-turbo-16k".to_string(), ModelPricing::new(0.5, 1.5));

        // Anthropic pricing (approximate, as of 2024)
        prices.insert(
            "claude-3-opus-20240229".to_string(),
            ModelPricing::new(15.0, 75.0),
        );
        prices.insert(
            "claude-3-sonnet-20240229".to_string(),
            ModelPricing::new(3.0, 15.0),
        );
        prices.insert(
            "claude-3-haiku-20240307".to_string(),
            ModelPricing::new(0.25, 1.25),
        );
        prices.insert("claude-2".to_string(), ModelPricing::new(8.0, 24.0));
        prices.insert("claude-instant-1".to_string(), ModelPricing::new(0.8, 2.4));

        // Google pricing (approximate, as of 2024)
        prices.insert("gemini-pro".to_string(), ModelPricing::new(0.5, 1.5));
        prices.insert("gemini-ultra".to_string(), ModelPricing::new(2.0, 8.0));

        Self { prices }
    }

    /// Create an empty pricing table.
    pub fn empty() -> Self {
        Self {
            prices: HashMap::new(),
        }
    }

    /// Add or update pricing for a model.
    pub fn set_pricing(&mut self, model: impl Into<String>, pricing: ModelPricing) {
        self.prices.insert(model.into(), pricing);
    }

    /// Get pricing for a model, if available.
    pub fn get_pricing(&self, model: &str) -> Option<ModelPricing> {
        // Try exact match first
        if let Some(pricing) = self.prices.get(model) {
            return Some(*pricing);
        }

        // Try prefix matching for model families
        for (model_name, pricing) in &self.prices {
            if model.starts_with(model_name) {
                return Some(*pricing);
            }
        }

        None
    }

    /// Check if a model has pricing information.
    pub fn has_pricing(&self, model: &str) -> bool {
        self.get_pricing(model).is_some()
    }

    /// Get all model names in the pricing table.
    pub fn models(&self) -> Vec<&str> {
        self.prices.keys().map(|k| k.as_str()).collect()
    }
}

/// Cost tracking for a single scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostSnapshot {
    /// Total prompt tokens used.
    pub prompt_tokens: u64,

    /// Total completion tokens used.
    pub completion_tokens: u64,

    /// Total cost in USD.
    pub total_cost: f64,

    /// Number of requests/calls made.
    pub request_count: u64,
}

impl CostSnapshot {
    /// Create a new cost snapshot.
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_cost: 0.0,
            request_count: 0,
        }
    }

    /// Get total tokens used.
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }

    /// Calculate average cost per request.
    pub fn avg_cost_per_request(&self) -> f64 {
        if self.request_count == 0 {
            return 0.0;
        }
        self.total_cost / self.request_count as f64
    }

    /// Calculate average tokens per request.
    pub fn avg_tokens_per_request(&self) -> u64 {
        if self.request_count == 0 {
            return 0;
        }
        self.total_tokens() / self.request_count
    }
}

impl Default for CostSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulator for tracking costs across operations.
#[derive(Debug, Clone)]
pub struct CostAccumulator {
    /// Cost tracking at different scopes.
    scopes: HashMap<String, CostSnapshot>,
}

impl Default for CostAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostAccumulator {
    /// Create a new cost accumulator.
    pub fn new() -> Self {
        Self {
            scopes: HashMap::new(),
        }
    }

    /// Record usage for a model at a specific scope.
    pub fn record(
        &mut self,
        scope: impl Into<String>,
        model: &str,
        usage: &Usage,
        pricing_table: &PricingTable,
    ) {
        let scope = scope.into();
        let snapshot = self.scopes.entry(scope).or_insert_with(CostSnapshot::new);

        snapshot.prompt_tokens += usage.prompt_tokens as u64;
        snapshot.completion_tokens += usage.completion_tokens as u64;
        snapshot.request_count += 1;

        // Add cost if pricing is available
        if let Some(pricing) = pricing_table.get_pricing(model) {
            snapshot.total_cost += pricing.calculate_cost(usage);
        }
    }

    /// Record usage across multiple scopes.
    pub fn record_many<I, S>(
        &mut self,
        scopes: I,
        model: &str,
        usage: &Usage,
        pricing_table: &PricingTable,
    ) where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen = HashSet::new();

        for scope in scopes {
            let scope = scope.into();
            if seen.insert(scope.clone()) {
                self.record(scope, model, usage, pricing_table);
            }
        }
    }

    /// Get cost snapshot for a scope.
    pub fn get(&self, scope: &str) -> Option<&CostSnapshot> {
        self.scopes.get(scope)
    }

    /// Get cost snapshot for a scope, or empty if not found.
    pub fn get_or_default(&self, scope: &str) -> CostSnapshot {
        self.get(scope).copied().unwrap_or_default()
    }

    /// Get all scopes being tracked.
    pub fn scopes(&self) -> Vec<&str> {
        self.scopes.keys().map(|k| k.as_str()).collect()
    }

    /// Get total costs across all scopes.
    pub fn total(&self) -> CostSnapshot {
        let mut total = CostSnapshot::new();
        for snapshot in self.scopes.values() {
            total.prompt_tokens += snapshot.prompt_tokens;
            total.completion_tokens += snapshot.completion_tokens;
            total.total_cost += snapshot.total_cost;
            total.request_count += snapshot.request_count;
        }
        total
    }

    /// Reset all tracking for a scope.
    pub fn reset(&mut self, scope: &str) {
        self.scopes.remove(scope);
    }

    /// Reset all tracking.
    pub fn reset_all(&mut self) {
        self.scopes.clear();
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: CostAccumulator) {
        for (scope, other_snapshot) in other.scopes {
            let snapshot = self.scopes.entry(scope).or_insert_with(CostSnapshot::new);
            snapshot.prompt_tokens += other_snapshot.prompt_tokens;
            snapshot.completion_tokens += other_snapshot.completion_tokens;
            snapshot.total_cost += other_snapshot.total_cost;
            snapshot.request_count += other_snapshot.request_count;
        }
    }
}

/// Thread-safe, shared cost tracker.
#[derive(Debug, Clone)]
pub struct CostTracker {
    inner: Arc<RwLock<CostAccumulator>>,
    pricing_table: PricingTable,
}

impl CostTracker {
    /// Create a new cost tracker with default pricing.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CostAccumulator::new())),
            pricing_table: PricingTable::default(),
        }
    }

    /// Create a new cost tracker with custom pricing.
    pub fn with_pricing(pricing_table: PricingTable) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CostAccumulator::new())),
            pricing_table,
        }
    }

    /// Record usage for a model at a specific scope.
    pub async fn record(&self, scope: impl Into<String>, model: &str, usage: &Usage) {
        let scope = scope.into();
        let mut inner = self.inner.write().await;
        inner.record(&scope, model, usage, &self.pricing_table);
    }

    /// Record usage for multiple scopes at once.
    pub async fn record_many<I, S>(&self, scopes: I, model: &str, usage: &Usage)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut inner = self.inner.write().await;
        inner.record_many(scopes, model, usage, &self.pricing_table);
    }

    /// Get cost snapshot for a scope.
    pub async fn get(&self, scope: &str) -> CostSnapshot {
        let inner = self.inner.read().await;
        inner.get_or_default(scope)
    }

    /// Get total costs across all scopes.
    pub async fn total(&self) -> CostSnapshot {
        let inner = self.inner.read().await;
        inner.total()
    }

    /// Get all scopes being tracked.
    pub async fn scopes(&self) -> Vec<String> {
        let inner = self.inner.read().await;
        inner.scopes().into_iter().map(String::from).collect()
    }

    /// Reset tracking for a scope.
    pub async fn reset(&self, scope: &str) {
        let mut inner = self.inner.write().await;
        inner.reset(scope);
    }

    /// Reset all tracking.
    pub async fn reset_all(&self) {
        let mut inner = self.inner.write().await;
        inner.reset_all();
    }

    /// Get a reference to the pricing table.
    pub fn pricing_table(&self) -> &PricingTable {
        &self.pricing_table
    }

    /// Estimate the cost of a single response usage record.
    pub fn estimate_cost(&self, model: &str, usage: &Usage) -> f64 {
        self.pricing_table
            .get_pricing(model)
            .map(|pricing| pricing.calculate_cost(usage))
            .unwrap_or(0.0)
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_calculate_cost() {
        let pricing = ModelPricing::new(10.0, 20.0);
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        let cost = pricing.calculate_cost(&usage);
        assert!((cost - 0.02).abs() < 0.0001); // $0.01 for prompts + $0.01 for completions
    }

    #[test]
    fn test_pricing_table_default() {
        let table = PricingTable::default();
        assert!(table.has_pricing("gpt-4"));
        assert!(table.has_pricing("claude-3-opus-20240229"));
        assert!(table.has_pricing("gemini-pro"));
    }

    #[test]
    fn test_pricing_table_custom() {
        let mut table = PricingTable::empty();
        table.set_pricing("custom-model", ModelPricing::new(5.0, 10.0));

        assert!(table.has_pricing("custom-model"));
        let pricing = table.get_pricing("custom-model").unwrap();
        assert_eq!(pricing.prompt_price_per_m, 5.0);
        assert_eq!(pricing.completion_price_per_m, 10.0);
    }

    #[test]
    fn test_pricing_table_prefix_matching() {
        let table = PricingTable::default();
        // Should match gpt-4 even with suffixes
        assert!(table.has_pricing("gpt-4-turbo-preview"));
        assert!(table.has_pricing("gpt-4-32k-0314"));
    }

    #[test]
    fn test_cost_snapshot() {
        let mut snapshot = CostSnapshot::new();
        snapshot.prompt_tokens = 1000;
        snapshot.completion_tokens = 500;
        snapshot.total_cost = 0.05;
        snapshot.request_count = 5;

        assert_eq!(snapshot.total_tokens(), 1500);
        assert_eq!(snapshot.avg_cost_per_request(), 0.01);
        assert_eq!(snapshot.avg_tokens_per_request(), 300);
    }

    #[test]
    fn test_cost_snapshot_empty() {
        let snapshot = CostSnapshot::new();
        assert_eq!(snapshot.total_tokens(), 0);
        assert_eq!(snapshot.avg_cost_per_request(), 0.0);
        assert_eq!(snapshot.avg_tokens_per_request(), 0);
    }

    #[test]
    fn test_cost_accumulator_record() {
        let mut accumulator = CostAccumulator::new();
        let pricing_table = PricingTable::default();

        let usage1 = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        let usage2 = Usage {
            prompt_tokens: 2000,
            completion_tokens: 1000,
            total_tokens: 3000,
        };

        accumulator.record("agent-1", "gpt-4", &usage1, &pricing_table);
        accumulator.record("agent-1", "gpt-4", &usage2, &pricing_table);

        let snapshot = accumulator.get_or_default("agent-1");
        assert_eq!(snapshot.prompt_tokens, 3000);
        assert_eq!(snapshot.completion_tokens, 1500);
        assert_eq!(snapshot.request_count, 2);
        assert!(snapshot.total_cost > 0.0);
    }

    #[test]
    fn test_cost_accumulator_multiple_scopes() {
        let mut accumulator = CostAccumulator::new();
        let pricing_table = PricingTable::default();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        accumulator.record("agent-1", "gpt-4", &usage, &pricing_table);
        accumulator.record("agent-2", "gpt-4", &usage, &pricing_table);
        accumulator.record("workflow-1", "gpt-4", &usage, &pricing_table);

        assert_eq!(accumulator.scopes().len(), 3);

        let total = accumulator.total();
        assert_eq!(total.request_count, 3);
        assert_eq!(total.prompt_tokens, 3000);
        assert_eq!(total.completion_tokens, 1500);
    }

    #[test]
    fn test_cost_accumulator_record_many_deduplicates_scopes() {
        let mut accumulator = CostAccumulator::new();
        let pricing_table = PricingTable::default();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        accumulator.record_many(
            ["agent:test", "agent:test", GLOBAL_SCOPE],
            "gpt-4",
            &usage,
            &pricing_table,
        );

        assert_eq!(accumulator.get_or_default("agent:test").request_count, 1);
        assert_eq!(accumulator.get_or_default(GLOBAL_SCOPE).request_count, 1);
    }

    #[test]
    fn test_cost_accumulator_reset() {
        let mut accumulator = CostAccumulator::new();
        let pricing_table = PricingTable::default();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        accumulator.record("agent-1", "gpt-4", &usage, &pricing_table);
        assert!(accumulator.get("agent-1").is_some());

        accumulator.reset("agent-1");
        assert!(accumulator.get("agent-1").is_none());
    }

    #[test]
    fn test_cost_accumulator_merge() {
        let mut acc1 = CostAccumulator::new();
        let mut acc2 = CostAccumulator::new();
        let pricing_table = PricingTable::default();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        acc1.record("agent-1", "gpt-4", &usage, &pricing_table);
        acc2.record("agent-1", "gpt-4", &usage, &pricing_table);
        acc2.record("agent-2", "gpt-4", &usage, &pricing_table);

        acc1.merge(acc2);

        let snapshot1 = acc1.get_or_default("agent-1");
        assert_eq!(snapshot1.request_count, 2);

        let snapshot2 = acc1.get_or_default("agent-2");
        assert_eq!(snapshot2.request_count, 1);
    }

    #[tokio::test]
    async fn test_cost_tracker_record() {
        let tracker = CostTracker::new();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        tracker.record("agent-1", "gpt-4", &usage).await;

        let snapshot = tracker.get("agent-1").await;
        assert_eq!(snapshot.prompt_tokens, 1000);
        assert_eq!(snapshot.completion_tokens, 500);
        assert_eq!(snapshot.request_count, 1);
    }

    #[tokio::test]
    async fn test_cost_tracker_total() {
        let tracker = CostTracker::new();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        tracker.record("agent-1", "gpt-4", &usage).await;
        tracker.record("agent-2", "gpt-4", &usage).await;

        let total = tracker.total().await;
        assert_eq!(total.request_count, 2);
        assert_eq!(total.prompt_tokens, 2000);
        assert_eq!(total.completion_tokens, 1000);
    }

    #[tokio::test]
    async fn test_cost_tracker_scopes() {
        let tracker = CostTracker::new();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        tracker.record("agent-1", "gpt-4", &usage).await;
        tracker.record("agent-2", "gpt-4", &usage).await;

        let scopes = tracker.scopes().await;
        assert_eq!(scopes.len(), 2);
        assert!(scopes.contains(&"agent-1".to_string()));
        assert!(scopes.contains(&"agent-2".to_string()));
    }

    #[tokio::test]
    async fn test_cost_tracker_reset() {
        let tracker = CostTracker::new();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        tracker.record("agent-1", "gpt-4", &usage).await;

        tracker.reset("agent-1").await;

        let snapshot = tracker.get("agent-1").await;
        assert_eq!(snapshot.prompt_tokens, 0);
        assert_eq!(snapshot.completion_tokens, 0);
    }

    #[tokio::test]
    async fn test_cost_tracker_with_custom_pricing() {
        let mut pricing_table = PricingTable::empty();
        pricing_table.set_pricing("custom-model", ModelPricing::new(1.0, 2.0));

        let tracker = CostTracker::with_pricing(pricing_table);

        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 500_000,
            total_tokens: 1_500_000,
        };

        tracker.record("test", "custom-model", &usage).await;

        let snapshot = tracker.get("test").await;
        // Cost should be $1 for prompts + $1 for completions = $2
        assert!((snapshot.total_cost - 2.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cost_tracker_record_many() {
        let tracker = CostTracker::new();

        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };

        tracker
            .record_many(["request:abc", "agent:test", GLOBAL_SCOPE], "gpt-4", &usage)
            .await;

        assert_eq!(tracker.get("request:abc").await.request_count, 1);
        assert_eq!(tracker.get("agent:test").await.request_count, 1);
        assert_eq!(tracker.get(GLOBAL_SCOPE).await.request_count, 1);
    }
}
