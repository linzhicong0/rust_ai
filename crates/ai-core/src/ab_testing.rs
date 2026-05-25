// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # A/B Testing (REQ-10.2)
//!
//! Provides A/B testing of prompts, models, and agent configurations with
//! statistical significance analysis. Includes variant assignment, traffic
//! splitting, metric collection per variant, and statistical significance
//! calculation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during A/B testing.
#[derive(Debug, Error)]
pub enum AbTestError {
    /// Invalid test configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Test not found.
    #[error("Test not found: {0}")]
    NotFound(String),

    /// Variant not found.
    #[error("Variant not found: {0}")]
    VariantNotFound(String),
}

/// A variant in an A/B test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestVariant {
    /// Unique identifier for this variant.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Traffic weight (relative to other variants).
    pub weight: f64,
    /// Arbitrary configuration for this variant (e.g., model name, prompt template).
    pub config: HashMap<String, String>,
}

impl TestVariant {
    /// Create a new variant.
    pub fn new(id: impl Into<String>, name: impl Into<String>, weight: f64) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            weight,
            config: HashMap::new(),
        }
    }

    /// Add a config parameter.
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }
}

/// Configuration for an A/B test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestConfig {
    /// Unique test identifier.
    pub test_id: String,
    /// Human-readable test name.
    pub name: String,
    /// The variants to test.
    pub variants: Vec<TestVariant>,
    /// Minimum sample size per variant for significance.
    pub min_sample_size: usize,
    /// Significance level (alpha), typically 0.05.
    pub significance_level: f64,
}

impl AbTestConfig {
    /// Create a new A/B test with two variants (50/50 split).
    pub fn new(
        test_id: impl Into<String>,
        name: impl Into<String>,
        control: TestVariant,
        treatment: TestVariant,
    ) -> Self {
        Self {
            test_id: test_id.into(),
            name: name.into(),
            variants: vec![control, treatment],
            min_sample_size: 30,
            significance_level: 0.05,
        }
    }

    /// Create a test with multiple variants.
    pub fn with_variants(
        test_id: impl Into<String>,
        name: impl Into<String>,
        variants: Vec<TestVariant>,
    ) -> Self {
        Self {
            test_id: test_id.into(),
            name: name.into(),
            variants,
            min_sample_size: 30,
            significance_level: 0.05,
        }
    }

    /// Set minimum sample size.
    pub fn with_min_sample_size(mut self, n: usize) -> Self {
        self.min_sample_size = n;
        self
    }

    /// Set significance level.
    pub fn with_significance_level(mut self, alpha: f64) -> Self {
        self.significance_level = alpha.clamp(0.001, 0.5);
        self
    }

    /// Get total weight across all variants.
    pub fn total_weight(&self) -> f64 {
        self.variants.iter().map(|v| v.weight).sum()
    }

    /// Validate the test configuration.
    pub fn validate(&self) -> Result<(), AbTestError> {
        if self.variants.len() < 2 {
            return Err(AbTestError::InvalidConfig(
                "A/B test requires at least 2 variants".to_string(),
            ));
        }
        if self.total_weight() <= 0.0 {
            return Err(AbTestError::InvalidConfig(
                "Total weight must be positive".to_string(),
            ));
        }
        let ids: Vec<&str> = self.variants.iter().map(|v| v.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            return Err(AbTestError::InvalidConfig(
                "Variant IDs must be unique".to_string(),
            ));
        }
        Ok(())
    }
}

/// Assigns users/requests to variants based on a deterministic hash.
pub struct VariantAssigner {
    config: AbTestConfig,
}

impl VariantAssigner {
    /// Create a new assigner from a test config.
    pub fn new(config: AbTestConfig) -> Result<Self, AbTestError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Assign a variant based on a user/request identifier.
    ///
    /// Uses a simple hash-based approach for deterministic assignment.
    pub fn assign(&self, user_id: &str) -> &TestVariant {
        let hash = simple_hash(user_id);
        let total_weight = self.config.total_weight();
        let normalized = (hash as f64) / (u64::MAX as f64) * total_weight;

        let mut cumulative = 0.0;
        for variant in &self.config.variants {
            cumulative += variant.weight;
            if normalized < cumulative {
                return variant;
            }
        }

        // Fallback to last variant (shouldn't happen with valid config)
        self.config.variants.last().unwrap()
    }

    /// Get the test config.
    pub fn config(&self) -> &AbTestConfig {
        &self.config
    }
}

/// Simple deterministic hash function for variant assignment (FNV-1a inspired).
fn simple_hash(input: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Collected metrics for a single observation in a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// The metric value (e.g., score, latency, success=1/failure=0).
    pub value: f64,
    /// Timestamp of the observation.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Aggregated metrics for a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantMetrics {
    /// Variant ID.
    pub variant_id: String,
    /// Number of observations.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Sum of squared values (for variance calculation).
    pub sum_sq: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
}

impl VariantMetrics {
    /// Create empty metrics for a variant.
    pub fn new(variant_id: impl Into<String>) -> Self {
        Self {
            variant_id: variant_id.into(),
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Record an observation.
    pub fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Compute the mean.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Compute the sample variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_sq - (self.sum * self.sum) / n) / (n - 1.0)
    }

    /// Compute the sample standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Result of a statistical significance test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceResult {
    /// Z-score of the test.
    pub z_score: f64,
    /// P-value (two-tailed).
    pub p_value: f64,
    /// Whether the result is statistically significant.
    pub is_significant: bool,
    /// Confidence level (1 - p_value).
    pub confidence: f64,
    /// Effect size (difference in means).
    pub effect_size: f64,
}

/// Compute statistical significance between two variants using a two-sample Z-test.
///
/// Returns None if either variant has insufficient data.
pub fn compute_significance(
    control: &VariantMetrics,
    treatment: &VariantMetrics,
    alpha: f64,
) -> Option<SignificanceResult> {
    if control.count < 2 || treatment.count < 2 {
        return None;
    }

    let mean_diff = treatment.mean() - control.mean();
    let se = ((control.variance() / control.count as f64)
        + (treatment.variance() / treatment.count as f64))
        .sqrt();

    if se == 0.0 {
        return Some(SignificanceResult {
            z_score: 0.0,
            p_value: 1.0,
            is_significant: false,
            confidence: 0.0,
            effect_size: mean_diff,
        });
    }

    let z_score = mean_diff / se;
    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));
    let is_significant = p_value < alpha;

    Some(SignificanceResult {
        z_score,
        p_value,
        is_significant,
        confidence: 1.0 - p_value,
        effect_size: mean_diff,
    })
}

/// Approximation of the standard normal CDF using the Abramowitz and Stegun formula.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804 * (-x * x / 2.0).exp();
    let p =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// A/B test metric collector that tracks metrics per variant.
pub struct AbTestCollector {
    test_id: String,
    metrics: HashMap<String, VariantMetrics>,
    significance_level: f64,
}

impl AbTestCollector {
    /// Create a new collector for a test.
    pub fn new(config: &AbTestConfig) -> Self {
        let mut metrics = HashMap::new();
        for variant in &config.variants {
            metrics.insert(variant.id.clone(), VariantMetrics::new(&variant.id));
        }
        Self {
            test_id: config.test_id.clone(),
            metrics,
            significance_level: config.significance_level,
        }
    }

    /// Record a metric value for a variant.
    pub fn record(&mut self, variant_id: &str, value: f64) -> Result<(), AbTestError> {
        self.metrics
            .get_mut(variant_id)
            .ok_or_else(|| AbTestError::VariantNotFound(variant_id.to_string()))?
            .record(value);
        Ok(())
    }

    /// Get metrics for a variant.
    pub fn get_metrics(&self, variant_id: &str) -> Option<&VariantMetrics> {
        self.metrics.get(variant_id)
    }

    /// Get the test ID.
    pub fn test_id(&self) -> &str {
        &self.test_id
    }

    /// Compute significance between two variants.
    pub fn significance(
        &self,
        control_id: &str,
        treatment_id: &str,
    ) -> Result<Option<SignificanceResult>, AbTestError> {
        let control = self
            .metrics
            .get(control_id)
            .ok_or_else(|| AbTestError::VariantNotFound(control_id.to_string()))?;
        let treatment = self
            .metrics
            .get(treatment_id)
            .ok_or_else(|| AbTestError::VariantNotFound(treatment_id.to_string()))?;

        Ok(compute_significance(
            control,
            treatment,
            self.significance_level,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-10.2: A/B Testing tests

    #[test]
    fn test_variant_creation() {
        let variant = TestVariant::new("control", "Control Group", 0.5)
            .with_config("model", "gpt-4")
            .with_config("temperature", "0.7");

        assert_eq!(variant.id, "control");
        assert_eq!(variant.name, "Control Group");
        assert_eq!(variant.weight, 0.5);
        assert_eq!(variant.config.get("model").unwrap(), "gpt-4");
        assert_eq!(variant.config.get("temperature").unwrap(), "0.7");
    }

    #[test]
    fn test_ab_test_config_creation() {
        let control = TestVariant::new("control", "Control", 0.5);
        let treatment = TestVariant::new("treatment", "Treatment", 0.5);
        let config = AbTestConfig::new("test-1", "Model Comparison", control, treatment);

        assert_eq!(config.test_id, "test-1");
        assert_eq!(config.variants.len(), 2);
        assert_eq!(config.total_weight(), 1.0);
        assert_eq!(config.min_sample_size, 30);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_ab_test_config_validation_success() {
        let control = TestVariant::new("a", "A", 0.5);
        let treatment = TestVariant::new("b", "B", 0.5);
        let config = AbTestConfig::new("t", "Test", control, treatment);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ab_test_config_validation_too_few_variants() {
        let config = AbTestConfig {
            test_id: "t".to_string(),
            name: "Test".to_string(),
            variants: vec![TestVariant::new("a", "A", 1.0)],
            min_sample_size: 30,
            significance_level: 0.05,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ab_test_config_validation_duplicate_ids() {
        let config = AbTestConfig::with_variants(
            "t",
            "Test",
            vec![
                TestVariant::new("a", "A", 0.5),
                TestVariant::new("a", "B", 0.5), // duplicate ID
            ],
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ab_test_config_significance_level_clamped() {
        let control = TestVariant::new("a", "A", 0.5);
        let treatment = TestVariant::new("b", "B", 0.5);
        let config =
            AbTestConfig::new("t", "Test", control, treatment).with_significance_level(0.0);
        assert_eq!(config.significance_level, 0.001);

        let control2 = TestVariant::new("a", "A", 0.5);
        let treatment2 = TestVariant::new("b", "B", 0.5);
        let config2 =
            AbTestConfig::new("t", "Test", control2, treatment2).with_significance_level(1.0);
        assert_eq!(config2.significance_level, 0.5);
    }

    #[test]
    fn test_variant_assigner_deterministic() {
        let control = TestVariant::new("control", "Control", 0.5);
        let treatment = TestVariant::new("treatment", "Treatment", 0.5);
        let config = AbTestConfig::new("test", "Test", control, treatment);
        let assigner = VariantAssigner::new(config).unwrap();

        // Same user should always get the same variant
        let v1 = assigner.assign("user-123");
        let v2 = assigner.assign("user-123");
        assert_eq!(v1.id, v2.id);
    }

    #[test]
    fn test_variant_assigner_distribution() {
        let control = TestVariant::new("control", "Control", 0.5);
        let treatment = TestVariant::new("treatment", "Treatment", 0.5);
        let config = AbTestConfig::new("test", "Test", control, treatment);
        let assigner = VariantAssigner::new(config).unwrap();

        let mut counts: HashMap<&str, usize> = HashMap::new();
        for i in 0..1000 {
            let variant = assigner.assign(&format!("user-{}", i));
            *counts.entry(variant.id.as_str()).or_default() += 1;
        }

        // With 50/50 split, each should get roughly 500 (allow ±15%)
        let control_count = *counts.get("control").unwrap_or(&0);
        let treatment_count = *counts.get("treatment").unwrap_or(&0);
        assert!(control_count > 350, "control: {}", control_count);
        assert!(treatment_count > 350, "treatment: {}", treatment_count);
    }

    #[test]
    fn test_variant_assigner_weighted() {
        let a = TestVariant::new("a", "A", 0.8);
        let b = TestVariant::new("b", "B", 0.2);
        let config = AbTestConfig::new("test", "Test", a, b);
        let assigner = VariantAssigner::new(config).unwrap();

        let mut counts: HashMap<&str, usize> = HashMap::new();
        for i in 0..1000 {
            let variant = assigner.assign(&format!("user-{}", i));
            *counts.entry(variant.id.as_str()).or_default() += 1;
        }

        let a_count = *counts.get("a").unwrap_or(&0);
        let b_count = *counts.get("b").unwrap_or(&0);
        // 80% weight should give more to "a" than to "b"
        assert!(a_count > b_count, "a={}, b={}", a_count, b_count);
        // a should be at least 60% of total
        assert!(a_count > 600, "a: {}", a_count);
    }

    #[test]
    fn test_variant_metrics_basic() {
        let mut metrics = VariantMetrics::new("control");
        assert_eq!(metrics.count, 0);
        assert_eq!(metrics.mean(), 0.0);

        metrics.record(1.0);
        metrics.record(2.0);
        metrics.record(3.0);

        assert_eq!(metrics.count, 3);
        assert_eq!(metrics.sum, 6.0);
        assert_eq!(metrics.mean(), 2.0);
        assert_eq!(metrics.min, 1.0);
        assert_eq!(metrics.max, 3.0);
    }

    #[test]
    fn test_variant_metrics_variance() {
        let mut metrics = VariantMetrics::new("test");
        metrics.record(2.0);
        metrics.record(4.0);
        metrics.record(4.0);
        metrics.record(4.0);
        metrics.record(5.0);
        metrics.record(5.0);
        metrics.record(7.0);
        metrics.record(9.0);

        let mean = metrics.mean();
        assert!((mean - 5.0).abs() < 0.001);

        let variance = metrics.variance();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_variant_metrics_single_observation() {
        let mut metrics = VariantMetrics::new("test");
        metrics.record(5.0);

        assert_eq!(metrics.count, 1);
        assert_eq!(metrics.mean(), 5.0);
        assert_eq!(metrics.variance(), 0.0); // Can't compute with n < 2
    }

    #[test]
    fn test_significance_computation() {
        // Control: mean ~0.5, treatment: mean ~0.8
        let mut control = VariantMetrics::new("control");
        let mut treatment = VariantMetrics::new("treatment");

        // Add enough samples for significance
        for _ in 0..50 {
            control.record(0.5);
            treatment.record(0.8);
        }

        let result = compute_significance(&control, &treatment, 0.05).unwrap();
        // With zero variance this is a degenerate case - both groups are constant
        // The means differ by 0.3, so effect size should be 0.3
        assert!((result.effect_size - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_significance_insufficient_data() {
        let mut control = VariantMetrics::new("control");
        let treatment = VariantMetrics::new("treatment");

        control.record(1.0);
        // treatment has no observations

        let result = compute_significance(&control, &treatment, 0.05);
        assert!(result.is_none());
    }

    #[test]
    fn test_significance_with_variance() {
        let mut control = VariantMetrics::new("control");
        let mut treatment = VariantMetrics::new("treatment");

        // Control: ~0.5 with some noise
        for i in 0..100 {
            control.record(0.4 + (i % 3) as f64 * 0.1);
        }
        // Treatment: ~0.7 with some noise
        for i in 0..100 {
            treatment.record(0.6 + (i % 3) as f64 * 0.1);
        }

        let result = compute_significance(&control, &treatment, 0.05).unwrap();
        assert!(result.effect_size > 0.0); // treatment is better
        assert!(result.is_significant); // should be significant with 100 samples
    }

    #[test]
    fn test_significance_no_difference() {
        let mut control = VariantMetrics::new("control");
        let mut treatment = VariantMetrics::new("treatment");

        // Both groups: same distribution
        for i in 0..50 {
            let val = (i % 5) as f64 * 0.2;
            control.record(val);
            treatment.record(val);
        }

        let result = compute_significance(&control, &treatment, 0.05).unwrap();
        assert!(!result.is_significant);
        assert!((result.effect_size).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_collector() {
        let control = TestVariant::new("control", "Control", 0.5);
        let treatment = TestVariant::new("treatment", "Treatment", 0.5);
        let config = AbTestConfig::new("test-1", "Test", control, treatment);

        let mut collector = AbTestCollector::new(&config);
        assert_eq!(collector.test_id(), "test-1");

        // Record observations
        collector.record("control", 0.7).unwrap();
        collector.record("control", 0.6).unwrap();
        collector.record("treatment", 0.9).unwrap();
        collector.record("treatment", 0.85).unwrap();

        let control_metrics = collector.get_metrics("control").unwrap();
        assert_eq!(control_metrics.count, 2);
        assert!((control_metrics.mean() - 0.65).abs() < 0.001);

        let treatment_metrics = collector.get_metrics("treatment").unwrap();
        assert_eq!(treatment_metrics.count, 2);
        assert!((treatment_metrics.mean() - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_ab_test_collector_invalid_variant() {
        let control = TestVariant::new("a", "A", 0.5);
        let treatment = TestVariant::new("b", "B", 0.5);
        let config = AbTestConfig::new("t", "Test", control, treatment);

        let mut collector = AbTestCollector::new(&config);
        let result = collector.record("nonexistent", 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ab_test_collector_significance() {
        let control = TestVariant::new("control", "Control", 0.5);
        let treatment = TestVariant::new("treatment", "Treatment", 0.5);
        let config = AbTestConfig::new("t", "Test", control, treatment);

        let mut collector = AbTestCollector::new(&config);

        // Add enough data
        for _ in 0..50 {
            collector.record("control", 0.5).unwrap();
            collector.record("treatment", 0.8).unwrap();
        }

        let sig = collector.significance("control", "treatment").unwrap();
        assert!(sig.is_some());
    }

    #[test]
    fn test_normal_cdf_values() {
        // Standard normal CDF at 0 should be ~0.5
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        // CDF at 2 should be ~0.977
        let cdf_2 = normal_cdf(2.0);
        assert!((cdf_2 - 0.977).abs() < 0.01);

        // CDF at -2 should be ~0.023
        let cdf_neg2 = normal_cdf(-2.0);
        assert!((cdf_neg2 - 0.023).abs() < 0.01);
    }

    #[test]
    fn test_ab_test_error_display() {
        let err = AbTestError::InvalidConfig("bad config".to_string());
        assert!(err.to_string().contains("bad config"));

        let err = AbTestError::NotFound("test-1".to_string());
        assert!(err.to_string().contains("test-1"));

        let err = AbTestError::VariantNotFound("variant-x".to_string());
        assert!(err.to_string().contains("variant-x"));
    }

    #[test]
    fn test_simple_hash_deterministic() {
        let h1 = simple_hash("user-123");
        let h2 = simple_hash("user-123");
        assert_eq!(h1, h2);

        let h3 = simple_hash("user-456");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_multi_variant_test() {
        let config = AbTestConfig::with_variants(
            "multi",
            "Multi-variant Test",
            vec![
                TestVariant::new("a", "Variant A", 0.33),
                TestVariant::new("b", "Variant B", 0.33),
                TestVariant::new("c", "Variant C", 0.34),
            ],
        )
        .with_min_sample_size(50);

        assert!(config.validate().is_ok());
        assert_eq!(config.variants.len(), 3);
        assert_eq!(config.min_sample_size, 50);
    }
}
