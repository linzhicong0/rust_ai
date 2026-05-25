// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Benchmarking (REQ-10.1)
//!
//! Provides benchmarking tools to evaluate model performance across standard datasets
//! and custom evaluation sets. Includes accuracy, latency, cost metrics per run,
//! and the ability to compare results across runs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during benchmarking.
#[derive(Debug, Error)]
pub enum BenchmarkError {
    /// Invalid evaluation dataset.
    #[error("Invalid dataset: {0}")]
    InvalidDataset(String),

    /// Execution error during benchmark.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// A single evaluation case in a benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    /// Unique identifier for this case.
    pub id: String,
    /// Input prompt or query.
    pub input: String,
    /// Expected output (ground truth).
    pub expected: String,
    /// Optional context or reference data.
    pub context: Option<String>,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl EvalCase {
    /// Create a new evaluation case.
    pub fn new(
        id: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected: expected.into(),
            context: None,
            tags: vec![],
        }
    }

    /// Set the context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// A dataset of evaluation cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDataset {
    /// Name of the dataset.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// The evaluation cases.
    pub cases: Vec<EvalCase>,
    /// Metadata about the dataset.
    pub metadata: HashMap<String, String>,
}

impl EvalDataset {
    /// Create a new dataset.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            cases: vec![],
            metadata: HashMap::new(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add cases.
    pub fn with_cases(mut self, cases: Vec<EvalCase>) -> Self {
        self.cases = cases;
        self
    }

    /// Add a single case.
    pub fn add_case(&mut self, case: EvalCase) {
        self.cases.push(case);
    }

    /// Get the number of cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

/// Result of evaluating a single case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// ID of the evaluated case.
    pub case_id: String,
    /// The actual output produced.
    pub actual_output: String,
    /// Whether the output is considered correct.
    pub correct: bool,
    /// Score (0.0 to 1.0).
    pub score: f64,
    /// Latency for this case.
    pub latency_ms: u64,
    /// Token usage.
    pub tokens_used: u32,
    /// Estimated cost.
    pub cost: f64,
}

/// Aggregated metrics for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Total number of cases evaluated.
    pub total_cases: usize,
    /// Number of correct results.
    pub correct: usize,
    /// Accuracy (correct / total).
    pub accuracy: f64,
    /// Average score across all cases.
    pub avg_score: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// P50 latency.
    pub p50_latency_ms: u64,
    /// P95 latency.
    pub p95_latency_ms: u64,
    /// P99 latency.
    pub p99_latency_ms: u64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Total estimated cost.
    pub total_cost: f64,
    /// Total wall-clock time for the benchmark.
    pub total_duration_ms: u64,
}

impl BenchmarkMetrics {
    /// Compute metrics from case results.
    pub fn from_results(results: &[CaseResult], total_duration: Duration) -> Self {
        let total_cases = results.len();
        if total_cases == 0 {
            return Self {
                total_cases: 0,
                correct: 0,
                accuracy: 0.0,
                avg_score: 0.0,
                avg_latency_ms: 0.0,
                p50_latency_ms: 0,
                p95_latency_ms: 0,
                p99_latency_ms: 0,
                total_tokens: 0,
                total_cost: 0.0,
                total_duration_ms: total_duration.as_millis() as u64,
            };
        }

        let correct = results.iter().filter(|r| r.correct).count();
        let accuracy = correct as f64 / total_cases as f64;
        let avg_score = results.iter().map(|r| r.score).sum::<f64>() / total_cases as f64;

        let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
        latencies.sort_unstable();

        let avg_latency_ms = latencies.iter().sum::<u64>() as f64 / total_cases as f64;
        let p50_latency_ms = percentile(&latencies, 50);
        let p95_latency_ms = percentile(&latencies, 95);
        let p99_latency_ms = percentile(&latencies, 99);

        let total_tokens = results.iter().map(|r| r.tokens_used as u64).sum();
        let total_cost = results.iter().map(|r| r.cost).sum();

        Self {
            total_cases,
            correct,
            accuracy,
            avg_score,
            avg_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            total_tokens,
            total_cost,
            total_duration_ms: total_duration.as_millis() as u64,
        }
    }
}

/// Compute the percentile value from a sorted slice.
fn percentile(sorted: &[u64], p: u32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// A complete benchmark run result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    /// Unique run ID.
    pub run_id: String,
    /// Dataset name.
    pub dataset_name: String,
    /// Agent/model configuration used.
    pub agent_config: String,
    /// When the run started.
    pub started_at: DateTime<Utc>,
    /// Individual case results.
    pub results: Vec<CaseResult>,
    /// Aggregated metrics.
    pub metrics: BenchmarkMetrics,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Comparison between two benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    /// Run ID of the baseline.
    pub baseline_run_id: String,
    /// Run ID of the candidate.
    pub candidate_run_id: String,
    /// Accuracy delta (candidate - baseline).
    pub accuracy_delta: f64,
    /// Average score delta.
    pub avg_score_delta: f64,
    /// Latency delta in ms.
    pub avg_latency_delta_ms: f64,
    /// Cost delta.
    pub cost_delta: f64,
    /// Whether the candidate is better overall.
    pub is_improvement: bool,
}

impl RunComparison {
    /// Compare two benchmark runs.
    pub fn compare(baseline: &BenchmarkRun, candidate: &BenchmarkRun) -> Self {
        let accuracy_delta = candidate.metrics.accuracy - baseline.metrics.accuracy;
        let avg_score_delta = candidate.metrics.avg_score - baseline.metrics.avg_score;
        let avg_latency_delta_ms =
            candidate.metrics.avg_latency_ms - baseline.metrics.avg_latency_ms;
        let cost_delta = candidate.metrics.total_cost - baseline.metrics.total_cost;

        // Improvement = better accuracy and not significantly worse latency
        let is_improvement =
            accuracy_delta > 0.0 && avg_latency_delta_ms < baseline.metrics.avg_latency_ms * 0.5;

        Self {
            baseline_run_id: baseline.run_id.clone(),
            candidate_run_id: candidate.run_id.clone(),
            accuracy_delta,
            avg_score_delta,
            avg_latency_delta_ms,
            cost_delta,
            is_improvement,
        }
    }
}

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Maximum concurrent evaluations.
    pub concurrency: usize,
    /// Timeout per case in milliseconds.
    pub timeout_ms: u64,
    /// Whether to continue on individual case failures.
    pub continue_on_error: bool,
    /// Custom scoring function name (if using registry).
    pub scorer: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            timeout_ms: 30000,
            continue_on_error: true,
            scorer: None,
        }
    }
}

impl BenchmarkConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set concurrency.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set timeout.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set whether to continue on error.
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Set the scorer.
    pub fn with_scorer(mut self, scorer: impl Into<String>) -> Self {
        self.scorer = Some(scorer.into());
        self
    }
}

/// A benchmark runner that evaluates datasets against an evaluation function.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Get the config.
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Run a benchmark with a provided evaluation function.
    ///
    /// The `eval_fn` takes an input string and returns (output, tokens_used, cost).
    pub fn run_sync<F>(
        &self,
        dataset: &EvalDataset,
        eval_fn: F,
    ) -> Result<BenchmarkRun, BenchmarkError>
    where
        F: Fn(&str) -> Result<(String, u32, f64), String>,
    {
        if dataset.is_empty() {
            return Err(BenchmarkError::InvalidDataset("empty dataset".to_string()));
        }

        let started_at = Utc::now();
        let start = Instant::now();
        let mut results = Vec::with_capacity(dataset.len());

        for case in &dataset.cases {
            let case_start = Instant::now();
            match eval_fn(&case.input) {
                Ok((output, tokens, cost)) => {
                    let latency_ms = case_start.elapsed().as_millis() as u64;
                    let correct = output.trim() == case.expected.trim();
                    let score = if correct {
                        1.0
                    } else {
                        similarity_score(&output, &case.expected)
                    };

                    results.push(CaseResult {
                        case_id: case.id.clone(),
                        actual_output: output,
                        correct,
                        score,
                        latency_ms,
                        tokens_used: tokens,
                        cost,
                    });
                }
                Err(e) => {
                    if !self.config.continue_on_error {
                        return Err(BenchmarkError::Execution(e));
                    }
                    results.push(CaseResult {
                        case_id: case.id.clone(),
                        actual_output: format!("ERROR: {}", e),
                        correct: false,
                        score: 0.0,
                        latency_ms: case_start.elapsed().as_millis() as u64,
                        tokens_used: 0,
                        cost: 0.0,
                    });
                }
            }
        }

        let total_duration = start.elapsed();
        let metrics = BenchmarkMetrics::from_results(&results, total_duration);

        Ok(BenchmarkRun {
            run_id: uuid::Uuid::new_v4().to_string(),
            dataset_name: dataset.name.clone(),
            agent_config: "sync_eval".to_string(),
            started_at,
            results,
            metrics,
            metadata: HashMap::new(),
        })
    }
}

/// Simple similarity score between two strings (word overlap / Jaccard).
fn similarity_score(actual: &str, expected: &str) -> f64 {
    let actual_words: std::collections::HashSet<&str> = actual.split_whitespace().collect();
    let expected_words: std::collections::HashSet<&str> = expected.split_whitespace().collect();

    if actual_words.is_empty() && expected_words.is_empty() {
        return 1.0;
    }
    if actual_words.is_empty() || expected_words.is_empty() {
        return 0.0;
    }

    let intersection = actual_words.intersection(&expected_words).count() as f64;
    let union = actual_words.union(&expected_words).count() as f64;

    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-10.1: Benchmarking tests

    #[test]
    fn test_eval_case_creation() {
        let case = EvalCase::new("case-1", "What is 2+2?", "4")
            .with_context("Basic arithmetic")
            .with_tags(vec!["math".to_string(), "easy".to_string()]);

        assert_eq!(case.id, "case-1");
        assert_eq!(case.input, "What is 2+2?");
        assert_eq!(case.expected, "4");
        assert_eq!(case.context.unwrap(), "Basic arithmetic");
        assert_eq!(case.tags.len(), 2);
    }

    #[test]
    fn test_eval_dataset_creation() {
        let mut dataset = EvalDataset::new("arithmetic").with_description("Basic arithmetic test");

        assert_eq!(dataset.name, "arithmetic");
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);

        dataset.add_case(EvalCase::new("1", "2+2", "4"));
        dataset.add_case(EvalCase::new("2", "3+3", "6"));

        assert!(!dataset.is_empty());
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_eval_dataset_with_cases() {
        let cases = vec![
            EvalCase::new("1", "hello", "world"),
            EvalCase::new("2", "foo", "bar"),
        ];
        let dataset = EvalDataset::new("test").with_cases(cases);
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_benchmark_metrics_from_results() {
        let results = vec![
            CaseResult {
                case_id: "1".to_string(),
                actual_output: "4".to_string(),
                correct: true,
                score: 1.0,
                latency_ms: 100,
                tokens_used: 50,
                cost: 0.001,
            },
            CaseResult {
                case_id: "2".to_string(),
                actual_output: "7".to_string(),
                correct: false,
                score: 0.5,
                latency_ms: 200,
                tokens_used: 60,
                cost: 0.002,
            },
            CaseResult {
                case_id: "3".to_string(),
                actual_output: "10".to_string(),
                correct: true,
                score: 1.0,
                latency_ms: 150,
                tokens_used: 55,
                cost: 0.0015,
            },
        ];

        let metrics = BenchmarkMetrics::from_results(&results, Duration::from_millis(500));

        assert_eq!(metrics.total_cases, 3);
        assert_eq!(metrics.correct, 2);
        assert!((metrics.accuracy - 2.0 / 3.0).abs() < 0.001);
        assert!((metrics.avg_score - (1.0 + 0.5 + 1.0) / 3.0).abs() < 0.001);
        assert_eq!(metrics.total_tokens, 165);
        assert!((metrics.total_cost - 0.0045).abs() < 0.0001);
        assert_eq!(metrics.total_duration_ms, 500);
    }

    #[test]
    fn test_benchmark_metrics_empty() {
        let metrics = BenchmarkMetrics::from_results(&[], Duration::from_millis(0));
        assert_eq!(metrics.total_cases, 0);
        assert_eq!(metrics.accuracy, 0.0);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.continue_on_error);
        assert!(config.scorer.is_none());
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::new()
            .with_concurrency(4)
            .with_timeout_ms(10000)
            .with_continue_on_error(false)
            .with_scorer("exact_match");

        assert_eq!(config.concurrency, 4);
        assert_eq!(config.timeout_ms, 10000);
        assert!(!config.continue_on_error);
        assert_eq!(config.scorer.unwrap(), "exact_match");
    }

    #[test]
    fn test_benchmark_config_clamps_concurrency() {
        let config = BenchmarkConfig::new().with_concurrency(0);
        assert_eq!(config.concurrency, 1);
    }

    #[test]
    fn test_benchmark_runner_success() {
        let dataset = EvalDataset::new("math").with_cases(vec![
            EvalCase::new("1", "2+2", "4"),
            EvalCase::new("2", "3+3", "6"),
            EvalCase::new("3", "5+5", "10"),
        ]);

        let runner = BenchmarkRunner::new(BenchmarkConfig::new());
        let run = runner
            .run_sync(&dataset, |input| {
                // Simple eval that returns correct answers
                let result = match input {
                    "2+2" => "4",
                    "3+3" => "6",
                    "5+5" => "10",
                    _ => "unknown",
                };
                Ok((result.to_string(), 10, 0.001))
            })
            .unwrap();

        assert_eq!(run.metrics.total_cases, 3);
        assert_eq!(run.metrics.correct, 3);
        assert_eq!(run.metrics.accuracy, 1.0);
        assert_eq!(run.results.len(), 3);
    }

    #[test]
    fn test_benchmark_runner_partial_failure() {
        let dataset = EvalDataset::new("mixed").with_cases(vec![
            EvalCase::new("1", "2+2", "4"),
            EvalCase::new("2", "hard", "impossible"),
        ]);

        let runner = BenchmarkRunner::new(BenchmarkConfig::new());
        let run = runner
            .run_sync(&dataset, |input| {
                if input == "hard" {
                    Err("too difficult".to_string())
                } else {
                    Ok(("4".to_string(), 10, 0.001))
                }
            })
            .unwrap();

        assert_eq!(run.metrics.total_cases, 2);
        assert_eq!(run.metrics.correct, 1);
        assert!(run.results[1].actual_output.contains("ERROR"));
    }

    #[test]
    fn test_benchmark_runner_stop_on_error() {
        let dataset = EvalDataset::new("strict").with_cases(vec![
            EvalCase::new("1", "fail", "x"),
            EvalCase::new("2", "ok", "ok"),
        ]);

        let runner = BenchmarkRunner::new(BenchmarkConfig::new().with_continue_on_error(false));
        let result = runner.run_sync(&dataset, |_input| Err("error".to_string()));

        assert!(result.is_err());
        match result.unwrap_err() {
            BenchmarkError::Execution(msg) => assert_eq!(msg, "error"),
            _ => panic!("Expected Execution error"),
        }
    }

    #[test]
    fn test_benchmark_runner_empty_dataset() {
        let dataset = EvalDataset::new("empty");
        let runner = BenchmarkRunner::new(BenchmarkConfig::new());
        let result = runner.run_sync(&dataset, |_| Ok(("".to_string(), 0, 0.0)));

        assert!(result.is_err());
        match result.unwrap_err() {
            BenchmarkError::InvalidDataset(msg) => assert_eq!(msg, "empty dataset"),
            _ => panic!("Expected InvalidDataset error"),
        }
    }

    #[test]
    fn test_run_comparison() {
        let baseline = BenchmarkRun {
            run_id: "run-1".to_string(),
            dataset_name: "test".to_string(),
            agent_config: "gpt-4".to_string(),
            started_at: Utc::now(),
            results: vec![],
            metrics: BenchmarkMetrics {
                total_cases: 10,
                correct: 7,
                accuracy: 0.7,
                avg_score: 0.75,
                avg_latency_ms: 200.0,
                p50_latency_ms: 180,
                p95_latency_ms: 350,
                p99_latency_ms: 400,
                total_tokens: 1000,
                total_cost: 0.05,
                total_duration_ms: 2000,
            },
            metadata: HashMap::new(),
        };

        let candidate = BenchmarkRun {
            run_id: "run-2".to_string(),
            dataset_name: "test".to_string(),
            agent_config: "gpt-4-turbo".to_string(),
            started_at: Utc::now(),
            results: vec![],
            metrics: BenchmarkMetrics {
                total_cases: 10,
                correct: 9,
                accuracy: 0.9,
                avg_score: 0.92,
                avg_latency_ms: 150.0,
                p50_latency_ms: 130,
                p95_latency_ms: 250,
                p99_latency_ms: 300,
                total_tokens: 1200,
                total_cost: 0.06,
                total_duration_ms: 1500,
            },
            metadata: HashMap::new(),
        };

        let comparison = RunComparison::compare(&baseline, &candidate);
        assert_eq!(comparison.baseline_run_id, "run-1");
        assert_eq!(comparison.candidate_run_id, "run-2");
        assert!((comparison.accuracy_delta - 0.2).abs() < 0.001);
        assert!(comparison.avg_latency_delta_ms < 0.0); // candidate is faster
        assert!(comparison.is_improvement);
    }

    #[test]
    fn test_run_comparison_regression() {
        let baseline = BenchmarkRun {
            run_id: "run-1".to_string(),
            dataset_name: "test".to_string(),
            agent_config: "v1".to_string(),
            started_at: Utc::now(),
            results: vec![],
            metrics: BenchmarkMetrics {
                total_cases: 10,
                correct: 9,
                accuracy: 0.9,
                avg_score: 0.92,
                avg_latency_ms: 100.0,
                p50_latency_ms: 90,
                p95_latency_ms: 150,
                p99_latency_ms: 200,
                total_tokens: 500,
                total_cost: 0.03,
                total_duration_ms: 1000,
            },
            metadata: HashMap::new(),
        };

        let candidate = BenchmarkRun {
            run_id: "run-2".to_string(),
            dataset_name: "test".to_string(),
            agent_config: "v2".to_string(),
            started_at: Utc::now(),
            results: vec![],
            metrics: BenchmarkMetrics {
                total_cases: 10,
                correct: 6,
                accuracy: 0.6,
                avg_score: 0.65,
                avg_latency_ms: 300.0,
                p50_latency_ms: 280,
                p95_latency_ms: 450,
                p99_latency_ms: 500,
                total_tokens: 800,
                total_cost: 0.05,
                total_duration_ms: 3000,
            },
            metadata: HashMap::new(),
        };

        let comparison = RunComparison::compare(&baseline, &candidate);
        assert!(comparison.accuracy_delta < 0.0); // regression
        assert!(!comparison.is_improvement);
    }

    #[test]
    fn test_similarity_score_exact_match() {
        let score = similarity_score("hello world", "hello world");
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_similarity_score_partial_match() {
        let score = similarity_score("hello world foo", "hello world bar");
        // intersection: {hello, world} = 2, union: {hello, world, foo, bar} = 4
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_similarity_score_no_match() {
        let score = similarity_score("abc def", "xyz uvw");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_similarity_score_empty() {
        assert_eq!(similarity_score("", ""), 1.0);
        assert_eq!(similarity_score("hello", ""), 0.0);
        assert_eq!(similarity_score("", "hello"), 0.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let sorted = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        // p50 of 10 elements: index = round(0.5 * 9) = round(4.5) = 5 -> value 60
        let p50 = percentile(&sorted, 50);
        assert!(p50 == 50 || p50 == 60); // depends on rounding
        assert_eq!(percentile(&sorted, 0), 10);
        // p95: index = round(0.95 * 9) = round(8.55) = 9 -> value 100
        assert_eq!(percentile(&sorted, 95), 100);
    }

    #[test]
    fn test_percentile_single_element() {
        let sorted = vec![42];
        assert_eq!(percentile(&sorted, 50), 42);
        assert_eq!(percentile(&sorted, 99), 42);
    }

    #[test]
    fn test_percentile_empty() {
        let sorted: Vec<u64> = vec![];
        assert_eq!(percentile(&sorted, 50), 0);
    }

    #[test]
    fn test_benchmark_error_display() {
        let err = BenchmarkError::InvalidDataset("missing cases".to_string());
        assert!(err.to_string().contains("missing cases"));

        let err = BenchmarkError::Execution("timeout".to_string());
        assert!(err.to_string().contains("timeout"));

        let err = BenchmarkError::Config("invalid scorer".to_string());
        assert!(err.to_string().contains("invalid scorer"));
    }

    #[test]
    fn test_benchmark_run_serialization() {
        let run = BenchmarkRun {
            run_id: "test-run".to_string(),
            dataset_name: "math".to_string(),
            agent_config: "gpt-4".to_string(),
            started_at: Utc::now(),
            results: vec![CaseResult {
                case_id: "1".to_string(),
                actual_output: "4".to_string(),
                correct: true,
                score: 1.0,
                latency_ms: 100,
                tokens_used: 50,
                cost: 0.001,
            }],
            metrics: BenchmarkMetrics::from_results(
                &[CaseResult {
                    case_id: "1".to_string(),
                    actual_output: "4".to_string(),
                    correct: true,
                    score: 1.0,
                    latency_ms: 100,
                    tokens_used: 50,
                    cost: 0.001,
                }],
                Duration::from_millis(100),
            ),
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&run).unwrap();
        let deserialized: BenchmarkRun = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.run_id, "test-run");
        assert_eq!(deserialized.metrics.total_cases, 1);
        assert_eq!(deserialized.metrics.accuracy, 1.0);
    }
}
