// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Regression Testing (REQ-10.3)
//!
//! Provides automated regression testing for agent behavior, detecting drift
//! in output quality over time. Includes golden datasets with expected outputs,
//! semantic similarity scoring for regression detection, and CI integration with
//! pass/fail thresholds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during regression testing.
#[derive(Debug, Error)]
pub enum RegressionError {
    /// Golden dataset is invalid or empty.
    #[error("Invalid golden dataset: {0}")]
    InvalidDataset(String),

    /// Threshold violation.
    #[error("Regression detected: {0}")]
    RegressionDetected(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    Execution(String),
}

/// A single golden test case with expected output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCase {
    /// Unique identifier.
    pub id: String,
    /// Input prompt.
    pub input: String,
    /// Expected (golden) output.
    pub expected_output: String,
    /// Minimum acceptable similarity score (0.0 to 1.0).
    pub min_score: f64,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl GoldenCase {
    /// Create a new golden case.
    pub fn new(
        id: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected_output: expected.into(),
            min_score: 0.8,
            tags: vec![],
        }
    }

    /// Set the minimum acceptable score.
    pub fn with_min_score(mut self, score: f64) -> Self {
        self.min_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// A golden dataset for regression testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenDataset {
    /// Dataset name.
    pub name: String,
    /// Version identifier.
    pub version: String,
    /// The golden test cases.
    pub cases: Vec<GoldenCase>,
    /// Global minimum score threshold.
    pub threshold: f64,
}

impl GoldenDataset {
    /// Create a new golden dataset.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            cases: vec![],
            threshold: 0.8,
        }
    }

    /// Set the global threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add cases.
    pub fn with_cases(mut self, cases: Vec<GoldenCase>) -> Self {
        self.cases = cases;
        self
    }

    /// Add a single case.
    pub fn add_case(&mut self, case: GoldenCase) {
        self.cases.push(case);
    }

    /// Get the number of cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

/// Similarity scoring strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityStrategy {
    /// Exact string match (score is 0 or 1).
    ExactMatch,
    /// Word-level Jaccard similarity.
    Jaccard,
    /// Character-level Levenshtein distance normalized to 0-1.
    Levenshtein,
    /// Cosine similarity of word frequency vectors.
    CosineSimilarity,
}

/// Compute similarity between actual and expected output.
pub fn compute_similarity(actual: &str, expected: &str, strategy: &SimilarityStrategy) -> f64 {
    match strategy {
        SimilarityStrategy::ExactMatch => {
            if actual.trim() == expected.trim() {
                1.0
            } else {
                0.0
            }
        }
        SimilarityStrategy::Jaccard => jaccard_similarity(actual, expected),
        SimilarityStrategy::Levenshtein => levenshtein_similarity(actual, expected),
        SimilarityStrategy::CosineSimilarity => cosine_similarity(actual, expected),
    }
}

/// Jaccard similarity based on word overlap.
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let union = words_a.union(&words_b).count() as f64;
    intersection / union
}

/// Levenshtein distance normalized to a similarity score (1.0 = identical).
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len());

    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(&a_chars, &b_chars);
    1.0 - (distance as f64 / max_len as f64)
}

/// Compute Levenshtein edit distance.
fn levenshtein_distance(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Cosine similarity based on word frequency vectors.
fn cosine_similarity(a: &str, b: &str) -> f64 {
    let freq_a = word_frequency(a);
    let freq_b = word_frequency(b);

    if freq_a.is_empty() && freq_b.is_empty() {
        return 1.0;
    }
    if freq_a.is_empty() || freq_b.is_empty() {
        return 0.0;
    }

    // Compute dot product and magnitudes
    let mut dot = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;

    let all_words: std::collections::HashSet<&str> =
        freq_a.keys().chain(freq_b.keys()).copied().collect();

    for word in all_words {
        let va = *freq_a.get(word).unwrap_or(&0) as f64;
        let vb = *freq_b.get(word).unwrap_or(&0) as f64;
        dot += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    let denominator = mag_a.sqrt() * mag_b.sqrt();
    if denominator == 0.0 {
        return 0.0;
    }

    dot / denominator
}

/// Count word frequencies.
fn word_frequency(text: &str) -> HashMap<&str, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        *freq.entry(word).or_insert(0) += 1;
    }
    freq
}

/// Result of testing a single golden case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseTestResult {
    /// Case ID.
    pub case_id: String,
    /// Actual output produced.
    pub actual_output: String,
    /// Similarity score.
    pub score: f64,
    /// Whether it passes the threshold.
    pub passed: bool,
    /// The threshold used.
    pub threshold: f64,
}

/// Overall regression test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult {
    /// Dataset name.
    pub dataset_name: String,
    /// Dataset version.
    pub dataset_version: String,
    /// Individual case results.
    pub case_results: Vec<CaseTestResult>,
    /// Total cases tested.
    pub total_cases: usize,
    /// Cases that passed.
    pub passed_cases: usize,
    /// Cases that failed.
    pub failed_cases: usize,
    /// Average similarity score.
    pub avg_score: f64,
    /// Whether the overall test suite passed.
    pub overall_passed: bool,
    /// Strategy used for scoring.
    pub strategy: SimilarityStrategy,
}

/// Configuration for regression test execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Similarity strategy to use.
    pub strategy: SimilarityStrategy,
    /// Whether all cases must pass (strict) or just the average threshold (lenient).
    pub strict_mode: bool,
    /// Minimum overall average score required.
    pub min_avg_score: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            strategy: SimilarityStrategy::Jaccard,
            strict_mode: false,
            min_avg_score: 0.8,
        }
    }
}

impl RegressionConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the similarity strategy.
    pub fn with_strategy(mut self, strategy: SimilarityStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable strict mode (all cases must pass individually).
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Set minimum average score.
    pub fn with_min_avg_score(mut self, score: f64) -> Self {
        self.min_avg_score = score.clamp(0.0, 1.0);
        self
    }
}

/// Run regression tests against a golden dataset.
pub struct RegressionRunner {
    config: RegressionConfig,
}

impl RegressionRunner {
    /// Create a new runner.
    pub fn new(config: RegressionConfig) -> Self {
        Self { config }
    }

    /// Get the config.
    pub fn config(&self) -> &RegressionConfig {
        &self.config
    }

    /// Run the regression test with a provided evaluation function.
    ///
    /// The `eval_fn` takes an input string and returns the actual output.
    pub fn run<F>(
        &self,
        dataset: &GoldenDataset,
        eval_fn: F,
    ) -> Result<RegressionTestResult, RegressionError>
    where
        F: Fn(&str) -> Result<String, String>,
    {
        if dataset.is_empty() {
            return Err(RegressionError::InvalidDataset(
                "empty golden dataset".to_string(),
            ));
        }

        let mut case_results = Vec::with_capacity(dataset.len());

        for case in &dataset.cases {
            let actual_output = eval_fn(&case.input).map_err(RegressionError::Execution)?;

            let score =
                compute_similarity(&actual_output, &case.expected_output, &self.config.strategy);

            let threshold = case.min_score.max(dataset.threshold);
            let passed = score >= threshold;

            case_results.push(CaseTestResult {
                case_id: case.id.clone(),
                actual_output,
                score,
                passed,
                threshold,
            });
        }

        let total_cases = case_results.len();
        let passed_cases = case_results.iter().filter(|r| r.passed).count();
        let failed_cases = total_cases - passed_cases;
        let avg_score = case_results.iter().map(|r| r.score).sum::<f64>() / total_cases as f64;

        let overall_passed = if self.config.strict_mode {
            failed_cases == 0
        } else {
            avg_score >= self.config.min_avg_score
        };

        Ok(RegressionTestResult {
            dataset_name: dataset.name.clone(),
            dataset_version: dataset.version.clone(),
            case_results,
            total_cases,
            passed_cases,
            failed_cases,
            avg_score,
            overall_passed,
            strategy: self.config.strategy.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-10.3: Regression Testing tests

    #[test]
    fn test_golden_case_creation() {
        let case = GoldenCase::new("case-1", "What is AI?", "Artificial Intelligence")
            .with_min_score(0.9)
            .with_tags(vec!["definition".to_string()]);

        assert_eq!(case.id, "case-1");
        assert_eq!(case.input, "What is AI?");
        assert_eq!(case.expected_output, "Artificial Intelligence");
        assert_eq!(case.min_score, 0.9);
        assert_eq!(case.tags.len(), 1);
    }

    #[test]
    fn test_golden_case_score_clamped() {
        let case = GoldenCase::new("1", "a", "b").with_min_score(1.5);
        assert_eq!(case.min_score, 1.0);

        let case2 = GoldenCase::new("2", "a", "b").with_min_score(-0.5);
        assert_eq!(case2.min_score, 0.0);
    }

    #[test]
    fn test_golden_dataset_creation() {
        let mut dataset = GoldenDataset::new("qa-test", "v1.0").with_threshold(0.85);

        assert_eq!(dataset.name, "qa-test");
        assert_eq!(dataset.version, "v1.0");
        assert_eq!(dataset.threshold, 0.85);
        assert!(dataset.is_empty());

        dataset.add_case(GoldenCase::new("1", "hello", "world"));
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_golden_dataset_with_cases() {
        let dataset = GoldenDataset::new("test", "v1").with_cases(vec![
            GoldenCase::new("1", "a", "b"),
            GoldenCase::new("2", "c", "d"),
        ]);
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_exact_match_similarity() {
        assert_eq!(
            compute_similarity(
                "hello world",
                "hello world",
                &SimilarityStrategy::ExactMatch
            ),
            1.0
        );
        assert_eq!(
            compute_similarity("hello", "world", &SimilarityStrategy::ExactMatch),
            0.0
        );
        // Trims whitespace
        assert_eq!(
            compute_similarity("  hello  ", "hello", &SimilarityStrategy::ExactMatch),
            1.0
        );
    }

    #[test]
    fn test_jaccard_similarity() {
        // Same words
        assert_eq!(
            compute_similarity("hello world", "hello world", &SimilarityStrategy::Jaccard),
            1.0
        );
        // No overlap
        assert_eq!(
            compute_similarity("abc def", "xyz uvw", &SimilarityStrategy::Jaccard),
            0.0
        );
        // Partial overlap: {hello, world} ∩ {hello, there} = {hello}, union = {hello, world, there}
        let score = compute_similarity("hello world", "hello there", &SimilarityStrategy::Jaccard);
        assert!((score - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_similarity_empty() {
        assert_eq!(
            compute_similarity("", "", &SimilarityStrategy::Jaccard),
            1.0
        );
        assert_eq!(
            compute_similarity("hello", "", &SimilarityStrategy::Jaccard),
            0.0
        );
    }

    #[test]
    fn test_levenshtein_similarity() {
        // Identical strings
        assert_eq!(
            compute_similarity("hello", "hello", &SimilarityStrategy::Levenshtein),
            1.0
        );
        // Completely different
        let score = compute_similarity("abc", "xyz", &SimilarityStrategy::Levenshtein);
        assert_eq!(score, 0.0); // distance=3, max_len=3, 1 - 3/3 = 0

        // One character difference: "cat" vs "bat" -> distance=1, max=3 -> 1 - 1/3 ≈ 0.667
        let score = compute_similarity("cat", "bat", &SimilarityStrategy::Levenshtein);
        assert!((score - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_levenshtein_similarity_empty() {
        assert_eq!(
            compute_similarity("", "", &SimilarityStrategy::Levenshtein),
            1.0
        );
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical
        let score = compute_similarity(
            "the cat sat on the mat",
            "the cat sat on the mat",
            &SimilarityStrategy::CosineSimilarity,
        );
        assert!((score - 1.0).abs() < 1e-10);
        // No overlap
        assert_eq!(
            compute_similarity("abc def", "xyz uvw", &SimilarityStrategy::CosineSimilarity),
            0.0
        );
        // Partial overlap
        let score = compute_similarity(
            "hello world foo",
            "hello world bar",
            &SimilarityStrategy::CosineSimilarity,
        );
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(
            compute_similarity("", "", &SimilarityStrategy::CosineSimilarity),
            1.0
        );
        assert_eq!(
            compute_similarity("hello", "", &SimilarityStrategy::CosineSimilarity),
            0.0
        );
    }

    #[test]
    fn test_regression_config_default() {
        let config = RegressionConfig::default();
        assert_eq!(config.strategy, SimilarityStrategy::Jaccard);
        assert!(!config.strict_mode);
        assert_eq!(config.min_avg_score, 0.8);
    }

    #[test]
    fn test_regression_config_builder() {
        let config = RegressionConfig::new()
            .with_strategy(SimilarityStrategy::Levenshtein)
            .with_strict_mode()
            .with_min_avg_score(0.9);

        assert_eq!(config.strategy, SimilarityStrategy::Levenshtein);
        assert!(config.strict_mode);
        assert_eq!(config.min_avg_score, 0.9);
    }

    #[test]
    fn test_regression_runner_all_pass() {
        let dataset = GoldenDataset::new("test", "v1")
            .with_threshold(0.8)
            .with_cases(vec![
                GoldenCase::new("1", "hello", "hello world"),
                GoldenCase::new("2", "foo", "foo bar"),
            ]);

        let runner = RegressionRunner::new(
            RegressionConfig::new().with_strategy(SimilarityStrategy::Jaccard),
        );

        let result = runner
            .run(&dataset, |input| {
                Ok(match input {
                    "hello" => "hello world".to_string(),
                    "foo" => "foo bar".to_string(),
                    _ => "unknown".to_string(),
                })
            })
            .unwrap();

        assert_eq!(result.total_cases, 2);
        assert_eq!(result.passed_cases, 2);
        assert_eq!(result.failed_cases, 0);
        assert_eq!(result.avg_score, 1.0);
        assert!(result.overall_passed);
    }

    #[test]
    fn test_regression_runner_some_fail() {
        let dataset = GoldenDataset::new("test", "v1")
            .with_threshold(0.8)
            .with_cases(vec![
                GoldenCase::new("1", "hello", "hello world"),
                GoldenCase::new("2", "foo", "completely different expected output"),
            ]);

        let runner = RegressionRunner::new(
            RegressionConfig::new()
                .with_strategy(SimilarityStrategy::Jaccard)
                .with_min_avg_score(0.9),
        );

        let result = runner
            .run(&dataset, |input| {
                Ok(match input {
                    "hello" => "hello world".to_string(),
                    "foo" => "something totally different".to_string(),
                    _ => "unknown".to_string(),
                })
            })
            .unwrap();

        assert_eq!(result.total_cases, 2);
        assert!(result.failed_cases > 0);
        assert!(!result.overall_passed); // avg score below threshold
    }

    #[test]
    fn test_regression_runner_strict_mode() {
        let dataset = GoldenDataset::new("test", "v1")
            .with_threshold(0.9)
            .with_cases(vec![
                GoldenCase::new("1", "a", "perfect match"),
                GoldenCase::new("2", "b", "exact output"),
            ]);

        let runner = RegressionRunner::new(
            RegressionConfig::new()
                .with_strategy(SimilarityStrategy::ExactMatch)
                .with_strict_mode(),
        );

        let result = runner
            .run(&dataset, |input| {
                Ok(match input {
                    "a" => "perfect match".to_string(),
                    "b" => "wrong output".to_string(),
                    _ => "unknown".to_string(),
                })
            })
            .unwrap();

        assert!(!result.overall_passed); // strict mode: one failure = overall fail
        assert_eq!(result.passed_cases, 1);
        assert_eq!(result.failed_cases, 1);
    }

    #[test]
    fn test_regression_runner_empty_dataset() {
        let dataset = GoldenDataset::new("empty", "v1");
        let runner = RegressionRunner::new(RegressionConfig::new());

        let result = runner.run(&dataset, |_| Ok("output".to_string()));
        assert!(result.is_err());
        match result.unwrap_err() {
            RegressionError::InvalidDataset(msg) => assert_eq!(msg, "empty golden dataset"),
            _ => panic!("Expected InvalidDataset error"),
        }
    }

    #[test]
    fn test_regression_runner_eval_error() {
        let dataset = GoldenDataset::new("test", "v1")
            .with_cases(vec![GoldenCase::new("1", "fail", "expected")]);

        let runner = RegressionRunner::new(RegressionConfig::new());

        let result = runner.run(&dataset, |_| Err("eval failed".to_string()));
        assert!(result.is_err());
        match result.unwrap_err() {
            RegressionError::Execution(msg) => assert_eq!(msg, "eval failed"),
            _ => panic!("Expected Execution error"),
        }
    }

    #[test]
    fn test_regression_error_display() {
        let err = RegressionError::InvalidDataset("empty".to_string());
        assert!(err.to_string().contains("empty"));

        let err = RegressionError::RegressionDetected("score dropped".to_string());
        assert!(err.to_string().contains("score dropped"));

        let err = RegressionError::Execution("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_case_test_result_serialization() {
        let result = CaseTestResult {
            case_id: "1".to_string(),
            actual_output: "hello world".to_string(),
            score: 0.95,
            passed: true,
            threshold: 0.8,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: CaseTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.case_id, "1");
        assert_eq!(deserialized.score, 0.95);
        assert!(deserialized.passed);
    }

    #[test]
    fn test_regression_result_serialization() {
        let result = RegressionTestResult {
            dataset_name: "test".to_string(),
            dataset_version: "v1".to_string(),
            case_results: vec![],
            total_cases: 0,
            passed_cases: 0,
            failed_cases: 0,
            avg_score: 0.0,
            overall_passed: true,
            strategy: SimilarityStrategy::Jaccard,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: RegressionTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.dataset_name, "test");
        assert_eq!(deserialized.strategy, SimilarityStrategy::Jaccard);
    }

    #[test]
    fn test_levenshtein_distance_basic() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(levenshtein_distance(&a, &b), 3);
    }

    #[test]
    fn test_levenshtein_distance_empty() {
        let a: Vec<char> = "hello".chars().collect();
        let b: Vec<char> = Vec::new();
        assert_eq!(levenshtein_distance(&a, &b), 5);
        assert_eq!(levenshtein_distance(&b, &a), 5);
        assert_eq!(levenshtein_distance(&b, &b), 0);
    }

    #[test]
    fn test_word_frequency() {
        let freq = word_frequency("the cat sat on the mat");
        assert_eq!(*freq.get("the").unwrap(), 2);
        assert_eq!(*freq.get("cat").unwrap(), 1);
        assert_eq!(*freq.get("sat").unwrap(), 1);
    }

    #[test]
    fn test_regression_with_per_case_threshold() {
        let dataset = GoldenDataset::new("test", "v1")
            .with_threshold(0.5)
            .with_cases(vec![
                GoldenCase::new("1", "easy", "easy answer").with_min_score(0.9),
                GoldenCase::new("2", "hard", "hard answer").with_min_score(0.5),
            ]);

        let runner = RegressionRunner::new(
            RegressionConfig::new()
                .with_strategy(SimilarityStrategy::Jaccard)
                .with_strict_mode(),
        );

        let result = runner
            .run(&dataset, |input| {
                Ok(match input {
                    "easy" => "easy answer".to_string(),
                    "hard" => "somewhat hard answer that is close".to_string(),
                    _ => "unknown".to_string(),
                })
            })
            .unwrap();

        // First case: exact match (score=1.0), threshold=max(0.9, 0.5)=0.9 -> passes
        assert!(result.case_results[0].passed);
        // Second case: partial match, threshold=max(0.5, 0.5)=0.5
        // Jaccard of "somewhat hard answer that is close" vs "hard answer":
        // intersection: {hard, answer} = 2, union: {somewhat, hard, answer, that, is, close} = 6
        // score = 2/6 ≈ 0.33 < 0.5 -> fails
        assert!(!result.case_results[1].passed);
    }
}
