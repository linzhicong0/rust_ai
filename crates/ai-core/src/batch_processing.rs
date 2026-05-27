// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Batch Processing (REQ-14.4)
//!
//! Supports batch processing of requests for high-throughput scenarios with
//! automatic grouping, progress tracking, and partial failure handling.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::batch_processing::{BatchProcessor, BatchRequest, BatchConfig};
//!
//! let config = BatchConfig {
//!     max_batch_size: 10,
//!     max_concurrent: 3,
//!     retry_failed: true,
//!     max_retries: 2,
//! };
//! let mut processor = BatchProcessor::new(config);
//!
//! processor.submit(BatchRequest::new("req-1", "process this"));
//! processor.submit(BatchRequest::new("req-2", "process that"));
//!
//! assert_eq!(processor.pending_count(), 2);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── BatchConfig ───────────────────────────────────────────────────────────────

/// Configuration for batch processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum concurrent batches being processed.
    pub max_concurrent: usize,
    /// Whether to retry failed requests.
    pub retry_failed: bool,
    /// Maximum retry attempts per request.
    pub max_retries: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            max_concurrent: 5,
            retry_failed: true,
            max_retries: 3,
        }
    }
}

// ── BatchRequest ──────────────────────────────────────────────────────────────

/// A single request in a batch.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request identifier.
    pub id: String,
    /// The payload/content of the request.
    pub payload: String,
    /// Optional group key for automatic grouping.
    pub group_key: Option<String>,
    /// Priority (higher = more urgent).
    pub priority: u32,
    /// Number of retry attempts so far.
    pub retries: u32,
}

impl BatchRequest {
    /// Create a new batch request.
    pub fn new(id: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            payload: payload.into(),
            group_key: None,
            priority: 0,
            retries: 0,
        }
    }

    /// Set the group key for automatic grouping.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group_key = Some(group.into());
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

// ── BatchStatus ───────────────────────────────────────────────────────────────

/// Status of a batch request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStatus {
    /// Request is pending/queued.
    Pending,
    /// Request is currently being processed.
    Processing,
    /// Request completed successfully.
    Completed,
    /// Request failed with an error.
    Failed(String),
    /// Request was retried.
    Retrying,
}

// ── BatchResult ───────────────────────────────────────────────────────────────

/// Result of processing a single batch request.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// The request ID.
    pub request_id: String,
    /// Status of the request.
    pub status: BatchStatus,
    /// The output if successful.
    pub output: Option<String>,
    /// Processing duration.
    pub duration: Option<Duration>,
}

// ── BatchProgress ─────────────────────────────────────────────────────────────

/// Progress tracking for batch processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchProgress {
    /// Total requests in the batch.
    pub total: usize,
    /// Completed (success) count.
    pub completed: usize,
    /// Failed count.
    pub failed: usize,
    /// Currently processing count.
    pub processing: usize,
    /// Pending count.
    pub pending: usize,
}

impl BatchProgress {
    /// Check if all requests are done (completed or failed).
    pub fn is_done(&self) -> bool {
        self.pending == 0 && self.processing == 0
    }

    /// Get completion percentage (0.0 to 1.0).
    pub fn completion_ratio(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        (self.completed + self.failed) as f64 / self.total as f64
    }
}

// ── BatchError ────────────────────────────────────────────────────────────────

/// Errors in batch processing.
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    /// Batch size exceeds the configured maximum.
    #[error("Batch size {size} exceeds maximum {max}")]
    BatchTooLarge { size: usize, max: usize },
    /// Request not found.
    #[error("Request not found: {0}")]
    RequestNotFound(String),
    /// Processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

// ── BatchProcessor ────────────────────────────────────────────────────────────

/// Batch processor that manages request submission, grouping, and tracking.
#[derive(Debug)]
pub struct BatchProcessor {
    config: BatchConfig,
    /// Pending requests not yet in a batch.
    pending: Vec<BatchRequest>,
    /// Results indexed by request ID.
    results: HashMap<String, BatchResult>,
    /// Status of each request.
    statuses: HashMap<String, BatchStatus>,
    /// When processing started.
    started_at: Option<Instant>,
}

impl BatchProcessor {
    /// Create a new batch processor with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            results: HashMap::new(),
            statuses: HashMap::new(),
            started_at: None,
        }
    }

    /// Submit a request for batch processing.
    pub fn submit(&mut self, request: BatchRequest) {
        self.statuses
            .insert(request.id.clone(), BatchStatus::Pending);
        self.pending.push(request);
    }

    /// Submit multiple requests at once.
    pub fn submit_many(&mut self, requests: Vec<BatchRequest>) {
        for req in requests {
            self.submit(req);
        }
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get the total number of tracked requests.
    pub fn total_count(&self) -> usize {
        self.statuses.len()
    }

    /// Group pending requests by their group key and return batches.
    pub fn create_batches(&mut self) -> Vec<Vec<BatchRequest>> {
        let mut groups: HashMap<Option<String>, Vec<BatchRequest>> = HashMap::new();

        for req in self.pending.drain(..) {
            groups.entry(req.group_key.clone()).or_default().push(req);
        }

        let max_size = self.config.max_batch_size;
        let mut batches = Vec::new();

        for (_, requests) in groups {
            // Split into chunks of max_batch_size
            for chunk in requests.chunks(max_size) {
                batches.push(chunk.to_vec());
            }
        }

        // Sort batches by priority (highest priority first)
        for batch in &mut batches {
            batch.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        self.started_at = Some(Instant::now());
        batches
    }

    /// Record a successful result for a request.
    pub fn record_success(&mut self, request_id: &str, output: String, duration: Duration) {
        self.statuses
            .insert(request_id.to_string(), BatchStatus::Completed);
        self.results.insert(
            request_id.to_string(),
            BatchResult {
                request_id: request_id.to_string(),
                status: BatchStatus::Completed,
                output: Some(output),
                duration: Some(duration),
            },
        );
    }

    /// Record a failure for a request.
    pub fn record_failure(&mut self, request_id: &str, error: String) {
        self.statuses
            .insert(request_id.to_string(), BatchStatus::Failed(error.clone()));
        self.results.insert(
            request_id.to_string(),
            BatchResult {
                request_id: request_id.to_string(),
                status: BatchStatus::Failed(error),
                output: None,
                duration: None,
            },
        );
    }

    /// Get the result for a specific request.
    pub fn get_result(&self, request_id: &str) -> Option<&BatchResult> {
        self.results.get(request_id)
    }

    /// Get the status of a specific request.
    pub fn get_status(&self, request_id: &str) -> Option<&BatchStatus> {
        self.statuses.get(request_id)
    }

    /// Get current progress.
    pub fn progress(&self) -> BatchProgress {
        let mut completed = 0;
        let mut failed = 0;
        let mut processing = 0;
        let mut pending = 0;

        for status in self.statuses.values() {
            match status {
                BatchStatus::Completed => completed += 1,
                BatchStatus::Failed(_) => failed += 1,
                BatchStatus::Processing => processing += 1,
                BatchStatus::Pending => pending += 1,
                BatchStatus::Retrying => processing += 1,
            }
        }

        BatchProgress {
            total: self.statuses.len(),
            completed,
            failed,
            processing,
            pending,
        }
    }

    /// Get all failed requests that can be retried.
    pub fn get_retriable_failures(&self) -> Vec<&str> {
        if !self.config.retry_failed {
            return Vec::new();
        }

        self.statuses
            .iter()
            .filter(|(_, status)| matches!(status, BatchStatus::Failed(_)))
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get the batch configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-14.4: Batch submission
    #[test]
    fn test_batch_submission() {
        let mut processor = BatchProcessor::new(BatchConfig::default());
        processor.submit(BatchRequest::new("r1", "payload1"));
        processor.submit(BatchRequest::new("r2", "payload2"));

        assert_eq!(processor.pending_count(), 2);
        assert_eq!(processor.total_count(), 2);
    }

    // REQ-14.4: Automatic grouping
    #[test]
    fn test_automatic_grouping() {
        let mut processor = BatchProcessor::new(BatchConfig {
            max_batch_size: 10,
            ..Default::default()
        });

        processor.submit(BatchRequest::new("r1", "p1").with_group("openai"));
        processor.submit(BatchRequest::new("r2", "p2").with_group("openai"));
        processor.submit(BatchRequest::new("r3", "p3").with_group("anthropic"));

        let batches = processor.create_batches();
        // Should have 2 groups
        assert_eq!(batches.len(), 2);
        // Total requests across batches = 3
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 3);
    }

    // REQ-14.4: Batch size limits
    #[test]
    fn test_batch_size_limits() {
        let mut processor = BatchProcessor::new(BatchConfig {
            max_batch_size: 2,
            ..Default::default()
        });

        // Submit 5 requests to same group
        for i in 0..5 {
            processor.submit(BatchRequest::new(format!("r{i}"), format!("p{i}")).with_group("g1"));
        }

        let batches = processor.create_batches();
        // 5 requests with max_batch_size=2 should create 3 batches (2+2+1)
        assert_eq!(batches.len(), 3);
        assert!(batches.iter().all(|b| b.len() <= 2));
    }

    // REQ-14.4: Progress tracking
    #[test]
    fn test_progress_tracking() {
        let mut processor = BatchProcessor::new(BatchConfig::default());
        processor.submit(BatchRequest::new("r1", "p1"));
        processor.submit(BatchRequest::new("r2", "p2"));
        processor.submit(BatchRequest::new("r3", "p3"));

        let progress = processor.progress();
        assert_eq!(progress.total, 3);
        assert_eq!(progress.pending, 3);
        assert_eq!(progress.completed, 0);
        assert!(!progress.is_done());

        // Record results
        processor.record_success("r1", "out1".into(), Duration::from_millis(100));
        processor.record_failure("r2", "timeout".into());

        let progress = processor.progress();
        assert_eq!(progress.completed, 1);
        assert_eq!(progress.failed, 1);
        assert_eq!(progress.pending, 1);
        assert!(!progress.is_done());
    }

    // REQ-14.4: Partial failure handling
    #[test]
    fn test_partial_failure_handling() {
        let mut processor = BatchProcessor::new(BatchConfig {
            retry_failed: true,
            max_retries: 2,
            ..Default::default()
        });

        processor.submit(BatchRequest::new("r1", "p1"));
        processor.submit(BatchRequest::new("r2", "p2"));
        processor.submit(BatchRequest::new("r3", "p3"));

        // Some succeed, some fail
        processor.record_success("r1", "ok".into(), Duration::from_millis(50));
        processor.record_failure("r2", "network error".into());
        processor.record_success("r3", "ok".into(), Duration::from_millis(60));

        let progress = processor.progress();
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.failed, 1);

        // Get retriable failures
        let retriable = processor.get_retriable_failures();
        assert_eq!(retriable.len(), 1);
        assert!(retriable.contains(&"r2"));
    }

    // REQ-14.4: Completion ratio
    #[test]
    fn test_completion_ratio() {
        let mut processor = BatchProcessor::new(BatchConfig::default());
        processor.submit(BatchRequest::new("r1", "p1"));
        processor.submit(BatchRequest::new("r2", "p2"));
        processor.submit(BatchRequest::new("r3", "p3"));
        processor.submit(BatchRequest::new("r4", "p4"));

        processor.record_success("r1", "ok".into(), Duration::from_millis(10));
        processor.record_success("r2", "ok".into(), Duration::from_millis(10));

        let progress = processor.progress();
        assert!((progress.completion_ratio() - 0.5).abs() < f64::EPSILON);
    }

    // REQ-14.4: Submit many
    #[test]
    fn test_submit_many() {
        let mut processor = BatchProcessor::new(BatchConfig::default());
        let requests: Vec<_> = (0..10)
            .map(|i| BatchRequest::new(format!("r{i}"), format!("p{i}")))
            .collect();

        processor.submit_many(requests);
        assert_eq!(processor.pending_count(), 10);
    }

    // REQ-14.4: Get result by ID
    #[test]
    fn test_get_result() {
        let mut processor = BatchProcessor::new(BatchConfig::default());
        processor.submit(BatchRequest::new("r1", "p1"));
        processor.record_success("r1", "output".into(), Duration::from_millis(42));

        let result = processor.get_result("r1").unwrap();
        assert_eq!(result.output, Some("output".to_string()));
        assert_eq!(result.status, BatchStatus::Completed);
    }

    // REQ-14.4: Priority ordering
    #[test]
    fn test_priority_ordering_in_batch() {
        let mut processor = BatchProcessor::new(BatchConfig {
            max_batch_size: 10,
            ..Default::default()
        });

        processor.submit(
            BatchRequest::new("low", "p1")
                .with_group("g")
                .with_priority(1),
        );
        processor.submit(
            BatchRequest::new("high", "p2")
                .with_group("g")
                .with_priority(10),
        );
        processor.submit(
            BatchRequest::new("mid", "p3")
                .with_group("g")
                .with_priority(5),
        );

        let batches = processor.create_batches();
        assert_eq!(batches.len(), 1);
        // Highest priority should be first
        assert_eq!(batches[0][0].id, "high");
        assert_eq!(batches[0][1].id, "mid");
        assert_eq!(batches[0][2].id, "low");
    }

    // REQ-14.4: Empty batch progress
    #[test]
    fn test_empty_progress_is_done() {
        let processor = BatchProcessor::new(BatchConfig::default());
        let progress = processor.progress();
        assert!(progress.is_done());
        assert!((progress.completion_ratio() - 1.0).abs() < f64::EPSILON);
    }

    // REQ-14.4: Retry disabled
    #[test]
    fn test_retry_disabled() {
        let mut processor = BatchProcessor::new(BatchConfig {
            retry_failed: false,
            ..Default::default()
        });

        processor.submit(BatchRequest::new("r1", "p1"));
        processor.record_failure("r1", "error".into());

        let retriable = processor.get_retriable_failures();
        assert!(retriable.is_empty());
    }
}
