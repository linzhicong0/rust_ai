//! Prometheus-compatible metrics for the AI framework.
//!
//! Provides in-memory counters and histograms that can be rendered as
//! Prometheus text-format exposition data.
//!
//! ## Metrics exposed
//!
//! | Name | Type | Labels | Description |
//! |------|------|--------|-------------|
//! | `ai_request_latency_seconds` | Histogram | provider, model, agent | LLM request latency |
//! | `ai_tokens_total` | Counter | provider, model, agent, direction | Token throughput |
//! | `ai_errors_total` | Counter | provider, model, agent, kind | Request errors |
//! | `ai_cache_hits_total` | Counter | cache_type | Cache hit events |
//! | `ai_cache_misses_total` | Counter | cache_type | Cache miss events |
//!
//! ## Example
//!
//! ```rust
//! use ai_observability::metrics::{MetricsRegistry, Labels};
//!
//! let registry = MetricsRegistry::new();
//! let labels = Labels::new()
//!     .provider("openai")
//!     .model("gpt-4o-mini")
//!     .agent("my-agent");
//!
//! registry.record_latency(&labels, 0.123);
//! registry.inc_tokens(&labels, "input", 150);
//!
//! let text = registry.render_prometheus();
//! assert!(text.contains("ai_request_latency_seconds"));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Label set ───────────────────────────────────────────────────────────────

/// A set of key-value label pairs attached to a metric observation.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Labels {
    pairs: Vec<(String, String)>,
}

impl Labels {
    /// Create an empty label set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `provider` label.
    pub fn provider(mut self, v: impl Into<String>) -> Self {
        self.set("provider", v);
        self
    }

    /// Set the `model` label.
    pub fn model(mut self, v: impl Into<String>) -> Self {
        self.set("model", v);
        self
    }

    /// Set the `agent` label.
    pub fn agent(mut self, v: impl Into<String>) -> Self {
        self.set("agent", v);
        self
    }

    /// Add an arbitrary label.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, value);
        self
    }

    /// Get the value of a label by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(pair) = self.pairs.iter_mut().find(|(k, _)| k == &key) {
            pair.1 = value;
        } else {
            self.pairs.push((key, value));
        }
    }

    /// Render as a Prometheus label string, e.g. `{provider="openai",model="gpt-4"}`.
    fn render(&self) -> String {
        if self.pairs.is_empty() {
            return String::new();
        }
        let inner: Vec<String> = self
            .pairs
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v.replace('"', "\\\"")))
            .collect();
        format!("{{{}}}", inner.join(","))
    }
}

// ── Counter ──────────────────────────────────────────────────────────────────

/// A monotonically increasing counter metric.
#[derive(Debug, Default)]
struct Counter {
    series: HashMap<Labels, u64>,
}

impl Counter {
    fn inc_by(&mut self, labels: &Labels, delta: u64) {
        *self.series.entry(labels.clone()).or_insert(0) += delta;
    }

    fn get(&self, labels: &Labels) -> u64 {
        self.series.get(labels).copied().unwrap_or(0)
    }

    fn render(&self, name: &str, help: &str) -> String {
        let mut out = format!("# HELP {} {}\n# TYPE {} counter\n", name, help, name);
        let mut sorted: Vec<_> = self.series.iter().collect();
        sorted.sort_by_key(|(l, _)| format!("{:?}", l));
        for (labels, count) in sorted {
            out.push_str(&format!("{}{} {}\n", name, labels.render(), count));
        }
        out
    }
}

// ── Histogram ────────────────────────────────────────────────────────────────

/// Configurable fixed-width histogram buckets for latency in seconds.
const DEFAULT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Per-label-set sample accumulator for a histogram.
#[derive(Debug, Clone)]
struct HistogramData {
    /// Sorted list of all observed values (for percentile calculation).
    samples: Vec<f64>,
    /// Cumulative bucket counts (index matches DEFAULT_BUCKETS).
    buckets: Vec<u64>,
    /// Running sum of all observed values.
    sum: f64,
    /// Total observation count.
    count: u64,
}

impl HistogramData {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            buckets: vec![0; DEFAULT_BUCKETS.len()],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        self.samples.push(value);
        self.sum += value;
        self.count += 1;
        for (i, &upper) in DEFAULT_BUCKETS.iter().enumerate() {
            if value <= upper {
                self.buckets[i] += 1;
            }
        }
    }

    /// Compute an approximate quantile in [0, 1] using sorted samples.
    fn quantile(&self, q: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((q * (sorted.len() as f64)).ceil() as usize).saturating_sub(1);
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    fn render_buckets(&self, metric_name: &str, labels: &Labels) -> String {
        let mut out = String::new();
        let mut cumulative = 0u64;
        for (i, &upper) in DEFAULT_BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i];
            // Merge original labels with the `le` label.
            let mut merged = labels.clone();
            merged.set("le", upper.to_string());
            out.push_str(&format!(
                "{}_bucket{} {}\n",
                metric_name,
                merged.render(),
                cumulative
            ));
        }
        // +Inf bucket
        let mut inf_labels = labels.clone();
        inf_labels.set("le", "+Inf");
        out.push_str(&format!(
            "{}_bucket{} {}\n",
            metric_name,
            inf_labels.render(),
            self.count
        ));
        out.push_str(&format!(
            "{}_sum{} {}\n",
            metric_name,
            labels.render(),
            self.sum
        ));
        out.push_str(&format!(
            "{}_count{} {}\n",
            metric_name,
            labels.render(),
            self.count
        ));
        out
    }
}

/// A histogram metric that tracks latency distributions.
#[derive(Debug, Default)]
struct Histogram {
    series: HashMap<Labels, HistogramData>,
}

impl Histogram {
    fn observe(&mut self, labels: &Labels, value: f64) {
        self.series
            .entry(labels.clone())
            .or_insert_with(HistogramData::new)
            .observe(value);
    }

    fn quantile(&self, labels: &Labels, q: f64) -> Option<f64> {
        self.series.get(labels)?.quantile(q)
    }

    fn count(&self, labels: &Labels) -> u64 {
        self.series.get(labels).map(|d| d.count).unwrap_or(0)
    }

    fn sum(&self, labels: &Labels) -> f64 {
        self.series.get(labels).map(|d| d.sum).unwrap_or(0.0)
    }

    fn render(&self, name: &str, help: &str) -> String {
        let mut out = format!("# HELP {} {}\n# TYPE {} histogram\n", name, help, name);
        let mut sorted: Vec<_> = self.series.iter().collect();
        sorted.sort_by_key(|(l, _)| format!("{:?}", l));
        for (labels, data) in sorted {
            out.push_str(&data.render_buckets(name, labels));
        }
        out
    }
}

// ── Registry ─────────────────────────────────────────────────────────────────

/// Inner mutable state of the registry.
#[derive(Debug, Default)]
struct Inner {
    latency: Histogram,
    tokens: Counter,
    errors: Counter,
    cache_hits: Counter,
    cache_misses: Counter,
}

/// Thread-safe registry for all AI framework metrics.
///
/// Clone the registry cheaply — all clones share the same underlying state.
#[derive(Debug, Clone, Default)]
pub struct MetricsRegistry {
    inner: Arc<Mutex<Inner>>,
}

impl MetricsRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a request latency observation in **seconds**.
    pub fn record_latency(&self, labels: &Labels, seconds: f64) {
        self.inner.lock().unwrap().latency.observe(labels, seconds);
    }

    /// Increment the token counter.
    ///
    /// `direction` should be `"input"` or `"output"`.
    pub fn inc_tokens(&self, labels: &Labels, direction: impl Into<String>, count: u64) {
        let mut key = labels.clone();
        key.set("direction", direction.into());
        self.inner.lock().unwrap().tokens.inc_by(&key, count);
    }

    /// Increment the error counter.
    ///
    /// `kind` identifies the error category, e.g. `"timeout"` or `"rate_limit"`.
    pub fn inc_errors(&self, labels: &Labels, kind: impl Into<String>) {
        let mut key = labels.clone();
        key.set("kind", kind.into());
        self.inner.lock().unwrap().errors.inc_by(&key, 1);
    }

    /// Record a cache hit.
    pub fn inc_cache_hit(&self, cache_type: impl Into<String>) {
        let key = Labels::new().label("cache_type", cache_type.into());
        self.inner.lock().unwrap().cache_hits.inc_by(&key, 1);
    }

    /// Record a cache miss.
    pub fn inc_cache_miss(&self, cache_type: impl Into<String>) {
        let key = Labels::new().label("cache_type", cache_type.into());
        self.inner.lock().unwrap().cache_misses.inc_by(&key, 1);
    }

    // ── Read-back helpers ─────────────────────────────────────────────────

    /// Return the p50 latency for the given labels, or `None` if no observations.
    pub fn latency_p50(&self, labels: &Labels) -> Option<f64> {
        self.inner.lock().unwrap().latency.quantile(labels, 0.50)
    }

    /// Return the p95 latency for the given labels, or `None` if no observations.
    pub fn latency_p95(&self, labels: &Labels) -> Option<f64> {
        self.inner.lock().unwrap().latency.quantile(labels, 0.95)
    }

    /// Return the p99 latency for the given labels, or `None` if no observations.
    pub fn latency_p99(&self, labels: &Labels) -> Option<f64> {
        self.inner.lock().unwrap().latency.quantile(labels, 0.99)
    }

    /// Return the total number of latency observations for the given labels.
    pub fn latency_count(&self, labels: &Labels) -> u64 {
        self.inner.lock().unwrap().latency.count(labels)
    }

    /// Return the sum of all latency observations for the given labels.
    pub fn latency_sum(&self, labels: &Labels) -> f64 {
        self.inner.lock().unwrap().latency.sum(labels)
    }

    /// Return the current token count for the given labels (including `direction`).
    pub fn token_count(&self, labels: &Labels, direction: &str) -> u64 {
        let mut key = labels.clone();
        key.set("direction", direction);
        self.inner.lock().unwrap().tokens.get(&key)
    }

    /// Return the current error count for the given labels (including `kind`).
    pub fn error_count(&self, labels: &Labels, kind: &str) -> u64 {
        let mut key = labels.clone();
        key.set("kind", kind);
        self.inner.lock().unwrap().errors.get(&key)
    }

    /// Return the total cache hits for the given cache type.
    pub fn cache_hit_count(&self, cache_type: &str) -> u64 {
        let key = Labels::new().label("cache_type", cache_type);
        self.inner.lock().unwrap().cache_hits.get(&key)
    }

    /// Return the total cache misses for the given cache type.
    pub fn cache_miss_count(&self, cache_type: &str) -> u64 {
        let key = Labels::new().label("cache_type", cache_type);
        self.inner.lock().unwrap().cache_misses.get(&key)
    }

    // ── Prometheus text exposition ─────────────────────────────────────────

    /// Render all metrics in Prometheus text exposition format (version 0.0.4).
    pub fn render_prometheus(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let mut out = String::new();
        out.push_str(&inner.latency.render(
            "ai_request_latency_seconds",
            "LLM request latency in seconds",
        ));
        out.push('\n');
        out.push_str(
            &inner
                .tokens
                .render("ai_tokens_total", "Total tokens processed"),
        );
        out.push('\n');
        out.push_str(
            &inner
                .errors
                .render("ai_errors_total", "Total errors encountered"),
        );
        out.push('\n');
        out.push_str(
            &inner
                .cache_hits
                .render("ai_cache_hits_total", "Total cache hit events"),
        );
        out.push('\n');
        out.push_str(
            &inner
                .cache_misses
                .render("ai_cache_misses_total", "Total cache miss events"),
        );
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_labels() -> Labels {
        Labels::new()
            .provider("openai")
            .model("gpt-4o-mini")
            .agent("test-agent")
    }

    // REQ-11.3: histograms for latency with p50/p95/p99 percentiles
    #[test]
    fn test_latency_histogram_percentiles() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();

        // Record 100 observations: 1ms to 100ms
        for i in 1..=100 {
            registry.record_latency(&labels, i as f64 * 0.001);
        }

        let p50 = registry.latency_p50(&labels).unwrap();
        let p95 = registry.latency_p95(&labels).unwrap();
        let p99 = registry.latency_p99(&labels).unwrap();

        // p50 should be around 50ms
        assert!(p50 >= 0.049 && p50 <= 0.051, "p50={}", p50);
        // p95 should be around 95ms
        assert!(p95 >= 0.094 && p95 <= 0.096, "p95={}", p95);
        // p99 should be around 99ms
        assert!(p99 >= 0.098 && p99 <= 0.100, "p99={}", p99);
    }

    // REQ-11.3: latency count and sum are tracked
    #[test]
    fn test_latency_count_and_sum() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();

        registry.record_latency(&labels, 0.1);
        registry.record_latency(&labels, 0.2);
        registry.record_latency(&labels, 0.3);

        assert_eq!(registry.latency_count(&labels), 3);
        let sum = registry.latency_sum(&labels);
        assert!((sum - 0.6).abs() < 1e-9, "sum={}", sum);
    }

    // REQ-11.3: counters for tokens (input + output)
    #[test]
    fn test_token_counters_input_output() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();

        registry.inc_tokens(&labels, "input", 100);
        registry.inc_tokens(&labels, "input", 50);
        registry.inc_tokens(&labels, "output", 200);

        assert_eq!(registry.token_count(&labels, "input"), 150);
        assert_eq!(registry.token_count(&labels, "output"), 200);
    }

    // REQ-11.3: counters for errors
    #[test]
    fn test_error_counters() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();

        registry.inc_errors(&labels, "timeout");
        registry.inc_errors(&labels, "timeout");
        registry.inc_errors(&labels, "rate_limit");

        assert_eq!(registry.error_count(&labels, "timeout"), 2);
        assert_eq!(registry.error_count(&labels, "rate_limit"), 1);
        assert_eq!(registry.error_count(&labels, "other"), 0);
    }

    // REQ-11.3: cache hit/miss counters
    #[test]
    fn test_cache_counters() {
        let registry = MetricsRegistry::new();

        registry.inc_cache_hit("response");
        registry.inc_cache_hit("response");
        registry.inc_cache_hit("semantic");
        registry.inc_cache_miss("response");

        assert_eq!(registry.cache_hit_count("response"), 2);
        assert_eq!(registry.cache_hit_count("semantic"), 1);
        assert_eq!(registry.cache_miss_count("response"), 1);
        assert_eq!(registry.cache_miss_count("semantic"), 0);
    }

    // REQ-11.3: per-provider and per-model dimensions are respected
    #[test]
    fn test_per_provider_per_model_dimensions() {
        let registry = MetricsRegistry::new();

        let openai = Labels::new().provider("openai").model("gpt-4o");
        let anthropic = Labels::new().provider("anthropic").model("claude-3");

        registry.record_latency(&openai, 0.1);
        registry.record_latency(&anthropic, 0.5);

        assert_eq!(registry.latency_count(&openai), 1);
        assert_eq!(registry.latency_count(&anthropic), 1);

        // Counts must be independent
        let p50_openai = registry.latency_p50(&openai).unwrap();
        let p50_anthropic = registry.latency_p50(&anthropic).unwrap();
        assert!((p50_openai - 0.1).abs() < 1e-9);
        assert!((p50_anthropic - 0.5).abs() < 1e-9);
    }

    // REQ-11.3: Prometheus text format is rendered correctly
    #[test]
    fn test_prometheus_text_format() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();

        registry.record_latency(&labels, 0.123);
        registry.inc_tokens(&labels, "input", 42);
        registry.inc_errors(&labels, "timeout");
        registry.inc_cache_hit("response");
        registry.inc_cache_miss("semantic");

        let text = registry.render_prometheus();

        // Must contain metric names and TYPE annotations
        assert!(text.contains("# TYPE ai_request_latency_seconds histogram"));
        assert!(text.contains("# TYPE ai_tokens_total counter"));
        assert!(text.contains("# TYPE ai_errors_total counter"));
        assert!(text.contains("# TYPE ai_cache_hits_total counter"));
        assert!(text.contains("# TYPE ai_cache_misses_total counter"));

        // Must contain bucket lines
        assert!(text.contains("ai_request_latency_seconds_bucket"));
        assert!(text.contains("ai_request_latency_seconds_count"));
        assert!(text.contains("ai_request_latency_seconds_sum"));

        // Must contain label values
        assert!(text.contains("openai"));
        assert!(text.contains("gpt-4o-mini"));
    }

    // REQ-11.3: registry can be cloned (Arc-shared state)
    #[test]
    fn test_registry_clone_shares_state() {
        let r1 = MetricsRegistry::new();
        let r2 = r1.clone();
        let labels = provider_labels();

        r1.record_latency(&labels, 0.1);
        assert_eq!(r2.latency_count(&labels), 1);
    }

    // REQ-11.3: no observations returns None for percentiles
    #[test]
    fn test_no_observations_returns_none() {
        let registry = MetricsRegistry::new();
        let labels = provider_labels();
        assert!(registry.latency_p50(&labels).is_none());
        assert!(registry.latency_p95(&labels).is_none());
        assert!(registry.latency_p99(&labels).is_none());
    }
}
