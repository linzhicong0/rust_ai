use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// A tracing event that can be exported to an observability backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation: String,
    pub timestamp_ms: u64,
    pub duration_ms: Option<u64>,
    pub attributes: HashMap<String, String>,
    pub status: TraceStatus,
}

/// Status of a trace/span event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceStatus {
    Ok,
    Error(String),
    Unset,
}

/// A metric event that can be exported to an observability backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricEvent {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp_ms: u64,
    pub metric_type: MetricType,
}

/// Supported metric kinds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// Configuration shared by exporter implementations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExporterConfig {
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub service_name: String,
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            api_key: None,
            batch_size: 1024,
            flush_interval_ms: 1_000,
            service_name: "rust_ai".to_string(),
        }
    }
}

/// Errors returned by observability exporters.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExporterError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("export failed: {0}")]
    ExportFailed(String),
    #[error("invalid exporter configuration: {0}")]
    ConfigError(String),
    #[error("exporter buffer is full")]
    BufferFull,
}

/// Common interface for observability backends.
#[async_trait]
pub trait ObservabilityExporter: Send + Sync {
    async fn export_traces(&self, traces: Vec<TraceEvent>) -> Result<(), ExporterError>;
    async fn export_metrics(&self, metrics: Vec<MetricEvent>) -> Result<(), ExporterError>;
    async fn flush(&self) -> Result<(), ExporterError>;
    fn name(&self) -> &str;
}

#[derive(Debug, Default, Clone)]
struct EventStore {
    pending_traces: Vec<TraceEvent>,
    pending_metrics: Vec<MetricEvent>,
    flushed_traces: Vec<TraceEvent>,
    flushed_metrics: Vec<MetricEvent>,
}

#[derive(Debug, Clone)]
pub struct OpenTelemetryExporter {
    config: ExporterConfig,
    name: String,
    store: Arc<Mutex<EventStore>>,
}

impl OpenTelemetryExporter {
    pub fn new(config: ExporterConfig) -> Result<Self, ExporterError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            name: "opentelemetry".to_string(),
            store: Arc::new(Mutex::new(EventStore::default())),
        })
    }

    pub fn pending_traces(&self) -> Result<Vec<TraceEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_traces.clone())
    }

    pub fn pending_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_metrics.clone())
    }

    pub fn flushed_traces(&self) -> Result<Vec<TraceEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_traces.clone())
    }

    pub fn flushed_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_metrics.clone())
    }
}

#[async_trait]
impl ObservabilityExporter for OpenTelemetryExporter {
    async fn export_traces(&self, traces: Vec<TraceEvent>) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        ensure_capacity(
            self.config.batch_size,
            store.pending_traces.len(),
            traces.len(),
        )?;
        store.pending_traces.extend(traces);
        Ok(())
    }

    async fn export_metrics(&self, metrics: Vec<MetricEvent>) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        ensure_capacity(
            self.config.batch_size,
            store.pending_metrics.len(),
            metrics.len(),
        )?;
        store.pending_metrics.extend(metrics);
        Ok(())
    }

    async fn flush(&self) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        let traces = std::mem::take(&mut store.pending_traces);
        let metrics = std::mem::take(&mut store.pending_metrics);
        store.flushed_traces.extend(traces);
        store.flushed_metrics.extend(metrics);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Default, Clone)]
struct LangSmithStore {
    pending_traces: Vec<TraceEvent>,
    pending_metrics: Vec<MetricEvent>,
    flushed_records: Vec<String>,
    flushed_metrics: Vec<MetricEvent>,
}

#[derive(Debug, Clone)]
pub struct LangSmithExporter {
    config: ExporterConfig,
    name: String,
    store: Arc<Mutex<LangSmithStore>>,
}

impl LangSmithExporter {
    pub fn new(config: ExporterConfig) -> Result<Self, ExporterError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            name: "langsmith".to_string(),
            store: Arc::new(Mutex::new(LangSmithStore::default())),
        })
    }

    pub fn pending_traces(&self) -> Result<Vec<TraceEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_traces.clone())
    }

    pub fn pending_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_metrics.clone())
    }

    pub fn flushed_records(&self) -> Result<Vec<String>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_records.clone())
    }

    pub fn flushed_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_metrics.clone())
    }

    fn format_trace_record(&self, trace: TraceEvent) -> Result<String, ExporterError> {
        let status = match trace.status {
            TraceStatus::Ok => "ok".to_string(),
            TraceStatus::Unset => "unset".to_string(),
            TraceStatus::Error(message) => format!("error:{message}"),
        };

        serde_json::to_string(&serde_json::json!({
            "backend": "langsmith",
            "service_name": self.config.service_name,
            "trace_id": trace.trace_id,
            "run_id": trace.span_id,
            "parent_run_id": trace.parent_span_id,
            "name": trace.operation,
            "start_time_ms": trace.timestamp_ms,
            "end_time_ms": trace.duration_ms.map(|duration| trace.timestamp_ms + duration),
            "status": status,
            "attributes": trace.attributes,
        }))
        .map_err(|error| ExporterError::ExportFailed(error.to_string()))
    }
}

#[async_trait]
impl ObservabilityExporter for LangSmithExporter {
    async fn export_traces(&self, traces: Vec<TraceEvent>) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        ensure_capacity(
            self.config.batch_size,
            store.pending_traces.len(),
            traces.len(),
        )?;
        store.pending_traces.extend(traces);
        Ok(())
    }

    async fn export_metrics(&self, metrics: Vec<MetricEvent>) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        ensure_capacity(
            self.config.batch_size,
            store.pending_metrics.len(),
            metrics.len(),
        )?;
        store.pending_metrics.extend(metrics);
        Ok(())
    }

    async fn flush(&self) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        let traces = std::mem::take(&mut store.pending_traces);
        let metrics = std::mem::take(&mut store.pending_metrics);
        let mut formatted_records = Vec::with_capacity(traces.len());

        for trace in traces {
            formatted_records.push(self.format_trace_record(trace)?);
        }

        store.flushed_records.extend(formatted_records);
        store.flushed_metrics.extend(metrics);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct InMemoryExporter {
    name: String,
    store: Arc<Mutex<EventStore>>,
}

impl Default for InMemoryExporter {
    fn default() -> Self {
        Self::new("in-memory")
    }
}

impl InMemoryExporter {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            store: Arc::new(Mutex::new(EventStore::default())),
        }
    }

    pub fn pending_traces(&self) -> Result<Vec<TraceEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_traces.clone())
    }

    pub fn pending_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.pending_metrics.clone())
    }

    pub fn flushed_traces(&self) -> Result<Vec<TraceEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_traces.clone())
    }

    pub fn flushed_metrics(&self) -> Result<Vec<MetricEvent>, ExporterError> {
        Ok(lock(&self.store, self.name())?.flushed_metrics.clone())
    }
}

#[async_trait]
impl ObservabilityExporter for InMemoryExporter {
    async fn export_traces(&self, traces: Vec<TraceEvent>) -> Result<(), ExporterError> {
        lock(&self.store, self.name())?
            .pending_traces
            .extend(traces);
        Ok(())
    }

    async fn export_metrics(&self, metrics: Vec<MetricEvent>) -> Result<(), ExporterError> {
        lock(&self.store, self.name())?
            .pending_metrics
            .extend(metrics);
        Ok(())
    }

    async fn flush(&self) -> Result<(), ExporterError> {
        let mut store = lock(&self.store, self.name())?;
        let traces = std::mem::take(&mut store.pending_traces);
        let metrics = std::mem::take(&mut store.pending_metrics);
        store.flushed_traces.extend(traces);
        store.flushed_metrics.extend(metrics);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Default)]
pub struct CompositeExporter {
    exporters: Vec<Box<dyn ObservabilityExporter>>,
}

impl CompositeExporter {
    pub fn new(exporters: Vec<Box<dyn ObservabilityExporter>>) -> Self {
        Self { exporters }
    }

    pub fn add_exporter(&mut self, exporter: Box<dyn ObservabilityExporter>) {
        self.exporters.push(exporter);
    }
}

#[async_trait]
impl ObservabilityExporter for CompositeExporter {
    async fn export_traces(&self, traces: Vec<TraceEvent>) -> Result<(), ExporterError> {
        for exporter in &self.exporters {
            exporter.export_traces(traces.clone()).await?;
        }
        Ok(())
    }

    async fn export_metrics(&self, metrics: Vec<MetricEvent>) -> Result<(), ExporterError> {
        for exporter in &self.exporters {
            exporter.export_metrics(metrics.clone()).await?;
        }
        Ok(())
    }

    async fn flush(&self) -> Result<(), ExporterError> {
        for exporter in &self.exporters {
            exporter.flush().await?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "composite"
    }
}

fn validate_config(config: &ExporterConfig) -> Result<(), ExporterError> {
    if config.batch_size == 0 {
        return Err(ExporterError::ConfigError(
            "batch_size must be greater than zero".to_string(),
        ));
    }

    if config.service_name.trim().is_empty() {
        return Err(ExporterError::ConfigError(
            "service_name must not be empty".to_string(),
        ));
    }

    Ok(())
}

fn ensure_capacity(
    batch_size: usize,
    current_len: usize,
    incoming_len: usize,
) -> Result<(), ExporterError> {
    if current_len.saturating_add(incoming_len) > batch_size {
        return Err(ExporterError::BufferFull);
    }
    Ok(())
}

fn lock<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Result<MutexGuard<'a, T>, ExporterError> {
    mutex
        .lock()
        .map_err(|_| ExporterError::ExportFailed(format!("{name} exporter state is poisoned")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FailingExporter;

    #[async_trait]
    impl ObservabilityExporter for FailingExporter {
        async fn export_traces(&self, _traces: Vec<TraceEvent>) -> Result<(), ExporterError> {
            Err(ExporterError::ExportFailed(
                "trace export failed".to_string(),
            ))
        }

        async fn export_metrics(&self, _metrics: Vec<MetricEvent>) -> Result<(), ExporterError> {
            Err(ExporterError::ExportFailed(
                "metric export failed".to_string(),
            ))
        }

        async fn flush(&self) -> Result<(), ExporterError> {
            Err(ExporterError::ConnectionFailed("flush failed".to_string()))
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    fn sample_trace() -> TraceEvent {
        TraceEvent {
            trace_id: "trace-1".to_string(),
            span_id: "span-1".to_string(),
            parent_span_id: Some("parent-1".to_string()),
            operation: "llm.complete".to_string(),
            timestamp_ms: 1_700_000_000_000,
            duration_ms: Some(42),
            attributes: HashMap::from([
                ("provider".to_string(), "openai".to_string()),
                ("model".to_string(), "gpt-4o-mini".to_string()),
            ]),
            status: TraceStatus::Ok,
        }
    }

    fn sample_metric() -> MetricEvent {
        MetricEvent {
            name: "ai.request.duration".to_string(),
            value: 42.0,
            labels: HashMap::from([
                ("provider".to_string(), "openai".to_string()),
                ("kind".to_string(), "latency".to_string()),
            ]),
            timestamp_ms: 1_700_000_000_001,
            metric_type: MetricType::Histogram,
        }
    }

    #[tokio::test]
    async fn export_traces_through_in_memory_exporter() {
        let exporter = InMemoryExporter::default();
        let trace = sample_trace();

        exporter.export_traces(vec![trace.clone()]).await.unwrap();

        assert_eq!(exporter.pending_traces().unwrap(), vec![trace]);
    }

    #[tokio::test]
    async fn export_metrics_and_verify() {
        let exporter = InMemoryExporter::default();
        let metric = sample_metric();

        exporter.export_metrics(vec![metric.clone()]).await.unwrap();

        assert_eq!(exporter.pending_metrics().unwrap(), vec![metric]);
    }

    #[tokio::test]
    async fn composite_exporter_fans_out_to_multiple_exporters() {
        let exporter_a = InMemoryExporter::new("a");
        let exporter_b = InMemoryExporter::new("b");
        let trace = sample_trace();
        let metric = sample_metric();

        let composite = CompositeExporter::new(vec![
            Box::new(exporter_a.clone()),
            Box::new(exporter_b.clone()),
        ]);

        composite.export_traces(vec![trace.clone()]).await.unwrap();
        composite
            .export_metrics(vec![metric.clone()])
            .await
            .unwrap();

        assert_eq!(exporter_a.pending_traces().unwrap(), vec![trace.clone()]);
        assert_eq!(exporter_b.pending_traces().unwrap(), vec![trace]);
        assert_eq!(exporter_a.pending_metrics().unwrap(), vec![metric.clone()]);
        assert_eq!(exporter_b.pending_metrics().unwrap(), vec![metric]);
    }

    #[tokio::test]
    async fn flush_moves_pending_events_into_flushed_buffers() {
        let exporter = OpenTelemetryExporter::new(ExporterConfig::default()).unwrap();
        let trace = sample_trace();
        let metric = sample_metric();

        exporter.export_traces(vec![trace.clone()]).await.unwrap();
        exporter.export_metrics(vec![metric.clone()]).await.unwrap();
        exporter.flush().await.unwrap();

        assert!(exporter.pending_traces().unwrap().is_empty());
        assert!(exporter.pending_metrics().unwrap().is_empty());
        assert_eq!(exporter.flushed_traces().unwrap(), vec![trace]);
        assert_eq!(exporter.flushed_metrics().unwrap(), vec![metric]);
    }

    #[tokio::test]
    async fn langsmith_flush_formats_records() {
        let exporter = LangSmithExporter::new(ExporterConfig::default()).unwrap();
        exporter.export_traces(vec![sample_trace()]).await.unwrap();
        exporter.flush().await.unwrap();

        let records = exporter.flushed_records().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].contains("\"backend\":\"langsmith\""));
        assert!(records[0].contains("\"run_id\":\"span-1\""));
        assert!(records[0].contains("\"name\":\"llm.complete\""));
    }

    #[tokio::test]
    async fn error_propagation_from_failed_exporters() {
        let composite = CompositeExporter::new(vec![Box::new(FailingExporter)]);

        let trace_error = composite
            .export_traces(vec![sample_trace()])
            .await
            .unwrap_err();
        let metric_error = composite
            .export_metrics(vec![sample_metric()])
            .await
            .unwrap_err();
        let flush_error = composite.flush().await.unwrap_err();

        assert_eq!(
            trace_error,
            ExporterError::ExportFailed("trace export failed".to_string())
        );
        assert_eq!(
            metric_error,
            ExporterError::ExportFailed("metric export failed".to_string())
        );
        assert_eq!(
            flush_error,
            ExporterError::ConnectionFailed("flush failed".to_string())
        );
    }

    #[test]
    fn exporter_config_is_validated() {
        let error = OpenTelemetryExporter::new(ExporterConfig {
            batch_size: 0,
            ..ExporterConfig::default()
        })
        .unwrap_err();

        assert_eq!(
            error,
            ExporterError::ConfigError("batch_size must be greater than zero".to_string())
        );
    }

    #[tokio::test]
    async fn opentelemetry_exporter_enforces_buffer_capacity() {
        let exporter = OpenTelemetryExporter::new(ExporterConfig {
            batch_size: 1,
            ..ExporterConfig::default()
        })
        .unwrap();

        exporter.export_traces(vec![sample_trace()]).await.unwrap();
        let error = exporter
            .export_traces(vec![sample_trace()])
            .await
            .unwrap_err();

        assert_eq!(error, ExporterError::BufferFull);
    }
}
