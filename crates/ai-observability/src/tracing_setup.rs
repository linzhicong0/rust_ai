use opentelemetry::trace::Tracer;
use tracing_opentelemetry::PreSampledTracer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// Output format for tracing logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceFormat {
    Pretty,
    Json,
}

/// Configuration for tracing subscriber setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracingConfig {
    pub service_name: String,
    pub env_filter: String,
    pub format: TraceFormat,
    pub emit_span_events: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "rust_ai".to_string(),
            env_filter: "info".to_string(),
            format: TraceFormat::Pretty,
            emit_span_events: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TracingSetupError {
    #[error("failed to install tracing subscriber: {0}")]
    Install(String),

    #[error("invalid tracing env filter: {0}")]
    EnvFilter(String),
}

fn make_env_filter(config: &TracingConfig) -> Result<EnvFilter, TracingSetupError> {
    EnvFilter::try_new(config.env_filter.clone())
        .map_err(|error| TracingSetupError::EnvFilter(error.to_string()))
}

fn span_events(config: &TracingConfig) -> FmtSpan {
    if config.emit_span_events {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    }
}

/// Install a global tracing subscriber that emits standard tracing output.
pub fn init_tracing(config: TracingConfig) -> Result<(), TracingSetupError> {
    let env_filter = make_env_filter(&config)?;
    match config.format {
        TraceFormat::Pretty => Registry::default()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(true)
                    .with_span_events(span_events(&config)),
            )
            .try_init()
            .map_err(|error| TracingSetupError::Install(error.to_string())),
        TraceFormat::Json => Registry::default()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_target(true)
                    .with_span_events(span_events(&config)),
            )
            .try_init()
            .map_err(|error| TracingSetupError::Install(error.to_string())),
    }
}

/// Install a global tracing subscriber with an OpenTelemetry tracer and local log output.
pub fn init_tracing_with_opentelemetry<T>(
    config: TracingConfig,
    tracer: T,
) -> Result<(), TracingSetupError>
where
    T: Tracer + PreSampledTracer + Send + Sync + 'static,
    T::Span: Send + Sync + 'static,
{
    let env_filter = make_env_filter(&config)?;
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    Registry::default()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_span_events(span_events(&config)),
        )
        .with(telemetry)
        .try_init()
        .map_err(|error| TracingSetupError::Install(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "rust_ai");
        assert_eq!(config.env_filter, "info");
        assert_eq!(config.format, TraceFormat::Pretty);
        assert!(config.emit_span_events);
    }

    #[test]
    fn test_make_env_filter_rejects_invalid_expression() {
        let config = TracingConfig {
            env_filter: "[".to_string(),
            ..TracingConfig::default()
        };

        let result = make_env_filter(&config);
        assert!(result.is_err());
    }
}
