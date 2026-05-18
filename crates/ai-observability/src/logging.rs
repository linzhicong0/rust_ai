//! Structured logging for LLM interactions.
//!
//! Provides configurable logging with JSON output, multiple verbosity levels,
//! and automatic instrumentation of request/response cycles.

use serde::Serialize;
use std::time::Duration;
use tracing::{debug, error, info, span, Level, Span};
use tracing_subscriber::{
    fmt::format::FmtSpan, prelude::*, registry::Registry, EnvFilter, Layer,
};

/// Verbosity level for logging output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// No logging output.
    None,
    /// Only log errors.
    Error,
    /// Log warnings and errors.
    Warn,
    /// Log info, warnings, and errors (default).
    #[default]
    Info,
    /// Log debug info and above.
    Debug,
    /// Log everything including trace.
    Trace,
}

impl LogLevel {
    /// Convert to tracing Level.
    pub fn to_tracing_level(self) -> Option<Level> {
        match self {
            LogLevel::None => None,
            LogLevel::Error => Some(Level::ERROR),
            LogLevel::Warn => Some(Level::WARN),
            LogLevel::Info => Some(Level::INFO),
            LogLevel::Debug => Some(Level::DEBUG),
            LogLevel::Trace => Some(Level::TRACE),
        }
    }

    /// Parse from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(LogLevel::None),
            "error" => Some(LogLevel::Error),
            "warn" | "warning" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }
}

/// Configuration for the logging system.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level for stdout output.
    pub level: LogLevel,
    /// Whether to output JSON format (for machine parsing).
    pub json_format: bool,
    /// Whether to include span events (new spans, closes, etc.).
    pub span_events: bool,
    /// Whether to colorize output (only applies to non-JSON).
    pub ansi: bool,
    /// Target directory for log files (optional).
    pub log_file: Option<std::path::PathBuf>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::default(),
            json_format: true,
            span_events: false,
            ansi: true,
            log_file: None,
        }
    }
}

impl LoggingConfig {
    /// Create a new logging config with the specified level.
    pub fn new(level: LogLevel) -> Self {
        Self {
            level,
            ..Default::default()
        }
    }

    /// Enable JSON output for structured logging.
    pub fn with_json(mut self, json: bool) -> Self {
        self.json_format = json;
        self
    }

    /// Enable span events for detailed tracing.
    pub fn with_span_events(mut self, enabled: bool) -> Self {
        self.span_events = enabled;
        self
    }

    /// Disable ANSI colors in output.
    pub fn without_ansi(mut self) -> Self {
        self.ansi = false;
        self
    }

    /// Set a log file path for output.
    pub fn with_log_file(mut self, path: std::path::PathBuf) -> Self {
        self.log_file = Some(path);
        self
    }

    /// Initialize the logging system with this configuration.
    ///
    /// # Panics
    ///
    /// Panics if a global tracing subscriber is already set.
    pub fn init(self) -> Result<(), LoggingError> {
        let env_filter = self.build_env_filter();

        if self.json_format {
            self.init_json_logging(env_filter)
        } else {
            self.initPretty_logging(env_filter)
        }
    }

    fn build_env_filter(&self) -> EnvFilter {
        let base = match self.level.to_tracing_level() {
            Some(Level::ERROR) => "error",
            Some(Level::WARN) => "warn",
            Some(Level::INFO) => "info",
            Some(Level::DEBUG) => "debug",
            Some(Level::TRACE) => "trace",
            None => "off",
        };

        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(base))
    }

    fn init_json_logging(&self, env_filter: EnvFilter) -> Result<(), LoggingError> {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_file(true)
            .with_line_number(true)
            .with_span_events(if self.span_events {
                FmtSpan::NEW | FmtSpan::CLOSE
            } else {
                FmtSpan::NONE
            })
            .with_ansi(self.ansi)
            .with_filter(env_filter);

        let subscriber = Registry::default().with(fmt_layer);

        if let Some(log_path) = &self.log_file {
            // TODO: Add file appender layer
            tracing::debug!("Log file configured at: {:?}", log_path);
        }

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| LoggingError::AlreadyInitialized(e.to_string()))?;

        Ok(())
    }

    fn initPretty_logging(&self, env_filter: EnvFilter) -> Result<(), LoggingError> {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_file(true)
            .with_line_number(true)
            .with_span_events(if self.span_events {
                FmtSpan::NEW | FmtSpan::CLOSE
            } else {
                FmtSpan::NONE
            })
            .with_ansi(self.ansi)
            .with_filter(env_filter);

        let subscriber = Registry::default().with(fmt_layer);

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| LoggingError::AlreadyInitialized(e.to_string()))?;

        Ok(())
    }
}

/// Errors that can occur during logging initialization.
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    #[error("logging already initialized: {0}")]
    AlreadyInitialized(String),
}

/// LLM request logging context.
///
/// Use this to track the lifecycle of an LLM request with automatic timing.
#[derive(Debug)]
pub struct LlmRequestLogger {
    _span: Span,
    start: std::time::Instant,
}

impl LlmRequestLogger {
    /// Begin logging a new LLM request.
    ///
    /// Creates a tracing span with the request details for hierarchical logging.
    pub fn request(
        provider: &str,
        model: &str,
        messages_count: usize,
    ) -> Self {
        let _span = span!(
            Level::INFO,
            "llm_request",
            provider = %provider,
            model = %model,
            messages_count = messages_count
        );

        let enter = _span.enter();
        debug!("Starting LLM request");

        Self {
            _span,
            start: std::time::Instant::now(),
        }
    }

    /// Log the response and complete the request logging.
    ///
    /// Automatically calculates and logs timing information.
    pub fn complete(
        self,
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: &str,
    ) {
        let elapsed = self.start.elapsed();

        let total_tokens = prompt_tokens + completion_tokens;

        info!(
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens = total_tokens,
            finish_reason = %finish_reason,
            latency_ms = elapsed.as_millis(),
            "LLM request completed"
        );

        drop(self._span);
    }

    /// Log an error that occurred during the request.
    pub fn error(self, error: &str) {
        let elapsed = self.start.elapsed();

        error!(
            error = %error,
            latency_ms = elapsed.as_millis(),
            "LLM request failed"
        );

        drop(self._span);
    }
}

/// Log streaming chunk information.
pub fn log_stream_chunk(
    delta: Option<&str>,
    finish_reason: Option<&str>,
) {
    match (delta, finish_reason) {
        (Some(text), None) => {
            debug!(
                delta_length = text.len(),
                "Stream chunk received"
            );
        }
        (_, Some(reason)) => {
            debug!(
                finish_reason = %reason,
                "Stream completed"
            );
        }
        _ => {}
    }
}

/// Log tool execution.
pub fn log_tool_call(
    tool_name: &str,
    input: &str,
) {
    info!(
        tool = %tool_name,
        input_length = input.len(),
        "Tool call started"
    );
}

/// Log tool execution result.
pub fn log_tool_result(
    tool_name: &str,
    success: bool,
    duration: Duration,
) {
    if success {
        info!(
            tool = %tool_name,
            duration_ms = duration.as_millis(),
            "Tool call succeeded"
        );
    } else {
        error!(
            tool = %tool_name,
            duration_ms = duration.as_millis(),
            "Tool call failed"
        );
    }
}

/// Log agent iteration in the ReAct loop.
pub fn log_agent_iteration(
    agent_name: &str,
    iteration: u32,
    thinking: Option<&str>,
) {
    if let Some(thought) = thinking {
        debug!(
            agent = %agent_name,
            iteration = iteration,
            thought = %thought,
            "Agent thinking"
        );
    } else {
        debug!(
            agent = %agent_name,
            iteration = iteration,
            "Agent iteration"
        );
    }
}

/// Log pipeline step execution.
pub fn log_pipeline_step(
    pipeline_name: &str,
    step_name: &str,
) {
    info!(
        pipeline = %pipeline_name,
        step = %step_name,
        "Pipeline step started"
    );
}

/// Log caching events.
pub fn log_cache_hit(
    key: &str,
    hit: bool,
) {
    if hit {
        debug!(
            cache_key = %key,
            "Cache hit"
        );
    } else {
        debug!(
            cache_key = %key,
            "Cache miss"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!(LogLevel::from_str("none"), Some(LogLevel::None));
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_log_level_to_tracing() {
        assert_eq!(LogLevel::None.to_tracing_level(), None);
        assert_eq!(LogLevel::Error.to_tracing_level(), Some(Level::ERROR));
        assert_eq!(LogLevel::Warn.to_tracing_level(), Some(Level::WARN));
        assert_eq!(LogLevel::Info.to_tracing_level(), Some(Level::INFO));
        assert_eq!(LogLevel::Debug.to_tracing_level(), Some(Level::DEBUG));
        assert_eq!(LogLevel::Trace.to_tracing_level(), Some(Level::TRACE));
    }

    #[test]
    fn test_logging_config_builder() {
        let config = LoggingConfig::new(LogLevel::Debug)
            .with_json(true)
            .with_span_events(true)
            .without_ansi();

        assert_eq!(config.level, LogLevel::Debug);
        assert!(config.json_format);
        assert!(config.span_events);
        assert!(!config.ansi);
    }

    #[test]
    fn test_llm_request_logger_timing() {
        let logger = LlmRequestLogger::request("openai", "gpt-4", 2);
        std::thread::sleep(Duration::from_millis(10));
        logger.complete(100, 50, "stop");
    }
}
