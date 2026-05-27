// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Serverless Support (REQ-16.4)
//!
//! Support for serverless deployment for event-driven and low-cost workloads.
//! Covers AWS Lambda, Google Cloud Functions, Azure Functions with cold-start
//! optimization and multiple event trigger types (HTTP, queue, schedule).
//!
//! ## Example
//!
//! ```rust
//! use ai_core::serverless::{
//!     ServerlessFunction, ServerlessPlatform, EventTrigger, ColdStartConfig,
//!     ServerlessConfig,
//! };
//!
//! let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda)
//!     .with_trigger(EventTrigger::Http { path: "/api/chat".into(), method: "POST".into() })
//!     .with_cold_start(ColdStartConfig { pre_warm: true, ..Default::default() });
//!
//! let func = ServerlessFunction::new(config);
//! assert_eq!(func.platform(), &ServerlessPlatform::AwsLambda);
//! ```

use std::collections::HashMap;
use std::time::Duration;

// ── ServerlessPlatform ────────────────────────────────────────────────────────

/// Supported serverless platforms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ServerlessPlatform {
    /// AWS Lambda.
    AwsLambda,
    /// Google Cloud Functions.
    GoogleCloudFunctions,
    /// Azure Functions.
    AzureFunctions,
    /// Custom serverless platform.
    Custom(String),
}

impl ServerlessPlatform {
    /// Return the platform name.
    pub fn name(&self) -> &str {
        match self {
            ServerlessPlatform::AwsLambda => "aws_lambda",
            ServerlessPlatform::GoogleCloudFunctions => "google_cloud_functions",
            ServerlessPlatform::AzureFunctions => "azure_functions",
            ServerlessPlatform::Custom(name) => name.as_str(),
        }
    }
}

// ── EventTrigger ──────────────────────────────────────────────────────────────

/// Event triggers that can invoke a serverless function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventTrigger {
    /// HTTP request trigger (API Gateway, Cloud Endpoints, etc.).
    Http {
        /// URL path for the HTTP endpoint.
        path: String,
        /// HTTP method (GET, POST, etc.).
        method: String,
    },
    /// Message queue trigger (SQS, Pub/Sub, Service Bus).
    Queue {
        /// Queue name or ARN.
        queue_name: String,
        /// Batch size for processing messages.
        batch_size: u32,
    },
    /// Scheduled trigger (cron/rate expression).
    Schedule {
        /// Cron or rate expression.
        expression: String,
    },
    /// Storage event trigger (S3, GCS, Blob Storage).
    Storage {
        /// Bucket or container name.
        bucket: String,
        /// Event type (e.g., "object.created").
        event_type: String,
    },
    /// Custom event trigger.
    Custom {
        /// Trigger source name.
        source: String,
        /// Configuration as JSON-like map.
        config: HashMap<String, String>,
    },
}

impl EventTrigger {
    /// Return the trigger type name.
    pub fn trigger_type(&self) -> &str {
        match self {
            EventTrigger::Http { .. } => "http",
            EventTrigger::Queue { .. } => "queue",
            EventTrigger::Schedule { .. } => "schedule",
            EventTrigger::Storage { .. } => "storage",
            EventTrigger::Custom { .. } => "custom",
        }
    }
}

// ── ColdStartConfig ───────────────────────────────────────────────────────────

/// Cold-start optimization configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdStartConfig {
    /// Whether to pre-warm the function (provisioned concurrency / min instances).
    pub pre_warm: bool,
    /// Number of pre-warmed instances.
    pub pre_warm_count: u32,
    /// Whether to use lazy initialization for heavy resources.
    pub lazy_init: bool,
    /// Maximum initialization time allowed.
    pub max_init_duration: Duration,
    /// Whether to minimize dependencies for faster startup.
    pub minimize_dependencies: bool,
}

impl Default for ColdStartConfig {
    fn default() -> Self {
        Self {
            pre_warm: false,
            pre_warm_count: 1,
            lazy_init: true,
            max_init_duration: Duration::from_secs(10),
            minimize_dependencies: true,
        }
    }
}

// ── ServerlessConfig ──────────────────────────────────────────────────────────

/// Configuration for a serverless function deployment.
#[derive(Debug, Clone)]
pub struct ServerlessConfig {
    /// Function name/identifier.
    pub function_name: String,
    /// Target serverless platform.
    pub platform: ServerlessPlatform,
    /// Runtime (e.g., "provided.al2023" for Rust on Lambda).
    pub runtime: String,
    /// Memory allocation in MB.
    pub memory_mb: u32,
    /// Function timeout.
    pub timeout: Duration,
    /// Event triggers.
    pub triggers: Vec<EventTrigger>,
    /// Cold-start optimization settings.
    pub cold_start: ColdStartConfig,
    /// Environment variables.
    pub env_vars: HashMap<String, String>,
    /// Resource tags/labels.
    pub tags: HashMap<String, String>,
}

impl ServerlessConfig {
    /// Create a new serverless configuration.
    pub fn new(function_name: impl Into<String>, platform: ServerlessPlatform) -> Self {
        let runtime = match &platform {
            ServerlessPlatform::AwsLambda => "provided.al2023".to_string(),
            ServerlessPlatform::GoogleCloudFunctions => "rust".to_string(),
            ServerlessPlatform::AzureFunctions => "custom".to_string(),
            ServerlessPlatform::Custom(_) => "custom".to_string(),
        };

        Self {
            function_name: function_name.into(),
            platform,
            runtime,
            memory_mb: 256,
            timeout: Duration::from_secs(30),
            triggers: Vec::new(),
            cold_start: ColdStartConfig::default(),
            env_vars: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    /// Set the runtime.
    pub fn with_runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = runtime.into();
        self
    }

    /// Set memory allocation.
    pub fn with_memory_mb(mut self, memory_mb: u32) -> Self {
        self.memory_mb = memory_mb;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add an event trigger.
    pub fn with_trigger(mut self, trigger: EventTrigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Set cold-start optimization config.
    pub fn with_cold_start(mut self, config: ColdStartConfig) -> Self {
        self.cold_start = config;
        self
    }

    /// Add an environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Add a tag/label.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

// ── ServerlessError ───────────────────────────────────────────────────────────

/// Errors during serverless operations.
#[derive(Debug, thiserror::Error)]
pub enum ServerlessError {
    /// Invalid configuration.
    #[error("Invalid serverless config: {0}")]
    InvalidConfig(String),
    /// Platform-specific error.
    #[error("Platform error ({platform}): {message}")]
    PlatformError { platform: String, message: String },
    /// Cold-start optimization error.
    #[error("Cold-start error: {0}")]
    ColdStartError(String),
    /// Deployment error.
    #[error("Deployment error: {0}")]
    DeploymentError(String),
}

// ── FunctionStatus ────────────────────────────────────────────────────────────

/// Status of a serverless function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionStatus {
    /// Function is being created/updated.
    Pending,
    /// Function is active and ready to serve.
    Active,
    /// Function is inactive (scaled to zero).
    Inactive,
    /// Function deployment failed.
    Failed(String),
}

// ── ServerlessFunction ────────────────────────────────────────────────────────

/// Manages a serverless function deployment and lifecycle.
#[derive(Debug)]
pub struct ServerlessFunction {
    config: ServerlessConfig,
    status: FunctionStatus,
    invocation_count: u64,
    cold_starts: u64,
}

impl ServerlessFunction {
    /// Create a new serverless function.
    pub fn new(config: ServerlessConfig) -> Self {
        Self {
            config,
            status: FunctionStatus::Pending,
            invocation_count: 0,
            cold_starts: 0,
        }
    }

    /// Get the serverless platform.
    pub fn platform(&self) -> &ServerlessPlatform {
        &self.config.platform
    }

    /// Get the function configuration.
    pub fn config(&self) -> &ServerlessConfig {
        &self.config
    }

    /// Get the current status.
    pub fn status(&self) -> &FunctionStatus {
        &self.status
    }

    /// Get total invocation count.
    pub fn invocation_count(&self) -> u64 {
        self.invocation_count
    }

    /// Get cold-start count.
    pub fn cold_starts(&self) -> u64 {
        self.cold_starts
    }

    /// Validate the serverless configuration.
    pub fn validate(&self) -> Result<(), ServerlessError> {
        if self.config.function_name.is_empty() {
            return Err(ServerlessError::InvalidConfig(
                "function_name is required".into(),
            ));
        }

        if self.config.memory_mb == 0 {
            return Err(ServerlessError::InvalidConfig(
                "memory_mb must be greater than 0".into(),
            ));
        }

        if self.config.timeout.as_secs() == 0 {
            return Err(ServerlessError::InvalidConfig(
                "timeout must be greater than 0".into(),
            ));
        }

        // Platform-specific memory limits
        let max_memory = match &self.config.platform {
            ServerlessPlatform::AwsLambda => 10240,
            ServerlessPlatform::GoogleCloudFunctions => 32768,
            ServerlessPlatform::AzureFunctions => 14336,
            ServerlessPlatform::Custom(_) => u32::MAX,
        };

        if self.config.memory_mb > max_memory {
            return Err(ServerlessError::InvalidConfig(format!(
                "memory_mb {} exceeds platform limit {}",
                self.config.memory_mb, max_memory
            )));
        }

        Ok(())
    }

    /// Deploy the function (simulation - sets status to Active).
    pub fn deploy(&mut self) -> Result<(), ServerlessError> {
        self.validate()?;
        self.status = FunctionStatus::Active;
        Ok(())
    }

    /// Simulate an invocation.
    pub fn invoke(&mut self, is_cold_start: bool) -> Result<(), ServerlessError> {
        if self.status != FunctionStatus::Active {
            return Err(ServerlessError::PlatformError {
                platform: self.config.platform.name().to_string(),
                message: "function is not active".into(),
            });
        }

        self.invocation_count += 1;
        if is_cold_start {
            self.cold_starts += 1;
        }
        Ok(())
    }

    /// Deactivate the function (scale to zero).
    pub fn deactivate(&mut self) {
        self.status = FunctionStatus::Inactive;
    }

    /// Mark the function as failed.
    pub fn mark_failed(&mut self, reason: impl Into<String>) {
        self.status = FunctionStatus::Failed(reason.into());
    }

    /// Get the configured triggers.
    pub fn triggers(&self) -> &[EventTrigger] {
        &self.config.triggers
    }

    /// Calculate the cold-start ratio.
    pub fn cold_start_ratio(&self) -> f64 {
        if self.invocation_count == 0 {
            return 0.0;
        }
        self.cold_starts as f64 / self.invocation_count as f64
    }

    /// Check if cold-start optimization is enabled.
    pub fn is_cold_start_optimized(&self) -> bool {
        self.config.cold_start.pre_warm || self.config.cold_start.lazy_init
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-16.4: Support AWS Lambda
    #[test]
    fn test_aws_lambda_function() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda)
            .with_memory_mb(512)
            .with_timeout(Duration::from_secs(60));

        let func = ServerlessFunction::new(config);
        assert_eq!(func.platform(), &ServerlessPlatform::AwsLambda);
        assert_eq!(func.config().memory_mb, 512);
        assert_eq!(func.config().timeout, Duration::from_secs(60));
        assert_eq!(func.config().runtime, "provided.al2023");
    }

    // REQ-16.4: Support Google Cloud Functions
    #[test]
    fn test_google_cloud_functions() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::GoogleCloudFunctions)
            .with_memory_mb(256);

        let func = ServerlessFunction::new(config);
        assert_eq!(func.platform(), &ServerlessPlatform::GoogleCloudFunctions);
        assert_eq!(func.platform().name(), "google_cloud_functions");
    }

    // REQ-16.4: Support Azure Functions
    #[test]
    fn test_azure_functions() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AzureFunctions);

        let func = ServerlessFunction::new(config);
        assert_eq!(func.platform(), &ServerlessPlatform::AzureFunctions);
        assert_eq!(func.platform().name(), "azure_functions");
    }

    // REQ-16.4: Cold-start optimization (minimal initialization)
    #[test]
    fn test_cold_start_optimization() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda)
            .with_cold_start(ColdStartConfig {
                pre_warm: true,
                pre_warm_count: 3,
                lazy_init: true,
                max_init_duration: Duration::from_secs(5),
                minimize_dependencies: true,
            });

        let func = ServerlessFunction::new(config);
        assert!(func.is_cold_start_optimized());
        assert!(func.config().cold_start.pre_warm);
        assert_eq!(func.config().cold_start.pre_warm_count, 3);
        assert!(func.config().cold_start.minimize_dependencies);
    }

    // REQ-16.4: Cold-start ratio tracking
    #[test]
    fn test_cold_start_tracking() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda);
        let mut func = ServerlessFunction::new(config);
        func.deploy().unwrap();

        // First invocation is cold
        func.invoke(true).unwrap();
        // Next invocations are warm
        func.invoke(false).unwrap();
        func.invoke(false).unwrap();
        func.invoke(false).unwrap();

        assert_eq!(func.invocation_count(), 4);
        assert_eq!(func.cold_starts(), 1);
        assert!((func.cold_start_ratio() - 0.25).abs() < f64::EPSILON);
    }

    // REQ-16.4: Event triggers - HTTP
    #[test]
    fn test_http_trigger() {
        let config = ServerlessConfig::new("api-handler", ServerlessPlatform::AwsLambda)
            .with_trigger(EventTrigger::Http {
                path: "/api/chat".into(),
                method: "POST".into(),
            });

        let func = ServerlessFunction::new(config);
        assert_eq!(func.triggers().len(), 1);
        assert_eq!(func.triggers()[0].trigger_type(), "http");

        if let EventTrigger::Http { path, method } = &func.triggers()[0] {
            assert_eq!(path, "/api/chat");
            assert_eq!(method, "POST");
        } else {
            panic!("Expected HTTP trigger");
        }
    }

    // REQ-16.4: Event triggers - Queue
    #[test]
    fn test_queue_trigger() {
        let config =
            ServerlessConfig::new("queue-processor", ServerlessPlatform::GoogleCloudFunctions)
                .with_trigger(EventTrigger::Queue {
                    queue_name: "ai-tasks".into(),
                    batch_size: 10,
                });

        let func = ServerlessFunction::new(config);
        assert_eq!(func.triggers().len(), 1);
        assert_eq!(func.triggers()[0].trigger_type(), "queue");

        if let EventTrigger::Queue {
            queue_name,
            batch_size,
        } = &func.triggers()[0]
        {
            assert_eq!(queue_name, "ai-tasks");
            assert_eq!(*batch_size, 10);
        } else {
            panic!("Expected Queue trigger");
        }
    }

    // REQ-16.4: Event triggers - Schedule
    #[test]
    fn test_schedule_trigger() {
        let config = ServerlessConfig::new("cron-job", ServerlessPlatform::AzureFunctions)
            .with_trigger(EventTrigger::Schedule {
                expression: "0 */5 * * * *".into(),
            });

        let func = ServerlessFunction::new(config);
        assert_eq!(func.triggers().len(), 1);
        assert_eq!(func.triggers()[0].trigger_type(), "schedule");

        if let EventTrigger::Schedule { expression } = &func.triggers()[0] {
            assert_eq!(expression, "0 */5 * * * *");
        } else {
            panic!("Expected Schedule trigger");
        }
    }

    // REQ-16.4: Multiple triggers
    #[test]
    fn test_multiple_triggers() {
        let config = ServerlessConfig::new("multi-handler", ServerlessPlatform::AwsLambda)
            .with_trigger(EventTrigger::Http {
                path: "/api/invoke".into(),
                method: "POST".into(),
            })
            .with_trigger(EventTrigger::Queue {
                queue_name: "tasks".into(),
                batch_size: 5,
            })
            .with_trigger(EventTrigger::Schedule {
                expression: "rate(1 hour)".into(),
            });

        let func = ServerlessFunction::new(config);
        assert_eq!(func.triggers().len(), 3);
        assert_eq!(func.triggers()[0].trigger_type(), "http");
        assert_eq!(func.triggers()[1].trigger_type(), "queue");
        assert_eq!(func.triggers()[2].trigger_type(), "schedule");
    }

    // REQ-16.4: Function lifecycle
    #[test]
    fn test_function_lifecycle() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda);
        let mut func = ServerlessFunction::new(config);

        assert_eq!(func.status(), &FunctionStatus::Pending);

        func.deploy().unwrap();
        assert_eq!(func.status(), &FunctionStatus::Active);

        func.invoke(false).unwrap();
        assert_eq!(func.invocation_count(), 1);

        func.deactivate();
        assert_eq!(func.status(), &FunctionStatus::Inactive);
    }

    // REQ-16.4: Cannot invoke inactive function
    #[test]
    fn test_invoke_inactive_function_fails() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda);
        let mut func = ServerlessFunction::new(config);

        // Not yet deployed
        let result = func.invoke(false);
        assert!(result.is_err());
    }

    // REQ-16.4: Validation rejects invalid config
    #[test]
    fn test_validation_empty_name() {
        let config = ServerlessConfig::new("", ServerlessPlatform::AwsLambda);
        let func = ServerlessFunction::new(config);
        assert!(func.validate().is_err());
    }

    // REQ-16.4: Validation rejects zero memory
    #[test]
    fn test_validation_zero_memory() {
        let config =
            ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda).with_memory_mb(0);
        let func = ServerlessFunction::new(config);
        assert!(func.validate().is_err());
    }

    // REQ-16.4: Validation rejects excessive memory
    #[test]
    fn test_validation_excessive_memory() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda)
            .with_memory_mb(20000); // Exceeds Lambda's 10240 limit
        let func = ServerlessFunction::new(config);
        assert!(func.validate().is_err());
    }

    // REQ-16.4: Environment variables
    #[test]
    fn test_env_vars() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda)
            .with_env("MODEL_NAME", "gpt-4")
            .with_env("LOG_LEVEL", "info");

        assert_eq!(
            config.env_vars.get("MODEL_NAME"),
            Some(&"gpt-4".to_string())
        );
        assert_eq!(config.env_vars.get("LOG_LEVEL"), Some(&"info".to_string()));
    }

    // REQ-16.4: Failed deployment
    #[test]
    fn test_failed_function() {
        let config = ServerlessConfig::new("my-handler", ServerlessPlatform::AwsLambda);
        let mut func = ServerlessFunction::new(config);

        func.mark_failed("quota exceeded");
        assert!(matches!(func.status(), FunctionStatus::Failed(_)));
    }
}
