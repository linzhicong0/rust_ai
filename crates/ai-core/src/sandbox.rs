// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Sandboxed Execution (REQ-3.6)
//!
//! Provides sandboxed tool execution with configurable permissions and resource limits.
//! Supports Docker, WASM, and native (seccomp) sandbox backends.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

/// Errors that can occur during sandboxed execution.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    #[error("Timeout: execution exceeded {0:?}")]
    Timeout(Duration),
    #[error("Sandbox creation error: {0}")]
    Creation(String),
    #[error("Execution error: {0}")]
    Execution(String),
    #[error("Sandbox backend not available: {0}")]
    BackendNotAvailable(String),
}

/// Sandbox backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxBackend {
    /// Docker container isolation.
    Docker,
    /// WebAssembly sandbox.
    Wasm,
    /// Native OS-level isolation (seccomp, namespaces).
    Native,
    /// No sandbox (for trusted tools).
    None,
}

/// Permission types for sandboxed execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SandboxPermission {
    /// Read filesystem access.
    FileRead,
    /// Write filesystem access.
    FileWrite,
    /// Network access (outbound).
    NetworkOutbound,
    /// Network access (inbound/listen).
    NetworkInbound,
    /// Process spawning.
    ProcessSpawn,
    /// Environment variable access.
    EnvAccess,
    /// System clock access.
    ClockAccess,
}

/// Resource limits for sandboxed execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum execution time.
    pub timeout: Duration,
    /// Maximum memory in bytes.
    pub max_memory_bytes: u64,
    /// Maximum CPU time in milliseconds.
    pub max_cpu_ms: u64,
    /// Maximum output size in bytes.
    pub max_output_bytes: u64,
    /// Maximum number of open file descriptors.
    pub max_open_files: u32,
    /// Maximum number of processes/threads.
    pub max_processes: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            max_cpu_ms: 10_000,                  // 10 seconds
            max_output_bytes: 1024 * 1024,       // 1 MB
            max_open_files: 64,
            max_processes: 10,
        }
    }
}

impl ResourceLimits {
    /// Create strict limits for untrusted code.
    pub fn strict() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
            max_cpu_ms: 2_000,                  // 2 seconds
            max_output_bytes: 256 * 1024,       // 256 KB
            max_open_files: 8,
            max_processes: 1,
        }
    }

    /// Create relaxed limits for trusted code.
    pub fn relaxed() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            max_cpu_ms: 120_000,                      // 2 minutes
            max_output_bytes: 10 * 1024 * 1024,       // 10 MB
            max_open_files: 1024,
            max_processes: 64,
        }
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set memory limit.
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set CPU time limit.
    pub fn with_max_cpu_ms(mut self, ms: u64) -> Self {
        self.max_cpu_ms = ms;
        self
    }
}

/// Configuration for a sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// The backend to use.
    pub backend: SandboxBackend,
    /// Permissions granted to the sandboxed code.
    pub permissions: HashSet<SandboxPermission>,
    /// Resource limits.
    pub resource_limits: ResourceLimits,
    /// Allowed filesystem paths (if FileRead/FileWrite granted).
    pub allowed_paths: Vec<String>,
    /// Allowed network hosts (if Network* granted).
    pub allowed_hosts: Vec<String>,
    /// Environment variables to pass through.
    pub env_vars: Vec<String>,
}

impl SandboxConfig {
    /// Create a new sandbox config with the given backend.
    pub fn new(backend: SandboxBackend) -> Self {
        Self {
            backend,
            permissions: HashSet::new(),
            resource_limits: ResourceLimits::default(),
            allowed_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            env_vars: Vec::new(),
        }
    }

    /// Create a minimal sandbox (no permissions).
    pub fn minimal(backend: SandboxBackend) -> Self {
        Self {
            backend,
            permissions: HashSet::new(),
            resource_limits: ResourceLimits::strict(),
            allowed_paths: Vec::new(),
            allowed_hosts: Vec::new(),
            env_vars: Vec::new(),
        }
    }

    /// Grant a permission.
    pub fn grant(mut self, permission: SandboxPermission) -> Self {
        self.permissions.insert(permission);
        self
    }

    /// Revoke a permission.
    pub fn revoke(mut self, permission: SandboxPermission) -> Self {
        self.permissions.remove(&permission);
        self
    }

    /// Set resource limits.
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = limits;
        self
    }

    /// Add an allowed filesystem path.
    pub fn allow_path(mut self, path: impl Into<String>) -> Self {
        self.allowed_paths.push(path.into());
        self
    }

    /// Add an allowed network host.
    pub fn allow_host(mut self, host: impl Into<String>) -> Self {
        self.allowed_hosts.push(host.into());
        self
    }

    /// Check if a permission is granted.
    pub fn has_permission(&self, permission: SandboxPermission) -> bool {
        self.permissions.contains(&permission)
    }

    /// Check if a path is allowed.
    pub fn is_path_allowed(&self, path: &str) -> bool {
        if !self.has_permission(SandboxPermission::FileRead)
            && !self.has_permission(SandboxPermission::FileWrite)
        {
            return false;
        }
        if self.allowed_paths.is_empty() {
            return true; // No restrictions when permission granted but no path list
        }
        self.allowed_paths.iter().any(|p| path.starts_with(p))
    }

    /// Check if a host is allowed.
    pub fn is_host_allowed(&self, host: &str) -> bool {
        if !self.has_permission(SandboxPermission::NetworkOutbound)
            && !self.has_permission(SandboxPermission::NetworkInbound)
        {
            return false;
        }
        if self.allowed_hosts.is_empty() {
            return true; // No restrictions when permission granted but no host list
        }
        self.allowed_hosts.iter().any(|h| host.contains(h))
    }
}

/// Result of sandboxed execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxExecutionResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Exit code (0 = success).
    pub exit_code: i32,
    /// Execution duration.
    pub duration_ms: u64,
    /// Memory used in bytes.
    pub memory_used_bytes: u64,
}

impl SandboxExecutionResult {
    /// Whether execution was successful.
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Trait for sandboxed execution backends.
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Execute code in the sandbox.
    async fn execute(
        &self,
        code: &str,
        config: &SandboxConfig,
    ) -> Result<SandboxExecutionResult, SandboxError>;

    /// Check permissions before execution.
    fn check_permissions(
        &self,
        required: &[SandboxPermission],
        config: &SandboxConfig,
    ) -> Result<(), SandboxError> {
        for perm in required {
            if !config.has_permission(*perm) {
                return Err(SandboxError::PermissionDenied(format!(
                    "Permission {:?} not granted",
                    perm
                )));
            }
        }
        Ok(())
    }

    /// Check if the backend is available.
    async fn is_available(&self) -> bool;
}

/// In-memory sandbox executor for testing purposes.
pub struct InMemorySandboxExecutor {
    available: bool,
}

impl InMemorySandboxExecutor {
    /// Create a new in-memory sandbox executor.
    pub fn new() -> Self {
        Self { available: true }
    }

    /// Create an unavailable executor.
    pub fn unavailable() -> Self {
        Self { available: false }
    }
}

impl Default for InMemorySandboxExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxExecutor for InMemorySandboxExecutor {
    async fn execute(
        &self,
        code: &str,
        config: &SandboxConfig,
    ) -> Result<SandboxExecutionResult, SandboxError> {
        if !self.available {
            return Err(SandboxError::BackendNotAvailable(format!(
                "{:?}",
                config.backend
            )));
        }

        // Simulate timeout check
        if config.resource_limits.timeout < Duration::from_millis(1) {
            return Err(SandboxError::Timeout(config.resource_limits.timeout));
        }

        // Simulate execution (in-memory just echoes code length info)
        let output = format!("Executed {} bytes of code", code.len());
        let memory_used = (code.len() as u64) * 10; // Simulated memory

        // Check memory limit
        if memory_used > config.resource_limits.max_memory_bytes {
            return Err(SandboxError::ResourceLimitExceeded(format!(
                "Memory: {memory_used} > {}",
                config.resource_limits.max_memory_bytes
            )));
        }

        Ok(SandboxExecutionResult {
            stdout: output,
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 10,
            memory_used_bytes: memory_used,
        })
    }

    async fn is_available(&self) -> bool {
        self.available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_config_minimal() {
        let config = SandboxConfig::minimal(SandboxBackend::Wasm);
        assert_eq!(config.backend, SandboxBackend::Wasm);
        assert!(config.permissions.is_empty());
        assert_eq!(config.resource_limits.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_sandbox_config_permissions() {
        let config = SandboxConfig::new(SandboxBackend::Docker)
            .grant(SandboxPermission::FileRead)
            .grant(SandboxPermission::NetworkOutbound);

        assert!(config.has_permission(SandboxPermission::FileRead));
        assert!(config.has_permission(SandboxPermission::NetworkOutbound));
        assert!(!config.has_permission(SandboxPermission::FileWrite));
        assert!(!config.has_permission(SandboxPermission::ProcessSpawn));
    }

    #[test]
    fn test_sandbox_config_revoke() {
        let config = SandboxConfig::new(SandboxBackend::Native)
            .grant(SandboxPermission::FileRead)
            .grant(SandboxPermission::FileWrite)
            .revoke(SandboxPermission::FileWrite);

        assert!(config.has_permission(SandboxPermission::FileRead));
        assert!(!config.has_permission(SandboxPermission::FileWrite));
    }

    #[test]
    fn test_sandbox_config_path_allowed() {
        let config = SandboxConfig::new(SandboxBackend::Docker)
            .grant(SandboxPermission::FileRead)
            .allow_path("/tmp")
            .allow_path("/data");

        assert!(config.is_path_allowed("/tmp/test.txt"));
        assert!(config.is_path_allowed("/data/file.csv"));
        assert!(!config.is_path_allowed("/etc/passwd"));
    }

    #[test]
    fn test_sandbox_config_path_no_permission() {
        let config = SandboxConfig::new(SandboxBackend::Docker).allow_path("/tmp");
        // No FileRead permission, so even allowed paths are denied
        assert!(!config.is_path_allowed("/tmp/test.txt"));
    }

    #[test]
    fn test_sandbox_config_host_allowed() {
        let config = SandboxConfig::new(SandboxBackend::Wasm)
            .grant(SandboxPermission::NetworkOutbound)
            .allow_host("api.example.com")
            .allow_host("cdn.example.com");

        assert!(config.is_host_allowed("api.example.com"));
        assert!(config.is_host_allowed("cdn.example.com"));
        assert!(!config.is_host_allowed("evil.com"));
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.timeout, Duration::from_secs(30));
        assert_eq!(limits.max_memory_bytes, 256 * 1024 * 1024);
        assert_eq!(limits.max_open_files, 64);
    }

    #[test]
    fn test_resource_limits_strict() {
        let limits = ResourceLimits::strict();
        assert_eq!(limits.timeout, Duration::from_secs(5));
        assert_eq!(limits.max_memory_bytes, 64 * 1024 * 1024);
        assert_eq!(limits.max_processes, 1);
    }

    #[test]
    fn test_resource_limits_relaxed() {
        let limits = ResourceLimits::relaxed();
        assert_eq!(limits.timeout, Duration::from_secs(300));
        assert_eq!(limits.max_memory_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(limits.max_processes, 64);
    }

    #[test]
    fn test_resource_limits_builder() {
        let limits = ResourceLimits::default()
            .with_timeout(Duration::from_secs(60))
            .with_max_memory(512 * 1024 * 1024)
            .with_max_cpu_ms(30_000);

        assert_eq!(limits.timeout, Duration::from_secs(60));
        assert_eq!(limits.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.max_cpu_ms, 30_000);
    }

    #[tokio::test]
    async fn test_sandbox_executor_execute() {
        let executor = InMemorySandboxExecutor::new();
        let config = SandboxConfig::new(SandboxBackend::Wasm);

        let result = executor.execute("print('hello')", &config).await.unwrap();
        assert!(result.success());
        assert!(result.stdout.contains("14 bytes"));
        assert!(result.stderr.is_empty());
    }

    #[tokio::test]
    async fn test_sandbox_executor_unavailable() {
        let executor = InMemorySandboxExecutor::unavailable();
        let config = SandboxConfig::new(SandboxBackend::Docker);

        let result = executor.execute("code", &config).await;
        assert!(matches!(result, Err(SandboxError::BackendNotAvailable(_))));
    }

    #[tokio::test]
    async fn test_sandbox_executor_is_available() {
        let available = InMemorySandboxExecutor::new();
        let unavailable = InMemorySandboxExecutor::unavailable();

        assert!(available.is_available().await);
        assert!(!unavailable.is_available().await);
    }

    #[tokio::test]
    async fn test_sandbox_executor_check_permissions() {
        let executor = InMemorySandboxExecutor::new();
        let config = SandboxConfig::new(SandboxBackend::Wasm).grant(SandboxPermission::FileRead);

        // Allowed
        let result = executor.check_permissions(&[SandboxPermission::FileRead], &config);
        assert!(result.is_ok());

        // Not allowed
        let result = executor.check_permissions(&[SandboxPermission::NetworkOutbound], &config);
        assert!(matches!(result, Err(SandboxError::PermissionDenied(_))));
    }

    #[tokio::test]
    async fn test_sandbox_executor_memory_limit() {
        let executor = InMemorySandboxExecutor::new();
        let config = SandboxConfig::new(SandboxBackend::Native)
            .with_limits(ResourceLimits::default().with_max_memory(10)); // Very low

        // Large code should exceed memory
        let large_code = "x".repeat(100);
        let result = executor.execute(&large_code, &config).await;
        assert!(matches!(
            result,
            Err(SandboxError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    fn test_sandbox_execution_result_success() {
        let result = SandboxExecutionResult {
            stdout: "ok".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 5,
            memory_used_bytes: 1024,
        };
        assert!(result.success());

        let failed = SandboxExecutionResult {
            exit_code: 1,
            ..result
        };
        assert!(!failed.success());
    }

    #[test]
    fn test_sandbox_backend_variants() {
        assert_eq!(SandboxBackend::Docker, SandboxBackend::Docker);
        assert_eq!(SandboxBackend::Wasm, SandboxBackend::Wasm);
        assert_eq!(SandboxBackend::Native, SandboxBackend::Native);
        assert_eq!(SandboxBackend::None, SandboxBackend::None);
        assert_ne!(SandboxBackend::Docker, SandboxBackend::Wasm);
    }
}
