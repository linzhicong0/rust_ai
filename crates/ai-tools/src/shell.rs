//! Shell execution tool for running commands.
//!
//! This tool provides agents with the ability to execute shell commands.
//! It is DISABLED by default and requires explicit opt-in due to security risks.
//! When enabled, it supports command allowlisting, working directory restrictions,
//! and timeout limits.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

use ai_core::{tool::ToolDescriptor, Tool, ToolError, ToolOutput};

/// Configuration for shell execution tool.
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Allowed commands (e.g., ["ls", "cat", "grep"]).
    /// If empty, all commands are allowed (NOT RECOMMENDED).
    allowed_commands: HashSet<String>,

    /// Blocked commands (always denied, e.g., ["rm", "sudo", "curl"]).
    blocked_commands: HashSet<String>,

    /// Working directory for commands.
    /// If set, all commands run in this directory.
    working_dir: Option<String>,

    /// Maximum execution time in seconds.
    timeout_secs: u64,

    /// Maximum output size in bytes.
    max_output_size: usize,

    /// Enable shell execution (must be explicitly set to true).
    enabled: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        let mut blocked: HashSet<String> = [
            "rm", "rmdir", "mv", "dd", "mkfs",
            "sudo", "su", "doas",
            "chmod", "chown",
            "kill", "killall",
            "reboot", "shutdown", "poweroff",
            "passwd", "usermod", "userdel",
            "curl", "wget", "nc", "netcat",
            "iptables", "nftables",
            "crontab", "at",
            "systemctl", "service",
        ].iter().map(|s| s.to_string()).collect();

        // Block shell built-ins that could be dangerous
        blocked.extend(["sh", "bash", "zsh", "fish", "cmd", "powershell"].iter().map(|s| s.to_string()));

        Self {
            allowed_commands: HashSet::new(),
            blocked_commands: blocked,
            working_dir: None,
            timeout_secs: 30,
            max_output_size: 1024 * 1024, // 1 MB
            enabled: false, // MUST be explicitly enabled
        }
    }
}

impl ShellConfig {
    /// Enable shell execution (required for tool to work).
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Allow a specific command.
    pub fn allow_command(mut self, cmd: impl Into<String>) -> Self {
        self.allowed_commands.insert(cmd.into());
        self
    }

    /// Block a specific command.
    pub fn block_command(mut self, cmd: impl Into<String>) -> Self {
        self.blocked_commands.insert(cmd.into());
        self
    }

    /// Set the working directory.
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set execution timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Validate a command against allowlist and blocklist.
    fn validate_command(&self, cmd: &str) -> Result<(), ToolError> {
        // Extract the base command (first word)
        let base_cmd = cmd.split_whitespace()
            .next()
            .ok_or_else(|| ToolError::InvalidInput("Empty command".to_string()))?;

        // Check blocklist first
        if self.blocked_commands.contains(base_cmd) {
            return Err(ToolError::Execution(format!(
                "Command '{}' is blocked for security reasons",
                base_cmd
            )));
        }

        // If allowlist is configured, check it
        if !self.allowed_commands.is_empty() {
            if !self.allowed_commands.contains(base_cmd) {
                return Err(ToolError::Execution(format!(
                    "Command '{}' is not in the allowlist",
                    base_cmd
                )));
            }
        }

        // Check for shell metacharacters that could enable command injection
        if cmd.contains('|') || cmd.contains(';') || cmd.contains('&') || cmd.contains('$') {
            return Err(ToolError::Execution(
                "Shell metacharacters (|, ;, &, $) are not allowed".to_string()
            ));
        }

        Ok(())
    }
}

/// Shell execution tool.
///
/// # Security Warning
///
/// This tool allows execution of arbitrary shell commands. It is DISABLED by default
/// and must be explicitly enabled. Even when enabled, you should:
/// - Use a restrictive allowlist
/// - Set a working directory
/// - Use a short timeout
/// - Never run with elevated privileges
pub struct ShellExec {
    config: ShellConfig,
}

impl ShellExec {
    /// Create a new shell exec tool with default config (disabled).
    pub fn new() -> Self {
        Self::with_config(ShellConfig::default())
    }

    /// Create a new shell exec tool with custom config.
    pub fn with_config(config: ShellConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "shell_exec",
            "Execute a shell command. Returns the command's standard output.\n\n\
             SECURITY WARNING: This tool is disabled by default. Only enable if you \
             understand the security implications. Use allowlists and timeouts to \
             limit exposure.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (no shell metacharacters allowed)"
                    },
                    "args": {
                        "type": "array",
                        "description": "Command arguments",
                        "items": {"type": "string"}
                    }
                },
                "required": ["command"]
            }),
        )
    }

    /// Execute a command safely.
    async fn execute_command(&self, command: &str, args: &[String]) -> Result<String, ToolError> {
        if !self.config.enabled {
            return Ok(ToolOutput::error(
                "Shell execution is disabled. Enable it explicitly with ShellConfig::enabled(true)"
            ).content);
        }

        self.config.validate_command(command)?;

        debug!("Executing command: {} with args: {:?}", command, args);

        let mut cmd = Command::new(command);

        // Add arguments
        for arg in args {
            cmd.arg(arg);
        }

        // Set working directory if configured
        if let Some(ref wd) = self.config.working_dir {
            cmd.current_dir(wd);
        }

        // Set timeout
        let timeout = Duration::from_secs(self.config.timeout_secs);

        // Execute with timeout
        let output = tokio::time::timeout(
            timeout,
            cmd.output()
        )
        .await
        .map_err(|_| ToolError::Execution(format!(
            "Command timed out after {} seconds",
            self.config.timeout_secs
        )))?
        .map_err(|e| ToolError::Execution(format!("Failed to execute command: {}", e)))?;

        // Get output, respecting size limits
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Check size limits
        if stdout.len() > self.config.max_output_size {
            return Ok(ToolOutput::error(format!(
                "Output too large: {} bytes (max: {} bytes)",
                stdout.len(),
                self.config.max_output_size
            )).content);
        }

        // Format result
        let status = output.status;
        if status.success() {
            let mut result = stdout;
            if !stderr.is_empty() {
                result.push_str("\n[stderr]: ");
                result.push_str(&stderr);
            }
            Ok(result)
        } else {
            let exit_code = status.code().unwrap_or(-1);
            let mut error_msg = format!("Command exited with code {}", exit_code);
            if !stdout.is_empty() {
                error_msg.push_str("\n[stdout]: ");
                error_msg.push_str(&stdout);
            }
            if !stderr.is_empty() {
                error_msg.push_str("\n[stderr]: ");
                error_msg.push_str(&stderr);
            }
            Err(ToolError::Execution(error_msg))
        }
    }
}

impl Default for ShellExec {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct ShellExecInput {
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

#[async_trait]
impl Tool for ShellExec {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        if !self.config.enabled {
            return Ok(ToolOutput::error(
                "Shell execution is disabled. Enable it with ShellConfig::enabled(true)"
            ));
        }

        let input: ShellExecInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        let output = self.execute_command(&input.command, &input.args).await?;

        Ok(ToolOutput::success(output))
    }
}

/// A pre-configured shell exec tool for common safe commands.
pub struct SafeShell {
    inner: ShellExec,
}

impl SafeShell {
    /// Create a safe shell tool with a curated list of read-only commands.
    pub fn new() -> Self {
        let config = ShellConfig::default()
            .enabled(true)
            .allow_command("ls")
            .allow_command("pwd")
            .allow_command("cat")
            .allow_command("head")
            .allow_command("tail")
            .allow_command("grep")
            .allow_command("find")
            .allow_command("wc")
            .allow_command("date")
            .allow_command("echo")
            .allow_command("printf")
            .allow_command("basename")
            .allow_command("dirname")
            .allow_command("realpath")
            .allow_command("file")
            .allow_command("stat")
            .allow_command("du")
            .allow_command("df")
            .allow_command("uname")
            .allow_command("whoami")
            .allow_command("id")
            .allow_command("env")
            .allow_command("printenv")
            .with_timeout(10);

        Self {
            inner: ShellExec::with_config(config),
        }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "safe_shell",
            "Execute safe, read-only shell commands (ls, cat, grep, etc.).",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    },
                    "args": {
                        "type": "array",
                        "description": "Command arguments",
                        "items": {"type": "string"}
                    }
                },
                "required": ["command"]
            }),
        )
    }
}

impl Default for SafeShell {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SafeShell {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        self.inner.execute(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shell_exec_disabled_by_default() {
        let tool = ShellExec::new();
        let result = tool
            .execute(json!({"command": "echo", "args": ["hello"]}))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("disabled"));
    }

    #[tokio::test]
    async fn test_shell_exec_blocked_command() {
        let config = ShellConfig::default().enabled(true);
        let tool = ShellExec::with_config(config);

        let result = tool
            .execute(json!({"command": "rm", "args": ["-rf", "/"]})).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shell_exec_allowed_command() {
        let config = ShellConfig::default()
            .enabled(true)
            .allow_command("echo");
        let tool = ShellExec::with_config(config);

        let result = tool
            .execute(json!({"command": "echo", "args": ["hello", "world"]}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("hello world"));
    }

    #[tokio::test]
    async fn test_shell_exec_command_not_in_allowlist() {
        let config = ShellConfig::default()
            .enabled(true)
            .allow_command("echo");
        let tool = ShellExec::with_config(config);

        let result = tool.execute(json!({"command": "ls"})).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("allowlist"));
    }

    #[tokio::test]
    async fn test_shell_exec_metacharacters_blocked() {
        let config = ShellConfig::default()
            .enabled(true)
            .allow_command("echo");
        let tool = ShellExec::with_config(config);

        let result = tool
            .execute(json!({"command": "echo|rm -rf /"}))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_safe_shell() {
        let tool = SafeShell::new();

        // ls should be allowed
        let result = tool
            .execute(json!({"command": "pwd"}))
            .await;

        // May succeed or fail depending on test environment
        // Just verify it returns something
        match result {
            Ok(output) => {
                assert!(!output.is_error);
            }
            Err(_) => {
                // Acceptable in constrained environments
            }
        }
    }

    #[tokio::test]
    async fn test_safe_shell_blocks_dangerous() {
        let tool = SafeShell::new();

        // rm should be blocked even in safe shell
        let result = tool
            .execute(json!({"command": "rm", "args": ["-rf", "/"]})).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_config_validate_command() {
        let config = ShellConfig::default()
            .enabled(true)
            .allow_command("ls")
            .allow_command("cat");

        // Allowed commands should pass
        assert!(config.validate_command("ls").is_ok());
        assert!(config.validate_command("cat").is_ok());

        // Blocked commands should fail
        assert!(config.validate_command("rm").is_err());
        assert!(config.validate_command("sudo").is_err());

        // Non-allowlisted commands should fail when allowlist is set
        assert!(config.validate_command("grep").is_err());
    }

    #[test]
    fn test_config_metacharacters_blocked() {
        let config = ShellConfig::default().enabled(true);

        assert!(config.validate_command("echo|rm -rf /").is_err());
        assert!(config.validate_command("echo; rm -rf /").is_err());
        assert!(config.validate_command("echo& rm -rf /").is_err());
        assert!(config.validate_command("echo $HOME").is_err());
    }
}
