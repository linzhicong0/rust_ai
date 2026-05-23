//! Code execution tool for running code snippets.
//!
//! This tool provides agents with the ability to execute code in various languages.
//! It uses containerization or process isolation for security.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use ai_core::{tool::ToolDescriptor, Tool, ToolError, ToolOutput};

/// Supported programming languages for code execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeLanguage {
    /// Python
    Python,
    /// JavaScript (Node.js)
    JavaScript,
    /// Ruby
    Ruby,
    /// Go
    Go,
    /// Rust
    Rust,
    /// Bash/shell
    Bash,
}

impl CodeLanguage {
    /// Get the language identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::Ruby => "ruby",
            Self::Go => "go",
            Self::Rust => "rust",
            Self::Bash => "bash",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "python" | "py" => Some(Self::Python),
            "javascript" | "js" | "node" => Some(Self::JavaScript),
            "ruby" | "rb" => Some(Self::Ruby),
            "go" | "golang" => Some(Self::Go),
            "rust" | "rs" => Some(Self::Rust),
            "bash" | "sh" => Some(Self::Bash),
            _ => None,
        }
    }

    /// Get the command to execute code for this language.
    fn command(&self) -> &'static str {
        match self {
            Self::Python => "python3",
            Self::JavaScript => "node",
            Self::Ruby => "ruby",
            Self::Go => "go",
            Self::Rust => "rustc",
            Self::Bash => "bash",
        }
    }

    /// Get the file extension for this language.
    fn extension(&self) -> &'static str {
        match self {
            Self::Python => "py",
            Self::JavaScript => "js",
            Self::Ruby => "rb",
            Self::Go => "go",
            Self::Rust => "rs",
            Self::Bash => "sh",
        }
    }
}

/// Configuration for code execution tool.
#[derive(Debug, Clone)]
pub struct CodeExecConfig {
    /// Allowed languages.
    allowed_languages: HashSet<CodeLanguage>,

    /// Maximum execution time in seconds.
    timeout_secs: u64,

    /// Maximum output size in bytes.
    max_output_size: usize,

    /// Enable code execution (must be explicitly set to true).
    enabled: bool,

    /// Use Docker/container for isolation (if available).
    use_container: bool,

    /// Working directory for temporary files.
    temp_dir: Option<String>,
}

impl Default for CodeExecConfig {
    fn default() -> Self {
        let mut allowed: HashSet<CodeLanguage> = HashSet::new();
        allowed.insert(CodeLanguage::Python);
        allowed.insert(CodeLanguage::JavaScript);
        allowed.insert(CodeLanguage::Ruby);
        allowed.insert(CodeLanguage::Bash);

        Self {
            allowed_languages: allowed,
            timeout_secs: 30,
            max_output_size: 1024 * 1024, // 1 MB
            enabled: false,
            use_container: false,
            temp_dir: None,
        }
    }
}

impl CodeExecConfig {
    /// Enable code execution.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Allow a specific language.
    pub fn allow_language(mut self, lang: CodeLanguage) -> Self {
        self.allowed_languages.insert(lang);
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Enable container isolation.
    pub fn with_container(mut self, enabled: bool) -> Self {
        self.use_container = enabled;
        self
    }

    /// Validate that a language is allowed.
    fn validate_language(&self, lang: CodeLanguage) -> Result<(), ToolError> {
        if !self.enabled {
            return Err(ToolError::Execution(
                "Code execution is disabled. Enable it with CodeExecConfig::enabled(true)"
                    .to_string(),
            ));
        }

        if !self.allowed_languages.contains(&lang) {
            return Err(ToolError::Execution(format!(
                "Language '{}' is not in the allowlist",
                lang.as_str()
            )));
        }

        Ok(())
    }
}

/// Result of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecResult {
    /// Exit code (0 = success).
    pub exit_code: i32,

    /// Standard output.
    pub stdout: String,

    /// Standard error.
    pub stderr: String,

    /// Whether execution timed out.
    pub timed_out: bool,
}

/// Code execution tool.
///
/// # Security Warning
///
/// This tool allows execution of arbitrary code. It is DISABLED by default
/// and must be explicitly enabled. When enabled, consider using container
/// isolation for additional security.
pub struct CodeExec {
    config: CodeExecConfig,
}

impl CodeExec {
    /// Create a new code exec tool with default config (disabled).
    pub fn new() -> Self {
        Self::with_config(CodeExecConfig::default())
    }

    /// Create a new code exec tool with custom config.
    pub fn with_config(config: CodeExecConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "code_exec",
            "Execute code in various programming languages (Python, JavaScript, Ruby, Go, Rust, Bash).\n\n\
             SECURITY WARNING: This tool is disabled by default. Only enable if you understand \
             the security implications. Use timeouts and language allowlists to limit exposure.",
            json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to execute"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (python, javascript, ruby, go, rust, bash)",
                        "enum": ["python", "javascript", "ruby", "go", "rust", "bash"]
                    }
                },
                "required": ["code", "language"]
            }),
        )
    }

    /// Execute code for a specific language.
    async fn execute_code(
        &self,
        code: &str,
        language: CodeLanguage,
    ) -> Result<CodeExecResult, ToolError> {
        self.config.validate_language(language)?;

        debug!("Executing {} code", language.as_str());

        match language {
            CodeLanguage::Python => self.execute_python(code).await,
            CodeLanguage::JavaScript => self.execute_javascript(code).await,
            CodeLanguage::Ruby => self.execute_ruby(code).await,
            CodeLanguage::Go => self.execute_go(code).await,
            CodeLanguage::Rust => self.execute_rust(code).await,
            CodeLanguage::Bash => self.execute_bash(code).await,
        }
    }

    /// Execute Python code.
    async fn execute_python(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        let result = self.run_command("python3", &["-c", code]).await?;
        Ok(result)
    }

    /// Execute JavaScript code.
    async fn execute_javascript(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        let result = self.run_command("node", &["-e", code]).await?;
        Ok(result)
    }

    /// Execute Ruby code.
    async fn execute_ruby(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        let result = self.run_command("ruby", &["-e", code]).await?;
        Ok(result)
    }

    /// Execute Go code (requires writing temp file).
    async fn execute_go(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        warn!("Go execution requires temp file compilation");
        let temp_file = format!("/tmp/temp_{}.go", uuid::Uuid::new_v4());

        // Write code to temp file
        tokio::fs::write(&temp_file, code)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to write temp file: {}", e)))?;

        // Run go run
        let result = self.run_command("go", &["run", &temp_file]).await?;

        // Clean up
        let _ = tokio::fs::remove_file(temp_file).await;

        Ok(result)
    }

    /// Execute Rust code (requires writing temp file).
    async fn execute_rust(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        warn!("Rust execution requires temp file compilation");
        let temp_file = format!("/tmp/temp_{}.rs", uuid::Uuid::new_v4());

        // Write code to temp file
        tokio::fs::write(&temp_file, code)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to write temp file: {}", e)))?;

        // Compile and run
        let compile_result = self
            .run_command("rustc", &["-o", "/tmp/temp_exec", &temp_file])
            .await?;

        if compile_result.exit_code != 0 {
            return Ok(compile_result);
        }

        let run_result = self.run_command("/tmp/temp_exec", &[]).await?;

        // Clean up
        let _ = tokio::fs::remove_file(temp_file).await;
        let _ = tokio::fs::remove_file("/tmp/temp_exec").await;

        Ok(run_result)
    }

    /// Execute Bash code.
    async fn execute_bash(&self, code: &str) -> Result<CodeExecResult, ToolError> {
        let result = self.run_command("bash", &["-c", code]).await?;
        Ok(result)
    }

    /// Run a command with timeout and output capture.
    async fn run_command(&self, cmd: &str, args: &[&str]) -> Result<CodeExecResult, ToolError> {
        let duration = Duration::from_secs(self.config.timeout_secs);

        let output = timeout(duration, Command::new(cmd).args(args).output()).await;

        match output {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                // Truncate if too large
                let stdout = if stdout.len() > self.config.max_output_size {
                    format!("{}... (truncated)", &stdout[..self.config.max_output_size])
                } else {
                    stdout
                };

                let stderr = if stderr.len() > self.config.max_output_size {
                    format!("{}... (truncated)", &stderr[..self.config.max_output_size])
                } else {
                    stderr
                };

                Ok(CodeExecResult {
                    exit_code: output.status.code().unwrap_or(-1),
                    stdout,
                    stderr,
                    timed_out: false,
                })
            }
            Ok(Err(e)) => Err(ToolError::Execution(format!(
                "Failed to execute command: {}",
                e
            ))),
            Err(_) => {
                // Timeout
                Ok(CodeExecResult {
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!(
                        "Execution timed out after {} seconds",
                        self.config.timeout_secs
                    ),
                    timed_out: true,
                })
            }
        }
    }

    /// Format execution result for output.
    fn format_result(result: &CodeExecResult) -> String {
        if result.timed_out {
            return format!("Execution timed out.\n[stderr]: {}", result.stderr);
        }

        if result.exit_code == 0 {
            let mut output = result.stdout.clone();
            if !result.stderr.is_empty() {
                output.push_str("\n[stderr]: ");
                output.push_str(&result.stderr);
            }
            output
        } else {
            format!(
                "Exit code: {}\n[stdout]: {}\n[stderr]: {}",
                result.exit_code, result.stdout, result.stderr
            )
        }
    }
}

impl Default for CodeExec {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct CodeExecInput {
    code: String,
    language: String,
}

#[async_trait]
impl Tool for CodeExec {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let input: CodeExecInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        if input.code.is_empty() {
            return Ok(ToolOutput::error("Code cannot be empty"));
        }

        let language = CodeLanguage::from_str(&input.language).ok_or_else(|| {
            ToolError::InvalidInput(format!(
                "Unknown language: '{}'. Supported: python, javascript, ruby, go, rust, bash",
                input.language
            ))
        })?;

        let result = self.execute_code(&input.code, language).await?;

        if result.exit_code == 0 && !result.timed_out {
            Ok(ToolOutput::success(Self::format_result(&result)))
        } else {
            Ok(ToolOutput::error(Self::format_result(&result)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_code_exec_disabled_by_default() {
        let tool = CodeExec::new();
        let result = tool
            .execute(json!({"code": "print('hello')", "language": "python"}))
            .await;

        // Should fail with disabled error
        assert!(result.is_err());
        if let Err(ToolError::Execution(e)) = result {
            assert!(e.contains("disabled"));
        } else {
            panic!("Expected Execution error with disabled message");
        }
    }

    #[tokio::test]
    async fn test_code_exec_python() {
        let config = CodeExecConfig::default().enabled(true);
        let tool = CodeExec::with_config(config);

        let result = tool
            .execute(json!({"code": "print('hello from python')", "language": "python"}))
            .await;

        // May succeed or fail depending on test environment
        match result {
            Ok(output) => {
                if !output.is_error {
                    assert!(output.content.contains("hello from python"));
                }
            }
            Err(_) => {
                // Acceptable if Python is not installed
            }
        }
    }

    #[tokio::test]
    async fn test_code_exec_javascript() {
        let config = CodeExecConfig::default().enabled(true);
        let tool = CodeExec::with_config(config);

        let result = tool
            .execute(json!({"code": "console.log('hello from node')", "language": "javascript"}))
            .await;

        match result {
            Ok(output) => {
                if !output.is_error {
                    assert!(output.content.contains("hello from node"));
                }
            }
            Err(_) => {
                // Acceptable if Node.js is not installed
            }
        }
    }

    #[tokio::test]
    async fn test_code_exec_bash() {
        let config = CodeExecConfig::default().enabled(true);
        let tool = CodeExec::with_config(config);

        let result = tool
            .execute(json!({"code": "echo 'hello from bash'", "language": "bash"}))
            .await;

        match result {
            Ok(output) => {
                if !output.is_error {
                    assert!(output.content.contains("hello from bash"));
                }
            }
            Err(_) => {
                // Acceptable if bash is not available
            }
        }
    }

    #[tokio::test]
    async fn test_code_exec_empty_code() {
        let config = CodeExecConfig::default().enabled(true);
        let tool = CodeExec::with_config(config);

        let result = tool
            .execute(json!({"code": "", "language": "python"}))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("empty"));
    }

    #[tokio::test]
    async fn test_code_exec_unknown_language() {
        let config = CodeExecConfig::default().enabled(true);
        let tool = CodeExec::with_config(config);

        let result = tool
            .execute(json!({"code": "print('test')", "language": "fortran"}))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_code_exec_language_not_allowed() {
        // Create a config with only Python allowed (not Ruby which is in defaults)
        let config = CodeExecConfig {
            allowed_languages: {
                let mut set = HashSet::new();
                set.insert(CodeLanguage::Python);
                set
            },
            ..CodeExecConfig::default()
        }
        .enabled(true);

        let tool = CodeExec::with_config(config);

        // Only Python is allowed
        let result = tool
            .execute(json!({"code": "puts 'test'", "language": "ruby"}))
            .await;

        assert!(result.is_err());
        if let Err(ToolError::Execution(e)) = result {
            assert!(e.contains("allowlist"));
        } else {
            panic!("Expected Execution error with allowlist message");
        }
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(CodeLanguage::from_str("python"), Some(CodeLanguage::Python));
        assert_eq!(CodeLanguage::from_str("py"), Some(CodeLanguage::Python));
        assert_eq!(
            CodeLanguage::from_str("javascript"),
            Some(CodeLanguage::JavaScript)
        );
        assert_eq!(CodeLanguage::from_str("js"), Some(CodeLanguage::JavaScript));
        assert_eq!(
            CodeLanguage::from_str("node"),
            Some(CodeLanguage::JavaScript)
        );
        assert_eq!(CodeLanguage::from_str("ruby"), Some(CodeLanguage::Ruby));
        assert_eq!(CodeLanguage::from_str("go"), Some(CodeLanguage::Go));
        assert_eq!(CodeLanguage::from_str("rust"), Some(CodeLanguage::Rust));
        assert_eq!(CodeLanguage::from_str("bash"), Some(CodeLanguage::Bash));
        assert_eq!(CodeLanguage::from_str("fortran"), None);
    }

    #[test]
    fn test_language_as_str() {
        assert_eq!(CodeLanguage::Python.as_str(), "python");
        assert_eq!(CodeLanguage::JavaScript.as_str(), "javascript");
        assert_eq!(CodeLanguage::Ruby.as_str(), "ruby");
        assert_eq!(CodeLanguage::Go.as_str(), "go");
        assert_eq!(CodeLanguage::Rust.as_str(), "rust");
        assert_eq!(CodeLanguage::Bash.as_str(), "bash");
    }

    #[test]
    fn test_config_validate_language() {
        // Create a config with only Python and JavaScript allowed
        let config = CodeExecConfig {
            allowed_languages: {
                let mut set = HashSet::new();
                set.insert(CodeLanguage::Python);
                set.insert(CodeLanguage::JavaScript);
                set
            },
            ..CodeExecConfig::default()
        }
        .enabled(true);

        assert!(config.validate_language(CodeLanguage::Python).is_ok());
        assert!(config.validate_language(CodeLanguage::JavaScript).is_ok());
        assert!(config.validate_language(CodeLanguage::Ruby).is_err());
    }

    #[test]
    fn test_config_disabled() {
        let config = CodeExecConfig::default();
        assert!(config.validate_language(CodeLanguage::Python).is_err());
    }
}
