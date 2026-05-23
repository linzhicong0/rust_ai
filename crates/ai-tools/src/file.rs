//! File I/O tools for reading and writing files.
//!
//! These tools provide agents with the ability to read and write files
//! on the local filesystem. They are sandboxed by default with configurable
//! base paths.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, warn};

use ai_core::{tool::ToolDescriptor, Tool, ToolError, ToolOutput};

/// Configuration for file tool sandboxing.
#[derive(Debug, Clone)]
pub struct FileToolConfig {
    /// Base directory that file operations are restricted to.
    /// If None, operations are not restricted (use with caution).
    pub base_dir: Option<PathBuf>,

    /// Maximum file size for reads (in bytes).
    pub max_read_size: usize,

    /// Allow file write operations.
    pub allow_write: bool,

    /// Allow file read operations.
    pub allow_read: bool,
}

impl Default for FileToolConfig {
    fn default() -> Self {
        Self {
            base_dir: Some(PathBuf::from(".")),
            max_read_size: 10 * 1024 * 1024, // 10 MB
            allow_write: false,
            allow_read: true,
        }
    }
}

impl FileToolConfig {
    /// Create a new config with a base directory for sandboxing.
    pub fn with_base_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.base_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Enable file read operations.
    pub fn with_read_enabled(mut self, enabled: bool) -> Self {
        self.allow_read = enabled;
        self
    }

    /// Enable file write operations.
    pub fn with_write_enabled(mut self, enabled: bool) -> Self {
        self.allow_write = enabled;
        self
    }

    /// Set maximum read size in bytes.
    pub fn with_max_read_size(mut self, size: usize) -> Self {
        self.max_read_size = size;
        self
    }

    /// Validate and resolve a path within the sandbox.
    fn resolve_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        let path = PathBuf::from(path);

        // Resolve to absolute path
        let absolute = if path.is_absolute() {
            path.clone()
        } else {
            std::env::current_dir()
                .map_err(|e| ToolError::Execution(format!("Failed to get current dir: {}", e)))?
                .join(&path)
        };

        // Apply base directory restriction if configured
        if let Some(ref base) = self.base_dir {
            let base_absolute = if base.is_absolute() {
                base.clone()
            } else {
                std::env::current_dir()
                    .map_err(|e| ToolError::Execution(format!("Failed to get current dir: {}", e)))?
                    .join(base)
            };

            if !absolute.starts_with(&base_absolute) {
                return Err(ToolError::Execution(format!(
                    "Path '{}' is outside allowed base directory '{}'",
                    path.display(),
                    base.display()
                )));
            }
        }

        // Check for path traversal attempts
        if path.to_string_lossy().contains("..") {
            warn!("Path traversal attempt detected: {}", path.display());
        }

        Ok(absolute)
    }
}

/// File read tool.
pub struct FileRead {
    config: FileToolConfig,
}

impl FileRead {
    /// Create a new file read tool with default config.
    pub fn new() -> Self {
        Self::with_config(FileToolConfig::default())
    }

    /// Create a new file read tool with custom config.
    pub fn with_config(config: FileToolConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "file_read",
            "Read the contents of a file. Returns the file contents as text.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        )
    }
}

impl Default for FileRead {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct FileReadInput {
    path: String,
}

#[async_trait]
impl Tool for FileRead {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        if !self.config.allow_read {
            return Ok(ToolOutput::error("File read operations are disabled"));
        }

        let input: FileReadInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        let path = self.config.resolve_path(&input.path)?;
        debug!("Reading file: {}", path.display());

        // Check file size before reading
        let metadata = fs::metadata(&path)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read file metadata: {}", e)))?;

        if metadata.len() as usize > self.config.max_read_size {
            return Ok(ToolOutput::error(format!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.config.max_read_size
            )));
        }

        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read file: {}", e)))?;

        Ok(ToolOutput::success(content))
    }
}

/// File write tool.
pub struct FileWrite {
    config: FileToolConfig,
}

impl FileWrite {
    /// Create a new file write tool with default config (write disabled).
    pub fn new() -> Self {
        Self::with_config(FileToolConfig::default())
    }

    /// Create a new file write tool with custom config.
    pub fn with_config(config: FileToolConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "file_write",
            "Write content to a file. Creates parent directories if needed.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist",
                        "default": false
                    }
                },
                "required": ["path", "content"]
            }),
        )
    }
}

impl Default for FileWrite {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct FileWriteInput {
    path: String,
    content: String,
    #[serde(default)]
    create_dirs: bool,
}

#[async_trait]
impl Tool for FileWrite {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        if !self.config.allow_write {
            return Ok(ToolOutput::error("File write operations are disabled"));
        }

        let input: FileWriteInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        let path = self.config.resolve_path(&input.path)?;
        debug!("Writing file: {}", path.display());

        // Create parent directories if requested
        if input.create_dirs {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ToolError::Execution(format!("Failed to create directories: {}", e)))?;
            }
        }

        let content_len = input.content.len();
        fs::write(&path, input.content)
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to write file: {}", e)))?;

        Ok(ToolOutput::success(format!(
            "Successfully wrote {} bytes to {}",
            content_len,
            path.display()
        )))
    }
}

/// File list tool - list files in a directory.
pub struct FileList {
    config: FileToolConfig,
}

impl FileList {
    /// Create a new file list tool with default config.
    pub fn new() -> Self {
        Self::with_config(FileToolConfig::default())
    }

    /// Create a new file list tool with custom config.
    pub fn with_config(config: FileToolConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "file_list",
            "List files and directories in a given path.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list (default: current directory)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List files recursively",
                        "default": false
                    }
                },
                "required": []
            }),
        )
    }
}

impl Default for FileList {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct FileListInput {
    #[serde(default)]
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[async_trait]
impl Tool for FileList {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        if !self.config.allow_read {
            return Ok(ToolOutput::error("File read operations are disabled"));
        }

        let input: FileListInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        let path = if input.path.is_empty() {
            ".".to_string()
        } else {
            input.path
        };

        let resolved = self.config.resolve_path(&path)?;
        debug!("Listing directory: {}", resolved.display());

        let mut entries = Vec::new();

        if input.recursive {
            let _ = fs::read_dir(&resolved)
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to read directory: {}", e)))?;

            // Use a stack for recursive traversal
            let mut stack = vec![resolved.clone()];

            while let Some(current_dir) = stack.pop() {
                if let Ok(mut dir) = fs::read_dir(&current_dir).await {
                    while let Some(entry) = dir.next_entry().await
                        .map_err(|e| ToolError::Execution(format!("Failed to read entry: {}", e)))?
                    {
                        let entry_path = entry.path();
                        let relative = entry_path
                            .strip_prefix(&resolved)
                            .unwrap_or(&entry_path)
                            .to_string_lossy()
                            .to_string();

                        let metadata = entry.metadata().await
                            .map_err(|e| ToolError::Execution(format!("Failed to get metadata: {}", e)))?;

                        let entry_type = if metadata.is_dir() {
                            stack.push(entry_path.clone());
                            "DIR"
                        } else if metadata.is_file() {
                            "FILE"
                        } else {
                            "OTHER"
                        };

                        entries.push(format!(
                            "{} | {} | {} bytes",
                            relative,
                            entry_type,
                            metadata.len()
                        ));
                    }
                }
            }
        } else {
            let mut dir = fs::read_dir(&resolved)
                .await
                .map_err(|e| ToolError::Execution(format!("Failed to read directory: {}", e)))?;

            while let Some(entry) = dir.next_entry().await
                .map_err(|e| ToolError::Execution(format!("Failed to read entry: {}", e)))?
            {
                let name = entry.file_name().to_string_lossy().to_string();
                let metadata = entry.metadata().await
                    .map_err(|e| ToolError::Execution(format!("Failed to get metadata: {}", e)))?;

                let entry_type = if metadata.is_dir() {
                    "DIR"
                } else if metadata.is_file() {
                    "FILE"
                } else {
                    "OTHER"
                };

                entries.push(format!(
                    "{} | {} | {} bytes",
                    name,
                    entry_type,
                    metadata.len()
                ));
            }
        }

        entries.sort();

        Ok(ToolOutput::success(if entries.is_empty() {
            "(empty directory)".to_string()
        } else {
            entries.join("\n")
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_file_read() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").await.unwrap();

        let config = FileToolConfig::default()
            .with_base_dir(temp.path())
            .with_read_enabled(true);

        let tool = FileRead::with_config(config);
        // Use absolute path to avoid resolution issues
        let result = tool
            .execute(json!({"path": file_path.to_str().unwrap()}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_file_write() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("new.txt");

        let config = FileToolConfig::default()
            .with_base_dir(temp.path())
            .with_write_enabled(true)
            .with_read_enabled(true);

        let tool = FileWrite::with_config(config.clone());
        let result = tool
            .execute(json!({"path": file_path.to_str().unwrap(), "content": "Test content"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Successfully wrote"));

        // Verify file was written
        let read_tool = FileRead::with_config(config);
        let read_result = read_tool
            .execute(json!({"path": file_path.to_str().unwrap()}))
            .await
            .unwrap();

        assert_eq!(read_result.content, "Test content");
    }

    #[tokio::test]
    async fn test_file_write_disabled() {
        let config = FileToolConfig::default().with_write_enabled(false);
        let tool = FileWrite::with_config(config);

        let result = tool
            .execute(json!({"path": "test.txt", "content": "test"}))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("disabled"));
    }

    #[tokio::test]
    async fn test_file_read_outside_base_dir() {
        let temp = TempDir::new().unwrap();
        let config = FileToolConfig::default()
            .with_base_dir(temp.path());

        let tool = FileRead::with_config(config);
        let result = tool
            .execute(json!({"path": "/etc/passwd"}))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_file_list() {
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("file1.txt"), "content1").await.unwrap();
        fs::write(temp.path().join("file2.txt"), "content2").await.unwrap();
        fs::create_dir(temp.path().join("subdir")).await.unwrap();

        let config = FileToolConfig::default()
            .with_base_dir(temp.path())
            .with_read_enabled(true);

        let tool = FileList::with_config(config);
        let result = tool
            .execute(json!({"path": temp.path().to_str().unwrap()}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("file1.txt"));
        assert!(result.content.contains("file2.txt"));
        assert!(result.content.contains("subdir"));
    }

    #[tokio::test]
    async fn test_file_write_with_create_dirs() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("nested/dir/file.txt");

        let config = FileToolConfig::default()
            .with_base_dir(temp.path())
            .with_write_enabled(true)
            .with_read_enabled(true);

        let tool = FileWrite::with_config(config.clone());
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "test",
                "create_dirs": true
            }))
            .await
            .unwrap();

        assert!(!result.is_error);

        // Verify file exists
        let read_tool = FileRead::with_config(config);
        let read_result = read_tool
            .execute(json!({"path": file_path.to_str().unwrap()}))
            .await
            .unwrap();

        assert_eq!(read_result.content, "test");
    }
}
