//! Built-in tools for the AI framework.
//!
//! This crate provides a collection of commonly-used tools that agents
//! can invoke during execution. Each tool is sandboxed and disabled by default,
//! requiring explicit opt-in for security.
//!
//! ## Available Tools
//!
//! - **File I/O** ([`FileRead`], [`FileWrite`], [`FileList`]) — Read and write files
//! - **HTTP** ([`HttpFetch`], [`HttpHead`]) — Make web requests
//! - **Search** ([`WebSearch`]) — Search the web
//! - **Shell** ([`ShellExec`], [`SafeShell`]) — Execute shell commands
//! - **Code** ([`CodeExec`]) — Execute code in various languages
//!
//! ## Security
//!
//! All tools are designed with security in mind:
//! - Sandbox restrictions on file operations
//! - Allowlist/blocklist for URLs and commands
//! - Timeout and size limits
//! - Disabled by default — must be explicitly enabled
//!
//! ## Example
//!
//! ```rust
//! use ai_tools::{FileRead, SafeShell, FileToolConfig};
//! use ai_core::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//!
//! // Register file read with sandbox
//! registry.register(FileRead::with_config(
//!     FileToolConfig::default()
//!         .with_base_dir("/workspace")
//!         .with_read_enabled(true)
//! ));
//!
//! // Register safe shell (read-only commands only)
//! registry.register(SafeShell::new());
//! ```

pub mod code;
pub mod file;
pub mod http;
pub mod search;
pub mod shell;

// Re-export main tool types
pub use code::{CodeExec, CodeExecConfig, CodeExecResult, CodeLanguage};
pub use file::{FileList, FileRead, FileWrite, FileToolConfig};
pub use http::{HttpFetch, HttpHead, HttpToolConfig};
pub use search::{SearchConfig, SearchProvider, SearchResult, WebSearch};
pub use shell::{SafeShell, ShellConfig, ShellExec};
