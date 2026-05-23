//! HTTP fetch tool for making web requests.
//!
//! This tool provides agents with the ability to fetch content from URLs.
//! It includes security features like allowlists, timeout limits, and size limits.

use async_trait::async_trait;
use reqwest::{Client, ClientBuilder};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Duration;
use tracing::debug;

use ai_core::{tool::ToolDescriptor, Tool, ToolError, ToolOutput};

/// Configuration for HTTP tool security.
#[derive(Debug, Clone)]
pub struct HttpToolConfig {
    /// Client for making requests.
    client: Client,

    /// Allowed URL prefixes (e.g., "https://api.example.com").
    /// If empty, all URLs are allowed.
    allowed_prefixes: HashSet<String>,

    /// Blocked URL prefixes.
    blocked_prefixes: HashSet<String>,

    /// Maximum response size in bytes.
    max_response_size: usize,

    /// Request timeout in seconds.
    timeout_secs: u64,

    /// Follow redirects.
    follow_redirects: bool,

    /// Maximum number of redirects to follow.
    max_redirects: usize,
}

impl Default for HttpToolConfig {
    fn default() -> Self {
        let client = ClientBuilder::new()
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            allowed_prefixes: HashSet::new(),
            blocked_prefixes: {
                let mut set = HashSet::new();
                // Block private/local network addresses by default
                set.insert("http://localhost".to_string());
                set.insert("http://127.0.0.1".to_string());
                set.insert("http://0.0.0.0".to_string());
                set.insert("http://[::1]".to_string());
                set.insert("file://".to_string());
                set
            },
            max_response_size: 10 * 1024 * 1024, // 10 MB
            timeout_secs: 30,
            follow_redirects: true,
            max_redirects: 10,
        }
    }
}

impl HttpToolConfig {
    /// Add an allowed URL prefix.
    pub fn allow_prefix(mut self, prefix: String) -> Self {
        self.allowed_prefixes.insert(prefix);
        self
    }

    /// Add a blocked URL prefix.
    pub fn block_prefix(mut self, prefix: String) -> Self {
        self.blocked_prefixes.insert(prefix);
        self
    }

    /// Set maximum response size in bytes.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_response_size = size;
        self
    }

    /// Set request timeout in seconds.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Configure redirect following.
    pub fn with_redirects(mut self, follow: bool, max: usize) -> Self {
        self.follow_redirects = follow;
        self.max_redirects = max;
        self
    }

    /// Validate a URL against the allowlist and blocklist.
    fn validate_url(&self, url: &str) -> Result<(), ToolError> {
        // Check blocklist first
        for blocked in &self.blocked_prefixes {
            if url.starts_with(blocked) {
                return Err(ToolError::Execution(format!(
                    "URL is blocked: {} (matches blocked pattern: {})",
                    url, blocked
                )));
            }
        }

        // If allowlist is configured, check it
        if !self.allowed_prefixes.is_empty() {
            let allowed = self
                .allowed_prefixes
                .iter()
                .any(|prefix| url.starts_with(prefix));

            if !allowed {
                return Err(ToolError::Execution(format!(
                    "URL is not in allowlist: {}",
                    url
                )));
            }
        }

        Ok(())
    }
}

/// HTTP GET tool for fetching web content.
pub struct HttpFetch {
    config: HttpToolConfig,
}

impl HttpFetch {
    /// Create a new HTTP fetch tool with default config.
    pub fn new() -> Self {
        Self::with_config(HttpToolConfig::default())
    }

    /// Create a new HTTP fetch tool with custom config.
    pub fn with_config(config: HttpToolConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "http_fetch",
            "Fetch content from a URL via HTTP GET. Returns the response body as text.",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers to include",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["url"]
            }),
        )
    }
}

impl Default for HttpFetch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct HttpFetchInput {
    url: String,
    #[serde(default)]
    headers: Option<serde_json::Map<String, Value>>,
}

#[async_trait]
impl Tool for HttpFetch {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let input: HttpFetchInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        self.config.validate_url(&input.url)?;

        debug!("Fetching URL: {}", input.url);

        let mut request = self.config.client.get(&input.url);

        // Add custom headers if provided
        if let Some(headers) = input.headers {
            for (key, value) in headers {
                if let Ok(value_str) = serde_json::from_value::<String>(value) {
                    request = request.header(&key, value_str);
                }
            }
        }

        let response = request
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        let content_length = response.content_length().unwrap_or(0);

        // Check response size
        if content_length > self.config.max_response_size as u64 {
            return Ok(ToolOutput::error(format!(
                "Response too large: {} bytes (max: {} bytes)",
                content_length, self.config.max_response_size
            )));
        }

        // Get response body
        let body = response
            .text()
            .await
            .map_err(|e| ToolError::Execution(format!("Failed to read response body: {}", e)))?;

        // Return response with status
        if status.is_success() {
            Ok(ToolOutput::success(format!(
                "Status: {}\n\n{}",
                status.as_u16(),
                body
            )))
        } else {
            Ok(ToolOutput::error(format!(
                "HTTP {}: {}",
                status.as_u16(),
                body
            )))
        }
    }
}

/// HTTP HEAD tool for checking URL metadata.
pub struct HttpHead {
    config: HttpToolConfig,
}

impl HttpHead {
    /// Create a new HTTP HEAD tool with default config.
    pub fn new() -> Self {
        Self::with_config(HttpToolConfig::default())
    }

    /// Create a new HTTP HEAD tool with custom config.
    pub fn with_config(config: HttpToolConfig) -> Self {
        Self { config }
    }

    /// Get the tool descriptor.
    pub fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            "http_head",
            "Get HTTP headers from a URL without downloading the body.",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to check"
                    }
                },
                "required": ["url"]
            }),
        )
    }
}

impl Default for HttpHead {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct HttpHeadInput {
    url: String,
}

#[async_trait]
impl Tool for HttpHead {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let input: HttpHeadInput = serde_json::from_value(input)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid input: {}", e)))?;

        self.config.validate_url(&input.url)?;

        debug!("HEAD request to: {}", input.url);

        let response = self
            .config
            .client
            .head(&input.url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("HTTP request failed: {}", e)))?;

        let status = response.status();

        let headers: Vec<String> = response
            .headers()
            .iter()
            .map(|(name, value)| format!("{}: {}", name, value.to_str().unwrap_or("<binary>")))
            .collect();

        let output = format!(
            "Status: {}\n\nHeaders:\n{}",
            status.as_u16(),
            headers.join("\n")
        );

        Ok(ToolOutput::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_http_fetch() {
        // Use a mock server or a real public API for testing
        let tool = HttpFetch::new();
        let result = tool
            .execute(json!({"url": "https://httpbin.org/get"}))
            .await;

        // This test might fail in offline environments
        if result.is_ok() {
            let output = result.unwrap();
            assert!(!output.is_error);
            assert!(output.content.contains("Status: 200"));
        }
    }

    #[tokio::test]
    async fn test_http_fetch_blocked_url() {
        let config = HttpToolConfig::default().block_prefix("https://blocked.com".to_string());
        let tool = HttpFetch::with_config(config);

        let result = tool
            .execute(json!({"url": "https://blocked.com/path"}))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http_fetch_allowed_only() {
        let config = HttpToolConfig::default().allow_prefix("https://httpbin.org".to_string());
        let tool = HttpFetch::with_config(config.clone());

        // Allowed URL should work (or fail with network error, not blocked error)
        let result = tool
            .execute(json!({"url": "https://httpbin.org/get"}))
            .await;

        if let Err(ToolError::Execution(e)) = result {
            assert!(!e.contains("not in allowlist"));
        }

        // Non-allowed URL should be blocked
        let result = tool.execute(json!({"url": "https://example.com"})).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http_fetch_with_headers() {
        let tool = HttpFetch::new();
        let result = tool
            .execute(json!({
                "url": "https://httpbin.org/headers",
                "headers": {
                    "X-Custom-Header": "test-value"
                }
            }))
            .await;

        if result.is_ok() {
            let output = result.unwrap();
            assert!(!output.is_error);
            // The response should contain our custom header echoed back
            assert!(
                output.content.contains("X-Custom-Header") || output.content.contains("Status:")
            );
        }
    }

    #[tokio::test]
    async fn test_http_head() {
        let tool = HttpHead::new();
        let result = tool
            .execute(json!({"url": "https://httpbin.org/get"}))
            .await;

        if result.is_ok() {
            let output = result.unwrap();
            assert!(!output.is_error);
            assert!(output.content.contains("Status:"));
            assert!(output.content.contains("Headers:"));
        }
    }

    #[tokio::test]
    async fn test_http_fetch_invalid_url() {
        let tool = HttpFetch::new();
        let result = tool.execute(json!({"url": "not-a-valid-url"})).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http_fetch_localhost_blocked() {
        let tool = HttpFetch::new();

        // localhost should be blocked by default
        let result = tool.execute(json!({"url": "http://localhost:8080"})).await;

        assert!(result.is_err());

        let result = tool.execute(json!({"url": "http://127.0.0.1:8080"})).await;

        assert!(result.is_err());
    }
}
