// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Model Context Protocol Support (REQ-17.3)
//!
//! Provides MCP client and server traits for standardized tool and resource integration.
//! Supports stdio and HTTP/SSE transport.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors that can occur during MCP operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Transport error: {0}")]
    Transport(String),
    #[error("Protocol error: {0}")]
    Protocol(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Connection closed")]
    ConnectionClosed,
}

/// MCP transport type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpTransport {
    /// Standard input/output transport.
    Stdio,
    /// HTTP with Server-Sent Events.
    HttpSse,
}

/// A tool descriptor in the MCP protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// A resource descriptor in the MCP protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    /// Resource URI.
    pub uri: String,
    /// Human-readable name.
    pub name: String,
    /// MIME type.
    pub mime_type: Option<String>,
    /// Resource description.
    pub description: Option<String>,
}

/// Result of calling an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    /// The result content.
    pub content: String,
    /// Whether this is an error result.
    pub is_error: bool,
    /// Optional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of reading an MCP resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    pub mime_type: String,
    /// Text content (for text resources).
    pub text: Option<String>,
    /// Binary content as base64 (for binary resources).
    pub blob: Option<String>,
}

/// MCP server capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    /// Whether the server supports tool listing and calling.
    pub tools: bool,
    /// Whether the server supports resource listing and reading.
    pub resources: bool,
    /// Whether the server supports prompts.
    pub prompts: bool,
}

/// MCP client for discovering and calling external tools via MCP protocol.
#[async_trait]
pub trait McpClient: Send + Sync {
    /// Connect to an MCP server.
    async fn connect(&mut self, transport: McpTransport)
        -> Result<McpServerCapabilities, McpError>;

    /// List available tools on the connected server.
    async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError>;

    /// Call a tool on the connected server.
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError>;

    /// List available resources on the connected server.
    async fn list_resources(&self) -> Result<Vec<McpResource>, McpError>;

    /// Read a resource from the connected server.
    async fn read_resource(&self, uri: &str) -> Result<McpResourceContent, McpError>;

    /// Disconnect from the server.
    async fn disconnect(&mut self) -> Result<(), McpError>;
}

/// MCP server for exposing framework tools via MCP protocol.
#[async_trait]
pub trait McpServer: Send + Sync {
    /// Get server capabilities.
    fn capabilities(&self) -> McpServerCapabilities;

    /// Handle a list_tools request.
    async fn handle_list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError>;

    /// Handle a call_tool request.
    async fn handle_call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError>;

    /// Handle a list_resources request.
    async fn handle_list_resources(&self) -> Result<Vec<McpResource>, McpError>;

    /// Handle a read_resource request.
    async fn handle_read_resource(&self, uri: &str) -> Result<McpResourceContent, McpError>;
}

/// In-memory MCP client for testing purposes.
pub struct InMemoryMcpClient {
    connected: bool,
    tools: Vec<McpToolDescriptor>,
    resources: Vec<McpResource>,
    tool_results: HashMap<String, McpToolResult>,
    resource_contents: HashMap<String, McpResourceContent>,
}

impl InMemoryMcpClient {
    /// Create a new in-memory MCP client.
    pub fn new() -> Self {
        Self {
            connected: false,
            tools: Vec::new(),
            resources: Vec::new(),
            tool_results: HashMap::new(),
            resource_contents: HashMap::new(),
        }
    }

    /// Register a tool for testing.
    pub fn register_tool(&mut self, descriptor: McpToolDescriptor, result: McpToolResult) {
        self.tool_results.insert(descriptor.name.clone(), result);
        self.tools.push(descriptor);
    }

    /// Register a resource for testing.
    pub fn register_resource(&mut self, resource: McpResource, content: McpResourceContent) {
        self.resource_contents.insert(resource.uri.clone(), content);
        self.resources.push(resource);
    }
}

impl Default for InMemoryMcpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl McpClient for InMemoryMcpClient {
    async fn connect(
        &mut self,
        _transport: McpTransport,
    ) -> Result<McpServerCapabilities, McpError> {
        self.connected = true;
        Ok(McpServerCapabilities {
            tools: !self.tools.is_empty(),
            resources: !self.resources.is_empty(),
            prompts: false,
        })
    }

    async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError> {
        if !self.connected {
            return Err(McpError::ConnectionClosed);
        }
        Ok(self.tools.clone())
    }

    async fn call_tool(
        &self,
        name: &str,
        _arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        if !self.connected {
            return Err(McpError::ConnectionClosed);
        }
        self.tool_results
            .get(name)
            .cloned()
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))
    }

    async fn list_resources(&self) -> Result<Vec<McpResource>, McpError> {
        if !self.connected {
            return Err(McpError::ConnectionClosed);
        }
        Ok(self.resources.clone())
    }

    async fn read_resource(&self, uri: &str) -> Result<McpResourceContent, McpError> {
        if !self.connected {
            return Err(McpError::ConnectionClosed);
        }
        self.resource_contents
            .get(uri)
            .cloned()
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))
    }

    async fn disconnect(&mut self) -> Result<(), McpError> {
        self.connected = false;
        Ok(())
    }
}

/// In-memory MCP server for testing purposes.
pub struct InMemoryMcpServer {
    tools: Vec<McpToolDescriptor>,
    resources: Vec<McpResource>,
    tool_handlers: HashMap<String, McpToolResult>,
    resource_contents: HashMap<String, McpResourceContent>,
}

impl InMemoryMcpServer {
    /// Create a new in-memory MCP server.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            resources: Vec::new(),
            tool_handlers: HashMap::new(),
            resource_contents: HashMap::new(),
        }
    }

    /// Add a tool to the server.
    pub fn add_tool(&mut self, descriptor: McpToolDescriptor, result: McpToolResult) {
        self.tool_handlers.insert(descriptor.name.clone(), result);
        self.tools.push(descriptor);
    }

    /// Add a resource to the server.
    pub fn add_resource(&mut self, resource: McpResource, content: McpResourceContent) {
        self.resource_contents.insert(resource.uri.clone(), content);
        self.resources.push(resource);
    }
}

impl Default for InMemoryMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl McpServer for InMemoryMcpServer {
    fn capabilities(&self) -> McpServerCapabilities {
        McpServerCapabilities {
            tools: !self.tools.is_empty(),
            resources: !self.resources.is_empty(),
            prompts: false,
        }
    }

    async fn handle_list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError> {
        Ok(self.tools.clone())
    }

    async fn handle_call_tool(
        &self,
        name: &str,
        _arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        self.tool_handlers
            .get(name)
            .cloned()
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))
    }

    async fn handle_list_resources(&self) -> Result<Vec<McpResource>, McpError> {
        Ok(self.resources.clone())
    }

    async fn handle_read_resource(&self, uri: &str) -> Result<McpResourceContent, McpError> {
        self.resource_contents
            .get(uri)
            .cloned()
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_client_connect_and_list_tools() {
        let mut client = InMemoryMcpClient::new();
        client.register_tool(
            McpToolDescriptor {
                name: "calculator".to_string(),
                description: "Performs arithmetic".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            },
            McpToolResult {
                content: "42".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        );

        let caps = client.connect(McpTransport::Stdio).await.unwrap();
        assert!(caps.tools);

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "calculator");
    }

    #[tokio::test]
    async fn test_mcp_client_call_tool() {
        let mut client = InMemoryMcpClient::new();
        client.register_tool(
            McpToolDescriptor {
                name: "echo".to_string(),
                description: "Echoes input".to_string(),
                input_schema: serde_json::json!({}),
            },
            McpToolResult {
                content: "hello back".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        );

        client.connect(McpTransport::HttpSse).await.unwrap();
        let result = client
            .call_tool("echo", serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        assert_eq!(result.content, "hello back");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_mcp_client_tool_not_found() {
        let mut client = InMemoryMcpClient::new();
        client.connect(McpTransport::Stdio).await.unwrap();

        let result = client.call_tool("nonexistent", serde_json::json!({})).await;
        assert!(matches!(result, Err(McpError::ToolNotFound(_))));
    }

    #[tokio::test]
    async fn test_mcp_client_not_connected() {
        let client = InMemoryMcpClient::new();
        let result = client.list_tools().await;
        assert!(matches!(result, Err(McpError::ConnectionClosed)));
    }

    #[tokio::test]
    async fn test_mcp_client_resources() {
        let mut client = InMemoryMcpClient::new();
        client.register_resource(
            McpResource {
                uri: "file:///config.json".to_string(),
                name: "Config".to_string(),
                mime_type: Some("application/json".to_string()),
                description: Some("App config".to_string()),
            },
            McpResourceContent {
                uri: "file:///config.json".to_string(),
                mime_type: "application/json".to_string(),
                text: Some(r#"{"key": "value"}"#.to_string()),
                blob: None,
            },
        );

        client.connect(McpTransport::Stdio).await.unwrap();

        let resources = client.list_resources().await.unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "file:///config.json");

        let content = client.read_resource("file:///config.json").await.unwrap();
        assert_eq!(content.text.unwrap(), r#"{"key": "value"}"#);
    }

    #[tokio::test]
    async fn test_mcp_client_resource_not_found() {
        let mut client = InMemoryMcpClient::new();
        client.connect(McpTransport::Stdio).await.unwrap();

        let result = client.read_resource("file:///missing").await;
        assert!(matches!(result, Err(McpError::ResourceNotFound(_))));
    }

    #[tokio::test]
    async fn test_mcp_client_disconnect() {
        let mut client = InMemoryMcpClient::new();
        client.connect(McpTransport::Stdio).await.unwrap();
        client.disconnect().await.unwrap();

        let result = client.list_tools().await;
        assert!(matches!(result, Err(McpError::ConnectionClosed)));
    }

    #[tokio::test]
    async fn test_mcp_server_capabilities() {
        let mut server = InMemoryMcpServer::new();
        let caps = server.capabilities();
        assert!(!caps.tools);
        assert!(!caps.resources);

        server.add_tool(
            McpToolDescriptor {
                name: "test".to_string(),
                description: "test tool".to_string(),
                input_schema: serde_json::json!({}),
            },
            McpToolResult {
                content: "ok".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        );

        let caps = server.capabilities();
        assert!(caps.tools);
        assert!(!caps.resources);
    }

    #[tokio::test]
    async fn test_mcp_server_handle_list_tools() {
        let mut server = InMemoryMcpServer::new();
        server.add_tool(
            McpToolDescriptor {
                name: "search".to_string(),
                description: "Web search".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            },
            McpToolResult {
                content: "results".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        );

        let tools = server.handle_list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
    }

    #[tokio::test]
    async fn test_mcp_server_handle_call_tool() {
        let mut server = InMemoryMcpServer::new();
        server.add_tool(
            McpToolDescriptor {
                name: "calc".to_string(),
                description: "Calculator".to_string(),
                input_schema: serde_json::json!({}),
            },
            McpToolResult {
                content: "7".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        );

        let result = server
            .handle_call_tool("calc", serde_json::json!({"a": 3, "b": 4}))
            .await
            .unwrap();
        assert_eq!(result.content, "7");
    }

    #[tokio::test]
    async fn test_mcp_server_handle_resources() {
        let mut server = InMemoryMcpServer::new();
        server.add_resource(
            McpResource {
                uri: "db://users".to_string(),
                name: "Users".to_string(),
                mime_type: Some("application/json".to_string()),
                description: None,
            },
            McpResourceContent {
                uri: "db://users".to_string(),
                mime_type: "application/json".to_string(),
                text: Some(r#"[{"id":1}]"#.to_string()),
                blob: None,
            },
        );

        let resources = server.handle_list_resources().await.unwrap();
        assert_eq!(resources.len(), 1);

        let content = server.handle_read_resource("db://users").await.unwrap();
        assert_eq!(content.text.unwrap(), r#"[{"id":1}]"#);
    }

    #[tokio::test]
    async fn test_mcp_transport_variants() {
        assert_eq!(McpTransport::Stdio, McpTransport::Stdio);
        assert_eq!(McpTransport::HttpSse, McpTransport::HttpSse);
        assert_ne!(McpTransport::Stdio, McpTransport::HttpSse);
    }
}
