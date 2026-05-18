//! Tool execution and definition for agent function calling.
//!
//! Tools are functions that agents can call during execution. This module
//! defines the [`Tool`] trait and related types for declaring and executing
//! tools with schema validation.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::{Tool, ToolDescriptor, ToolOutput};
//! use serde_json::json;
//!
//! struct WeatherTool;
//!
//! #[async_trait::async_trait]
//! impl Tool for WeatherTool {
//!     fn descriptor(&self) -> ToolDescriptor {
//!         ToolDescriptor {
//!             name: "get_weather".to_string(),
//!             description: "Get current weather for a location".to_string(),
//!             input_schema: json!({
//!                 "type": "object",
//!                 "properties": {
//!                     "location": {
//!                         "type": "string",
//!                         "description": "City name, e.g. San Francisco"
//!                     }
//!                 },
//!                 "required": ["location"]
//!             }),
//!         }
//!     }
//!
//!     async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
//!         let location = input["location"]
//!             .as_str()
//!             .ok_or_else(|| ToolError::InvalidInput("location required".to_string()))?;
//!
//!         // Call weather API...
//!         Ok(ToolOutput {
//!             content: format!("Weather in {location}: 72°F, sunny"),
//!             is_error: false,
//!         })
//!     }
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ToolError;

/// Describes a tool for function calling.
///
/// The descriptor includes the tool's name, description, and JSON Schema
/// for input validation. LLMs use this information to decide when and how
/// to call the tool.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    /// Unique identifier for this tool.
    ///
    /// Should be snake_case and descriptive, e.g., `web_search`, `file_read`.
    pub name: String,

    /// Human-readable description of what the tool does.
    ///
    /// This is shown to the LLM to help it understand when to use the tool.
    /// Best practice: describe inputs and outputs clearly.
    pub description: String,

    /// JSON Schema for the tool's input.
    ///
    /// The LLM uses this to generate valid input. Should validate all
    /// required parameters and include descriptions for each field.
    pub input_schema: Value,

    /// Optional JSON Schema for the tool's output.
    ///
    /// Used for structured output validation.
    pub output_schema: Option<Value>,
}

impl ToolDescriptor {
    /// Create a new tool descriptor.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
        }
    }

    /// Set the output schema for structured output validation.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

/// Output from a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The result content to return to the agent.
    pub content: String,

    /// Whether this output represents an error.
    ///
    /// If true, the agent may retry or attempt a different approach.
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful output.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error output.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }

    /// Create JSON output from a serializable value.
    pub fn json<T: Serialize>(value: &T) -> Result<Self, ToolError> {
        serde_json::to_string(value)
            .map(|content| Self {
                content,
                is_error: false,
            })
            .map_err(ToolError::from)
    }
}

/// A tool that an agent can call.
///
/// Tools are units of functionality that agents can invoke during reasoning.
/// They must declare their schema for validation and implement async execution.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns a descriptor for this tool.
    ///
    /// The descriptor is used to register the tool with agents and to
    /// generate the function calling schema sent to LLMs.
    fn descriptor(&self) -> ToolDescriptor;

    /// Execute the tool with the given input.
    ///
    /// # Arguments
    ///
    /// * `input` — Parsed JSON input validated against the schema
    ///
    /// # Returns
    ///
    /// A [`ToolOutput`] containing the result or error information.
    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError>;
}

/// A synchronous tool that executes on the calling thread.
///
/// For tools that don't need async I/O, implementing `SyncTool` may be
/// more convenient than `Tool`.
pub trait SyncTool: Send + Sync {
    /// Returns a descriptor for this tool.
    fn descriptor(&self) -> ToolDescriptor;

    /// Execute the tool synchronously.
    fn execute_sync(&self, input: Value) -> Result<ToolOutput, ToolError>;
}

/// Adapter to convert a `SyncTool` into a `Tool`.
///
/// Use this to wrap synchronous tools for use with async agents.
pub struct SyncToolAdapter<T: SyncTool>(pub T);

#[async_trait::async_trait]
impl<T: SyncTool> Tool for SyncToolAdapter<T> {
    fn descriptor(&self) -> ToolDescriptor {
        self.0.descriptor()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // Run the sync tool on the current thread
        self.0.execute_sync(input)
    }
}

/// Helper function to create a simple tool descriptor.
///
/// # Example
///
/// ```rust
/// use ai_core::tool::simple_descriptor;
/// use serde_json::json;
///
/// let descriptor = simple_descriptor(
///     "my_tool",
///     "Does something",
///     json!({"type": "object"}),
/// );
/// ```
pub fn simple_descriptor(
    name: impl Into<String>,
    description: impl Into<String>,
    input_schema: Value,
) -> ToolDescriptor {
    ToolDescriptor::new(name, description, input_schema)
}

/// A function-based tool created from a closure.
///
/// # Example
///
/// ```rust
/// use ai_core::tool::FnTool;
/// use serde_json::json;
///
/// let tool = FnTool::new(
///     "echo",
///     "Echoes the input",
///     json!({"type": "object"}),
///     |input| async move {
///         Ok(ToolOutput::success(format!("Echo: {}", input)))
///     },
/// );
/// ```
pub struct FnTool<F> {
    descriptor: ToolDescriptor,
    execute_fn: F,
}

impl<F> FnTool<F>
where
    F: Fn(Value) -> futures::future::BoxFuture<'static, Result<ToolOutput, ToolError>>
        + Send
        + Sync,
{
    /// Create a new function-based tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        execute_fn: F,
    ) -> Self {
        Self {
            descriptor: ToolDescriptor::new(name, description, input_schema),
            execute_fn,
        }
    }

    /// Set the output schema for this tool.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.descriptor = self.descriptor.with_output_schema(schema);
        self
    }
}

#[async_trait::async_trait]
impl<F> Tool for FnTool<F>
where
    F: Fn(Value) -> futures::future::BoxFuture<'static, Result<ToolOutput, ToolError>>
        + Send
        + Sync,
{
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor.clone()
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        (self.execute_fn)(input).await
    }
}

/// Registry for managing available tools.
///
/// The tool registry provides a centralized place to register, retrieve,
/// and list tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Box<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<_> = self.tools.keys().collect();
        f.debug_struct("ToolRegistry")
            .field("tools", &names)
            .finish()
    }
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    ///
    /// # Panics
    ///
    /// Panics if a tool with the same name is already registered.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let descriptor = tool.descriptor();
        let name = descriptor.name.clone();
        if self.tools.contains_key(&name) {
            panic!("Tool already registered: {}", name);
        }
        self.tools.insert(name, Box::new(tool));
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// List all registered tool names.
    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get all tool descriptors.
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .values()
            .map(|t| t.descriptor())
            .collect()
    }

    /// Check if a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTool;

    #[async_trait::async_trait]
    impl Tool for TestTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new(
                "test",
                "A test tool",
                json!({"type": "object"}),
            )
        }

        async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(format!("Received: {}", input)))
        }
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tool = TestTool;
        let result = tool
            .execute(json!({"key": "value"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Received"));
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(TestTool);

        assert!(registry.contains("test"));
        assert_eq!(registry.list(), vec!["test".to_string()]);
    }

    #[test]
    fn test_tool_output() {
        let success = ToolOutput::success("done");
        assert!(!success.is_error);
        assert_eq!(success.content, "done");

        let error = ToolOutput::error("failed");
        assert!(error.is_error);
        assert_eq!(error.content, "failed");
    }
}
