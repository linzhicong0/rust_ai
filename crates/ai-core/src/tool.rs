//! Tool execution and definition for agent function calling.
//!
//! Tools are functions that agents can call during execution. This module
//! defines the [`Tool`] trait and related types for declaring and executing
//! tools with schema validation.
//!
//! ## Example
//!
//! ```rust
//! # use ai_core::{Tool, ToolDescriptor, ToolOutput};
//! # use ai_core::error::ToolError;
//! # use serde_json::json;
//! struct WeatherTool;
//!
//! # #[async_trait::async_trait]
//! impl Tool for WeatherTool {
//!     fn descriptor(&self) -> ToolDescriptor {
//!         ToolDescriptor::new(
//!             "get_weather",
//!             "Get current weather for a location",
//!             json!({
//!                 "type": "object",
//!                 "properties": {
//!                     "location": {
//!                         "type": "string",
//!                         "description": "City name, e.g. San Francisco"
//!                     }
//!                 },
//!                 "required": ["location"]
//!             }),
//!         )
//!     }
//!
//!     async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
//!         let location = input["location"]
//!             .as_str()
//!             .ok_or_else(|| ToolError::InvalidInput("location required".to_string()))?;
//!
//!         // Call weather API...
//!         Ok(ToolOutput::success(format!("Weather in {location}: 72°F, sunny")))
//!     }
//! }
//! ```

use async_trait::async_trait;
use jsonschema::Validator;
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

    /// Tags for categorizing and querying tools (REQ-3.2).
    ///
    /// Tags enable filtering tools by capability, e.g., `["search", "web"]`.
    pub tags: Vec<String>,
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
            tags: Vec::new(),
        }
    }

    /// Set the output schema for structured output validation.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Add tags to this descriptor for discovery and querying (REQ-3.2).
    pub fn with_tags(mut self, tags: Vec<impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Add a single tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
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
/// # use ai_core::tool::FnTool;
/// # use ai_core::ToolOutput;
/// # use ai_core::error::ToolError;
/// # use serde_json::json;
/// # use futures::FutureExt;
/// let tool = FnTool::new(
///     "echo",
///     "Echoes the input",
///     json!({"type": "object"}),
///     |input| async move {
///         Ok(ToolOutput::success(format!("Echo: {}", input)))
///     }.boxed(),
/// );
/// # let _ = tool;
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

    /// Try to register a tool, returning an error if a duplicate name exists (REQ-3.2).
    ///
    /// This is the non-panicking variant suitable for runtime/plugin registration.
    pub fn try_register<T: Tool + 'static>(&mut self, tool: T) -> Result<(), ToolError> {
        let descriptor = tool.descriptor();
        let name = descriptor.name.clone();
        if self.tools.contains_key(&name) {
            return Err(ToolError::InvalidInput(format!(
                "Tool already registered: {}",
                name
            )));
        }
        self.tools.insert(name, Box::new(tool));
        Ok(())
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
        self.tools.values().map(|t| t.descriptor()).collect()
    }

    /// Check if a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Query tools by tag (REQ-3.2).
    ///
    /// Returns all tools that have the specified tag in their descriptor.
    pub fn query_by_tag(&self, tag: &str) -> Vec<&dyn Tool> {
        self.tools
            .values()
            .filter(|t| t.descriptor().tags.contains(&tag.to_string()))
            .map(|t| t.as_ref())
            .collect()
    }

    /// Query tools by capability substring match in description (REQ-3.2).
    ///
    /// Returns all tools whose description contains the given capability string
    /// (case-insensitive).
    pub fn query_by_capability(&self, capability: &str) -> Vec<&dyn Tool> {
        let capability_lower = capability.to_lowercase();
        self.tools
            .values()
            .filter(|t| {
                t.descriptor()
                    .description
                    .to_lowercase()
                    .contains(&capability_lower)
            })
            .map(|t| t.as_ref())
            .collect()
    }
}

/// Result of tool input validation.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Path to the invalid field (e.g., "properties.name").
    pub path: String,
    /// Descriptive error message.
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Coerce compatible types in tool input to match the schema.
///
/// For example, a string "5" will be coerced to integer 5 if the schema
/// expects an integer type.
pub fn coerce_input(input: &Value, schema: &Value) -> Value {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => {
            if let Value::Object(map) = input {
                let properties = schema
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .cloned()
                    .unwrap_or_default();
                let mut result = serde_json::Map::new();
                for (key, value) in map {
                    if let Some(prop_schema) = properties.get(key) {
                        result.insert(key.clone(), coerce_input(value, prop_schema));
                    } else {
                        result.insert(key.clone(), value.clone());
                    }
                }
                Value::Object(result)
            } else {
                input.clone()
            }
        }
        Some("integer") | Some("number") => {
            if let Value::String(s) = input {
                if let Ok(n) = s.parse::<i64>() {
                    Value::Number(serde_json::Number::from(n))
                } else if let Ok(n) = s.parse::<f64>() {
                    serde_json::Number::from_f64(n)
                        .map(Value::Number)
                        .unwrap_or_else(|| input.clone())
                } else {
                    input.clone()
                }
            } else {
                input.clone()
            }
        }
        Some("boolean") => {
            if let Value::String(s) = input {
                match s.as_str() {
                    "true" => Value::Bool(true),
                    "false" => Value::Bool(false),
                    _ => input.clone(),
                }
            } else {
                input.clone()
            }
        }
        Some("string") => match input {
            Value::Number(n) => Value::String(n.to_string()),
            Value::Bool(b) => Value::String(b.to_string()),
            _ => input.clone(),
        },
        Some("array") => {
            if let Value::Array(arr) = input {
                let item_schema = schema
                    .get("items")
                    .cloned()
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                let coerced: Vec<Value> = arr
                    .iter()
                    .map(|item| coerce_input(item, &item_schema))
                    .collect();
                Value::Array(coerced)
            } else {
                input.clone()
            }
        }
        _ => input.clone(),
    }
}

/// Validate tool input against its JSON Schema.
///
/// Returns `Ok(coerced_input)` if valid (after type coercion), or `Err` with
/// a descriptive error including the path to the invalid field.
pub fn validate_tool_input(input: &Value, schema: &Value) -> Result<Value, ToolError> {
    // First, coerce compatible types
    let coerced = coerce_input(input, schema);

    // Then validate against schema
    let validator = Validator::new(schema)
        .map_err(|e| ToolError::InvalidInput(format!("Invalid schema: {}", e)))?;

    if validator.is_valid(&coerced) {
        Ok(coerced)
    } else {
        // Collect validation errors with paths
        let errors: Vec<ValidationError> = validator
            .iter_errors(&coerced)
            .map(|error| {
                let path = error.instance_path.to_string();
                let path = if path.is_empty() {
                    error.schema_path.to_string()
                } else {
                    path
                };
                ValidationError {
                    path,
                    message: error.to_string(),
                }
            })
            .collect();

        if let Some(first) = errors.first() {
            Err(ToolError::InvalidInput(format!("{}", first)))
        } else {
            Err(ToolError::InvalidInput("Validation failed".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use serde_json::json;

    // REQ-3.1: Tool Definition Tests

    struct TestTool;

    #[async_trait::async_trait]
    impl Tool for TestTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("test", "A test tool", json!({"type": "object"}))
        }

        async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(format!("Received: {}", input)))
        }
    }

    struct AnotherTestTool;

    #[async_trait::async_trait]
    impl Tool for AnotherTestTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new(
                "another_test",
                "Another test tool",
                json!({"type": "object"}),
            )
        }

        async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("OK"))
        }
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tool = TestTool;
        let result = tool.execute(json!({"key": "value"})).await.unwrap();
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

    #[test]
    fn test_tool_descriptor_new() {
        let descriptor = ToolDescriptor::new("my_tool", "A description", json!({"type": "object"}));

        assert_eq!(descriptor.name, "my_tool");
        assert_eq!(descriptor.description, "A description");
        assert_eq!(descriptor.input_schema, json!({"type": "object"}));
        assert!(descriptor.output_schema.is_none());
    }

    #[test]
    fn test_tool_descriptor_with_output_schema() {
        let descriptor = ToolDescriptor::new("my_tool", "A description", json!({"type": "object"}))
            .with_output_schema(json!({"type": "string"}));

        assert_eq!(descriptor.output_schema, Some(json!({"type": "string"})));
    }

    #[test]
    fn test_tool_output_json() {
        #[derive(Serialize)]
        struct TestStruct {
            message: String,
            count: i32,
        }

        let value = TestStruct {
            message: "hello".to_string(),
            count: 42,
        };

        let output = ToolOutput::json(&value).unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("hello"));
        assert!(output.content.contains("42"));
    }

    #[test]
    fn test_tool_output_json_serialization_success() {
        // Test that serialization works correctly for valid types
        #[derive(Serialize)]
        struct TestStruct {
            message: String,
        }

        let test = TestStruct {
            message: "test".to_string(),
        };
        let result = ToolOutput::json(&test);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_error);
    }

    #[tokio::test]
    async fn test_tool_registry_multiple_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(TestTool);
        registry.register(AnotherTestTool);

        assert!(registry.contains("test"));
        assert!(registry.contains("another_test"));

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["another_test", "test"]);
    }

    #[test]
    fn test_tool_registry_get() {
        let mut registry = ToolRegistry::new();
        registry.register(TestTool);

        let tool = registry.get("test");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().descriptor().name, "test");

        let nonexistent = registry.get("nonexistent");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_tool_registry_descriptors() {
        let mut registry = ToolRegistry::new();
        registry.register(TestTool);
        registry.register(AnotherTestTool);

        let descriptors = registry.descriptors();
        assert_eq!(descriptors.len(), 2);

        let names: Vec<_> = descriptors.iter().map(|d| &d.name).collect();
        assert!(names.contains(&&"test".to_string()));
        assert!(names.contains(&&"another_test".to_string()));
    }

    #[test]
    #[should_panic(expected = "Tool already registered")]
    fn test_tool_registry_duplicate_panics() {
        let mut registry = ToolRegistry::new();
        registry.register(TestTool);
        registry.register(TestTool); // Should panic
    }

    // Test FnTool
    #[tokio::test]
    async fn test_fn_tool() {
        let tool = FnTool::new(
            "echo",
            "Echoes the input",
            json!({"type": "object"}),
            |input| async move { Ok(ToolOutput::success(format!("Echo: {}", input))) }.boxed(),
        );

        let descriptor = tool.descriptor();
        assert_eq!(descriptor.name, "echo");
        assert_eq!(descriptor.description, "Echoes the input");

        let result = tool.execute(json!("test")).await.unwrap();
        assert_eq!(result.content, "Echo: \"test\"");
    }

    #[tokio::test]
    async fn test_fn_tool_with_error() {
        let tool = FnTool::new(
            "fail_tool",
            "A tool that fails",
            json!({"type": "object"}),
            |_input| async move { Ok(ToolOutput::error("Something went wrong")) }.boxed(),
        );

        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.is_error);
        assert_eq!(result.content, "Something went wrong");
    }

    #[tokio::test]
    async fn test_fn_tool_with_output_schema() {
        let tool = FnTool::new(
            "json_tool",
            "Returns JSON",
            json!({"type": "object"}),
            |_input| async move { Ok(ToolOutput::success("{\"result\": 42}".to_string())) }.boxed(),
        )
        .with_output_schema(json!({"type": "string"}));

        let descriptor = tool.descriptor();
        assert_eq!(descriptor.output_schema, Some(json!({"type": "string"})));
    }

    // Test SyncTool and SyncToolAdapter
    struct SyncTestTool;

    impl SyncTool for SyncTestTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("sync_test", "A sync test tool", json!({"type": "object"}))
        }

        fn execute_sync(&self, input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(format!("Sync: {}", input)))
        }
    }

    #[tokio::test]
    async fn test_sync_tool_adapter() {
        let adapter = SyncToolAdapter(SyncTestTool);

        let descriptor = adapter.descriptor();
        assert_eq!(descriptor.name, "sync_test");

        let result = adapter.execute(json!("test")).await.unwrap();
        assert_eq!(result.content, "Sync: \"test\"");
    }

    #[tokio::test]
    async fn test_sync_tool_adapter_error() {
        struct FailingSyncTool;

        impl SyncTool for FailingSyncTool {
            fn descriptor(&self) -> ToolDescriptor {
                ToolDescriptor::new(
                    "failing_sync",
                    "A failing sync tool",
                    json!({"type": "object"}),
                )
            }

            fn execute_sync(&self, _input: Value) -> Result<ToolOutput, ToolError> {
                Err(ToolError::Execution("Failed".to_string()))
            }
        }

        let adapter = SyncToolAdapter(FailingSyncTool);
        let result = adapter.execute(json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::Execution(_)));
    }

    // Test simple_descriptor helper
    #[test]
    fn test_simple_descriptor() {
        let descriptor = simple_descriptor(
            "my_tool",
            "Does something",
            json!({"type": "object", "properties": {}}),
        );

        assert_eq!(descriptor.name, "my_tool");
        assert_eq!(descriptor.description, "Does something");
        assert_eq!(
            descriptor.input_schema,
            json!({"type": "object", "properties": {}})
        );
    }

    // Test ToolOutput edge cases
    #[test]
    fn test_tool_output_empty_content() {
        let success = ToolOutput::success("");
        assert!(!success.is_error);
        assert_eq!(success.content, "");

        let error = ToolOutput::error("");
        assert!(error.is_error);
        assert_eq!(error.content, "");
    }

    #[test]
    fn test_tool_output_from_string() {
        let success = ToolOutput::success(String::from("test"));
        assert_eq!(success.content, "test");

        let error = ToolOutput::error(String::from("failed"));
        assert_eq!(error.content, "failed");
    }

    #[test]
    fn test_tool_output_long_content() {
        let long_content = "a".repeat(10000);
        let output = ToolOutput::success(long_content.clone());
        assert_eq!(output.content.len(), 10000);
    }

    // Test ToolRegistry edge cases
    #[test]
    fn test_tool_registry_empty() {
        let registry = ToolRegistry::new();
        assert!(!registry.contains("anything"));
        assert!(registry.list().is_empty());
        assert!(registry.descriptors().is_empty());
        assert!(registry.get("anything").is_none());
    }

    // REQ-3.4: Tool Validation Tests

    #[test]
    fn test_valid_input_passes_validation_and_execution_proceeds() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let input = json!({"name": "Alice", "age": 30});
        let result = validate_tool_input(&input, &schema);
        assert!(result.is_ok());
        let coerced = result.unwrap();
        assert_eq!(coerced["name"], "Alice");
        assert_eq!(coerced["age"], 30);
    }

    #[test]
    fn test_missing_required_field_returns_error_with_field_path() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let input = json!({"age": 30}); // missing "name"
        let result = validate_tool_input(&input, &schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        // Error should mention the missing required field
        assert!(
            err_msg.contains("name") || err_msg.contains("required"),
            "Error should reference the missing field: {}",
            err_msg
        );
    }

    #[test]
    fn test_string_5_for_integer_field_is_coerced_to_5() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            },
            "required": ["count"]
        });

        let input = json!({"count": "5"});
        let result = validate_tool_input(&input, &schema);
        assert!(result.is_ok());
        let coerced = result.unwrap();
        assert_eq!(coerced["count"], 5);
    }

    #[test]
    fn test_invalid_type_returns_descriptive_type_error() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });

        // Pass an object where a string is expected
        let input = json!({"name": {"nested": "object"}});
        let result = validate_tool_input(&input, &schema);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        // Error should reference the path and type mismatch
        assert!(
            err_msg.contains("name") || err_msg.contains("type") || err_msg.contains("string"),
            "Error should describe type mismatch: {}",
            err_msg
        );
    }

    #[test]
    fn test_coerce_string_to_number() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        });
        let input = json!({"value": "3.14"});
        let coerced = coerce_input(&input, &schema);
        assert_eq!(coerced["value"], 3.14);
    }

    #[test]
    fn test_coerce_string_to_boolean() {
        let schema = json!({
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"}
            }
        });
        let input = json!({"flag": "true"});
        let coerced = coerce_input(&input, &schema);
        assert_eq!(coerced["flag"], true);
    }

    // REQ-3.2: Tool Discovery Tests

    struct TaggedSearchTool;

    #[async_trait::async_trait]
    impl Tool for TaggedSearchTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new(
                "web_search",
                "Search the web for information",
                json!({"type": "object"}),
            )
            .with_tags(vec!["search", "web"])
        }

        async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("search result"))
        }
    }

    struct TaggedDbTool;

    #[async_trait::async_trait]
    impl Tool for TaggedDbTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new(
                "db_search",
                "Search the database",
                json!({"type": "object"}),
            )
            .with_tags(vec!["search", "database"])
        }

        async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("db result"))
        }
    }

    struct TaggedFileTool;

    #[async_trait::async_trait]
    impl Tool for TaggedFileTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("file_read", "Read a file", json!({"type": "object"}))
                .with_tags(vec!["file", "io"])
        }

        async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("file content"))
        }
    }

    // REQ-3.2: Unit: register a tool and retrieve it by exact name
    #[test]
    fn test_register_and_retrieve_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(TaggedSearchTool);

        let tool = registry.get("web_search");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().descriptor().name, "web_search");
    }

    // REQ-3.2: Unit: list() returns all registered tools
    #[test]
    fn test_list_returns_all_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(TaggedSearchTool);
        registry.register(TaggedDbTool);
        registry.register(TaggedFileTool);

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["db_search", "file_read", "web_search"]);
    }

    // REQ-3.2: Unit: query by tag "search" returns all tools with that tag
    #[test]
    fn test_query_by_tag_search() {
        let mut registry = ToolRegistry::new();
        registry.register(TaggedSearchTool);
        registry.register(TaggedDbTool);
        registry.register(TaggedFileTool);

        let search_tools = registry.query_by_tag("search");
        assert_eq!(search_tools.len(), 2);

        let names: Vec<String> = search_tools.iter().map(|t| t.descriptor().name).collect();
        assert!(names.contains(&"web_search".to_string()));
        assert!(names.contains(&"db_search".to_string()));
    }

    // REQ-3.2: Unit: plugin registers a tool at runtime and it appears in list()
    #[test]
    fn test_runtime_registration_appears_in_list() {
        let mut registry = ToolRegistry::new();
        registry.register(TaggedSearchTool);

        // Simulate plugin runtime registration
        let result = registry.try_register(TaggedDbTool);
        assert!(result.is_ok());

        let names = registry.list();
        assert!(names.contains(&"web_search".to_string()));
        assert!(names.contains(&"db_search".to_string()));
    }

    // REQ-3.2: Unit: registering duplicate name returns an error
    #[test]
    fn test_try_register_duplicate_returns_error() {
        let mut registry = ToolRegistry::new();
        registry.register(TaggedSearchTool);

        // Create another tool with same name
        struct DuplicateTool;

        #[async_trait::async_trait]
        impl Tool for DuplicateTool {
            fn descriptor(&self) -> ToolDescriptor {
                ToolDescriptor::new("web_search", "Duplicate", json!({"type": "object"}))
            }

            async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
                Ok(ToolOutput::success(""))
            }
        }

        let result = registry.try_register(DuplicateTool);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("web_search"));
    }
}
