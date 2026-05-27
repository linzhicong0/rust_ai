// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # LangChain Compatibility (REQ-17.4)
//!
//! Compatibility layers and adapters for the LangChain ecosystem, enabling
//! interoperability between LangChain tools/chains/agents and this framework.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::langchain::{
//!     LangChainToolAdapter, LangChainTool, FrameworkChainAdapter, ChainStep,
//! };
//!
//! // Adapt a LangChain-style tool to our framework tool interface
//! let lc_tool = LangChainTool::new("web_search", "Search the web", |input| {
//!     Ok(format!("Results for: {}", input))
//! });
//! let adapter = LangChainToolAdapter::new(lc_tool);
//! assert_eq!(adapter.name(), "web_search");
//! ```

use std::collections::HashMap;
use std::fmt;

// ── LangChainTool ─────────────────────────────────────────────────────────────

/// Represents a LangChain-style tool (name, description, function).
#[derive(Clone)]
pub struct LangChainTool {
    /// Tool name.
    name: String,
    /// Tool description.
    description: String,
    /// The tool function: takes a string input, returns a string output.
    func: fn(&str) -> Result<String, String>,
}

impl fmt::Debug for LangChainTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LangChainTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl LangChainTool {
    /// Create a new LangChain tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        func: fn(&str) -> Result<String, String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            func,
        }
    }

    /// Get the tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the tool description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Invoke the tool with the given input.
    pub fn invoke(&self, input: &str) -> Result<String, String> {
        (self.func)(input)
    }
}

// ── LangChainToolAdapter ──────────────────────────────────────────────────────

/// Adapter: LangChain tool → framework tool interface.
///
/// Wraps a [`LangChainTool`] so it can be used as a framework tool.
#[derive(Debug, Clone)]
pub struct LangChainToolAdapter {
    inner: LangChainTool,
    /// JSON schema for the tool input (optional).
    input_schema: Option<serde_json::Value>,
}

impl LangChainToolAdapter {
    /// Create a new adapter wrapping a LangChain tool.
    pub fn new(tool: LangChainTool) -> Self {
        Self {
            inner: tool,
            input_schema: None,
        }
    }

    /// Set the input schema for the adapted tool.
    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Get the tool name.
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the tool description.
    pub fn description(&self) -> &str {
        self.inner.description()
    }

    /// Get the input schema.
    pub fn input_schema(&self) -> Option<&serde_json::Value> {
        self.input_schema.as_ref()
    }

    /// Execute the tool with a string input.
    pub fn execute(&self, input: &str) -> Result<String, LangChainError> {
        self.inner
            .invoke(input)
            .map_err(|e| LangChainError::ToolExecution(e))
    }

    /// Execute the tool with JSON input (extracts string from "input" field).
    pub fn execute_json(&self, input: &serde_json::Value) -> Result<String, LangChainError> {
        let input_str = match input {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(s)) = map.get("input") {
                    s.clone()
                } else {
                    serde_json::to_string(input)
                        .map_err(|e| LangChainError::Serialization(e.to_string()))?
                }
            }
            other => serde_json::to_string(other)
                .map_err(|e| LangChainError::Serialization(e.to_string()))?,
        };
        self.execute(&input_str)
    }
}

// ── ChainStep ─────────────────────────────────────────────────────────────────

/// A step in a LangChain-style chain.
#[derive(Debug, Clone)]
pub struct ChainStep {
    /// Step name.
    pub name: String,
    /// Step type.
    pub step_type: ChainStepType,
    /// Input mapping (from chain context key to step input).
    pub input_key: String,
    /// Output key (where to store the result in chain context).
    pub output_key: String,
}

/// Type of chain step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainStepType {
    /// LLM call step.
    Llm { model: String },
    /// Tool invocation step.
    Tool { tool_name: String },
    /// Transform step (applies a function).
    Transform { description: String },
    /// Conditional branch.
    Conditional { condition_key: String },
}

impl ChainStep {
    /// Create a new LLM chain step.
    pub fn llm(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            step_type: ChainStepType::Llm {
                model: model.into(),
            },
            input_key: "input".to_string(),
            output_key: "output".to_string(),
        }
    }

    /// Create a new tool chain step.
    pub fn tool(name: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            step_type: ChainStepType::Tool {
                tool_name: tool_name.into(),
            },
            input_key: "input".to_string(),
            output_key: "output".to_string(),
        }
    }

    /// Create a new transform chain step.
    pub fn transform(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            step_type: ChainStepType::Transform {
                description: description.into(),
            },
            input_key: "input".to_string(),
            output_key: "output".to_string(),
        }
    }

    /// Set the input key.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the output key.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }
}

// ── FrameworkChainAdapter ─────────────────────────────────────────────────────

/// Adapter: framework agent → LangChain chain.
///
/// Represents a framework agent as a LangChain-compatible chain with
/// sequential steps and context passing.
#[derive(Debug, Clone)]
pub struct FrameworkChainAdapter {
    /// Chain name.
    name: String,
    /// Chain steps.
    steps: Vec<ChainStep>,
    /// Chain context (key-value store).
    context: HashMap<String, String>,
    /// Whether the chain has been executed.
    executed: bool,
}

impl FrameworkChainAdapter {
    /// Create a new chain adapter.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            context: HashMap::new(),
            executed: false,
        }
    }

    /// Get the chain name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a step to the chain.
    pub fn add_step(&mut self, step: ChainStep) {
        self.steps.push(step);
    }

    /// Builder-style: add a step.
    pub fn with_step(mut self, step: ChainStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Get all steps.
    pub fn steps(&self) -> &[ChainStep] {
        &self.steps
    }

    /// Set a context value.
    pub fn set_context(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.context.insert(key.into(), value.into());
    }

    /// Get a context value.
    pub fn get_context(&self, key: &str) -> Option<&String> {
        self.context.get(key)
    }

    /// Check if the chain has been executed.
    pub fn is_executed(&self) -> bool {
        self.executed
    }

    /// Validate the chain configuration.
    pub fn validate(&self) -> Result<(), LangChainError> {
        if self.name.is_empty() {
            return Err(LangChainError::InvalidConfig(
                "chain name is required".into(),
            ));
        }
        if self.steps.is_empty() {
            return Err(LangChainError::InvalidConfig(
                "chain must have at least one step".into(),
            ));
        }
        Ok(())
    }

    /// Execute the chain (simulation - processes steps sequentially).
    pub fn execute(&mut self, input: &str) -> Result<String, LangChainError> {
        self.validate()?;

        // Set initial input
        self.context.insert("input".to_string(), input.to_string());

        let mut current_output = input.to_string();

        for step in &self.steps {
            // Get input for this step
            let step_input = self
                .context
                .get(&step.input_key)
                .cloned()
                .unwrap_or_else(|| current_output.clone());

            // Simulate step execution
            let step_output = match &step.step_type {
                ChainStepType::Llm { model } => {
                    format!("[LLM:{}] processed: {}", model, step_input)
                }
                ChainStepType::Tool { tool_name } => {
                    format!("[Tool:{}] result for: {}", tool_name, step_input)
                }
                ChainStepType::Transform { description } => {
                    format!("[Transform:{}] {}", description, step_input)
                }
                ChainStepType::Conditional { condition_key } => {
                    let condition = self.context.get(condition_key).cloned().unwrap_or_default();
                    format!("[Conditional:{}={}] {}", condition_key, condition, step_input)
                }
            };

            // Store output in context
            self.context
                .insert(step.output_key.clone(), step_output.clone());
            current_output = step_output;
        }

        self.executed = true;
        Ok(current_output)
    }
}

// ── PythonBridgeConfig ────────────────────────────────────────────────────────

/// Configuration for the Python bridge (PyO3 interop).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonBridgeConfig {
    /// Python executable path.
    pub python_path: String,
    /// Whether to use PyO3 for direct embedding.
    pub use_pyo3: bool,
    /// Virtual environment path (optional).
    pub venv_path: Option<String>,
    /// Additional Python packages required.
    pub required_packages: Vec<String>,
}

impl Default for PythonBridgeConfig {
    fn default() -> Self {
        Self {
            python_path: "python3".to_string(),
            use_pyo3: false,
            venv_path: None,
            required_packages: vec!["langchain".to_string(), "langchain-core".to_string()],
        }
    }
}

impl PythonBridgeConfig {
    /// Create a new Python bridge config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable PyO3 direct embedding.
    pub fn with_pyo3(mut self) -> Self {
        self.use_pyo3 = true;
        self
    }

    /// Set virtual environment path.
    pub fn with_venv(mut self, path: impl Into<String>) -> Self {
        self.venv_path = Some(path.into());
        self
    }

    /// Add a required package.
    pub fn with_package(mut self, package: impl Into<String>) -> Self {
        self.required_packages.push(package.into());
        self
    }
}

// ── LangChainError ────────────────────────────────────────────────────────────

/// Errors in the LangChain compatibility layer.
#[derive(Debug, thiserror::Error)]
pub enum LangChainError {
    /// Invalid configuration.
    #[error("Invalid LangChain config: {0}")]
    InvalidConfig(String),
    /// Tool execution error.
    #[error("Tool execution failed: {0}")]
    ToolExecution(String),
    /// Chain execution error.
    #[error("Chain execution failed: {0}")]
    ChainExecution(String),
    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
    /// Python bridge error.
    #[error("Python bridge error: {0}")]
    PythonBridge(String),
    /// Adapter error.
    #[error("Adapter error: {0}")]
    AdapterError(String),
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// Registry for managing LangChain tool adapters.
#[derive(Debug)]
pub struct ToolRegistry {
    tools: HashMap<String, LangChainToolAdapter>,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a LangChain tool.
    pub fn register(&mut self, tool: LangChainTool) {
        let name = tool.name().to_string();
        self.tools.insert(name, LangChainToolAdapter::new(tool));
    }

    /// Register a pre-built adapter.
    pub fn register_adapter(&mut self, adapter: LangChainToolAdapter) {
        let name = adapter.name().to_string();
        self.tools.insert(name, adapter);
    }

    /// Get a tool adapter by name.
    pub fn get(&self, name: &str) -> Option<&LangChainToolAdapter> {
        self.tools.get(name)
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|k| k.as_str()).collect()
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Remove a tool by name.
    pub fn remove(&mut self, name: &str) -> Option<LangChainToolAdapter> {
        self.tools.remove(name)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn search_tool(input: &str) -> Result<String, String> {
        Ok(format!("Results for: {}", input))
    }

    fn calculator_tool(input: &str) -> Result<String, String> {
        Ok(format!("Calculated: {}", input))
    }

    fn failing_tool(_input: &str) -> Result<String, String> {
        Err("tool failed".to_string())
    }

    // REQ-17.4: Adapter - LangChain tool → framework tool
    #[test]
    fn test_langchain_tool_to_framework_adapter() {
        let lc_tool = LangChainTool::new("web_search", "Search the web", search_tool);
        let adapter = LangChainToolAdapter::new(lc_tool);

        assert_eq!(adapter.name(), "web_search");
        assert_eq!(adapter.description(), "Search the web");

        let result = adapter.execute("rust programming").unwrap();
        assert_eq!(result, "Results for: rust programming");
    }

    // REQ-17.4: Adapter with input schema
    #[test]
    fn test_adapter_with_input_schema() {
        let lc_tool = LangChainTool::new("search", "Search", search_tool);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            }
        });

        let adapter = LangChainToolAdapter::new(lc_tool).with_input_schema(schema.clone());
        assert_eq!(adapter.input_schema(), Some(&schema));
    }

    // REQ-17.4: Adapter handles JSON input
    #[test]
    fn test_adapter_json_input() {
        let lc_tool = LangChainTool::new("search", "Search", search_tool);
        let adapter = LangChainToolAdapter::new(lc_tool);

        // String input
        let result = adapter
            .execute_json(&serde_json::json!("hello"))
            .unwrap();
        assert_eq!(result, "Results for: hello");

        // Object with "input" field
        let result = adapter
            .execute_json(&serde_json::json!({"input": "world"}))
            .unwrap();
        assert_eq!(result, "Results for: world");
    }

    // REQ-17.4: Adapter handles tool failure
    #[test]
    fn test_adapter_tool_failure() {
        let lc_tool = LangChainTool::new("broken", "A broken tool", failing_tool);
        let adapter = LangChainToolAdapter::new(lc_tool);

        let result = adapter.execute("anything");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LangChainError::ToolExecution(_)));
    }

    // REQ-17.4: Adapter - framework agent → LangChain chain
    #[test]
    fn test_framework_chain_adapter() {
        let mut chain = FrameworkChainAdapter::new("qa-chain")
            .with_step(ChainStep::llm("generate", "gpt-4"))
            .with_step(ChainStep::tool("search", "web_search"));

        assert_eq!(chain.name(), "qa-chain");
        assert_eq!(chain.steps().len(), 2);
        assert!(!chain.is_executed());

        let result = chain.execute("What is Rust?").unwrap();
        assert!(chain.is_executed());
        assert!(result.contains("Tool:web_search"));
    }

    // REQ-17.4: Chain with custom input/output keys
    #[test]
    fn test_chain_custom_keys() {
        let mut chain = FrameworkChainAdapter::new("custom-chain").with_step(
            ChainStep::llm("summarize", "gpt-4")
                .with_input_key("document")
                .with_output_key("summary"),
        );

        chain.set_context("document", "Long document text here...");
        let result = chain.execute("ignored").unwrap();
        assert!(result.contains("processed: Long document text here"));
    }

    // REQ-17.4: Chain validation
    #[test]
    fn test_chain_validation_empty_name() {
        let chain = FrameworkChainAdapter::new("")
            .with_step(ChainStep::llm("step1", "gpt-4"));
        assert!(chain.validate().is_err());
    }

    // REQ-17.4: Chain validation - no steps
    #[test]
    fn test_chain_validation_no_steps() {
        let chain = FrameworkChainAdapter::new("empty-chain");
        assert!(chain.validate().is_err());
    }

    // REQ-17.4: Chain context management
    #[test]
    fn test_chain_context() {
        let mut chain = FrameworkChainAdapter::new("ctx-chain")
            .with_step(ChainStep::llm("step1", "gpt-4"));

        chain.set_context("key1", "value1");
        chain.set_context("key2", "value2");

        assert_eq!(chain.get_context("key1"), Some(&"value1".to_string()));
        assert_eq!(chain.get_context("key2"), Some(&"value2".to_string()));
        assert_eq!(chain.get_context("missing"), None);
    }

    // REQ-17.4: Tool registry
    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(LangChainTool::new("search", "Search the web", search_tool));
        registry.register(LangChainTool::new("calc", "Calculator", calculator_tool));

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());

        let search = registry.get("search").unwrap();
        assert_eq!(search.name(), "search");

        let result = search.execute("test query").unwrap();
        assert_eq!(result, "Results for: test query");
    }

    // REQ-17.4: Tool registry - remove tool
    #[test]
    fn test_tool_registry_remove() {
        let mut registry = ToolRegistry::new();
        registry.register(LangChainTool::new("search", "Search", search_tool));

        assert_eq!(registry.len(), 1);
        let removed = registry.remove("search");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);
    }

    // REQ-17.4: Python bridge configuration
    #[test]
    fn test_python_bridge_config() {
        let config = PythonBridgeConfig::new()
            .with_pyo3()
            .with_venv("/opt/venv")
            .with_package("langchain-openai");

        assert!(config.use_pyo3);
        assert_eq!(config.venv_path, Some("/opt/venv".to_string()));
        assert!(config.required_packages.contains(&"langchain".to_string()));
        assert!(config
            .required_packages
            .contains(&"langchain-openai".to_string()));
    }

    // REQ-17.4: Python bridge default config
    #[test]
    fn test_python_bridge_default() {
        let config = PythonBridgeConfig::default();

        assert_eq!(config.python_path, "python3");
        assert!(!config.use_pyo3);
        assert_eq!(config.venv_path, None);
        assert!(config
            .required_packages
            .contains(&"langchain-core".to_string()));
    }

    // REQ-17.4: Chain with transform step
    #[test]
    fn test_chain_transform_step() {
        let mut chain = FrameworkChainAdapter::new("transform-chain")
            .with_step(ChainStep::transform("upper", "to uppercase"));

        let result = chain.execute("hello world").unwrap();
        assert!(result.contains("Transform:to uppercase"));
        assert!(result.contains("hello world"));
    }

    // REQ-17.4: Multiple chains can be composed
    #[test]
    fn test_chain_multi_step_execution() {
        let mut chain = FrameworkChainAdapter::new("multi-step")
            .with_step(ChainStep::llm("analyze", "gpt-4"))
            .with_step(ChainStep::tool("lookup", "web_search"))
            .with_step(ChainStep::transform("format", "format output"));

        let result = chain.execute("What is AI?").unwrap();
        assert!(chain.is_executed());
        // Final output should be from the last step (transform)
        assert!(result.contains("Transform:format output"));
    }
}
