//! Tool composition and piping (REQ-3.5).
//!
//! This module enables tool chaining where the output of one tool feeds
//! into the input of another within a single agent turn.
//!
//! ## Features
//!
//! - Pipeline of tools: `ToolPipeline::new().pipe(tool_a).pipe(tool_b)`
//! - Intermediate result inspection for debugging
//! - Type-checked composition (output JSON → input JSON)

use crate::error::ToolError;
use crate::tool::{Tool, ToolDescriptor, ToolOutput};
use async_trait::async_trait;
use serde_json::Value;

/// An intermediate result captured during pipeline execution.
#[derive(Debug, Clone)]
pub struct IntermediateResult {
    /// Name of the tool that produced this result.
    pub tool_name: String,
    /// The output from this tool.
    pub output: ToolOutput,
}

/// A pipeline that chains multiple tools together.
///
/// The output of each tool is parsed as JSON and fed as input to the next tool.
/// Intermediate results are captured for debugging/inspection.
pub struct ToolPipeline {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolPipeline {
    /// Create an empty tool pipeline.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool to the end of the pipeline.
    pub fn pipe<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Execute the pipeline, passing the initial input through all tools.
    ///
    /// Returns the final output and all intermediate results.
    pub async fn execute(
        &self,
        initial_input: Value,
    ) -> Result<(ToolOutput, Vec<IntermediateResult>), ToolError> {
        if self.tools.is_empty() {
            return Err(ToolError::Execution("Pipeline has no tools".to_string()));
        }

        let mut current_input = initial_input;
        let mut intermediates = Vec::new();

        for (i, tool) in self.tools.iter().enumerate() {
            let input_for_tool = current_input;
            let output = tool.execute(input_for_tool).await?;

            if output.is_error {
                return Err(ToolError::Execution(format!(
                    "Tool '{}' (step {}) returned error: {}",
                    tool.descriptor().name,
                    i,
                    output.content
                )));
            }

            let intermediate = IntermediateResult {
                tool_name: tool.descriptor().name.clone(),
                output: output.clone(),
            };
            intermediates.push(intermediate);

            // Parse the output content as JSON for the next tool's input
            current_input = serde_json::from_str(&output.content).unwrap_or_else(|_| {
                // If output is not valid JSON, wrap it as a string value
                Value::String(output.content.clone())
            });
        }

        let final_output = intermediates
            .last()
            .map(|r| r.output.clone())
            .unwrap_or_else(|| ToolOutput::success(""));

        Ok((final_output, intermediates))
    }

    /// Get the number of tools in the pipeline.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for tool piping.
///
/// Enables `tool_a.pipe(tool_b)` syntax.
pub trait ToolPipeExt: Tool + Sized + 'static {
    /// Chain this tool with another, creating a pipeline.
    fn pipe<T: Tool + 'static>(self, next: T) -> ToolPipeline {
        ToolPipeline::new().pipe(self).pipe(next)
    }
}

// Blanket implementation for all Tools that are Sized + 'static
impl<T: Tool + Sized + 'static> ToolPipeExt for T {}

/// A composed tool pipeline that implements the Tool trait itself.
///
/// This allows a pipeline to be used anywhere a single Tool is expected.
pub struct ComposedTool {
    name: String,
    description: String,
    pipeline: ToolPipeline,
}

impl ComposedTool {
    /// Create a new composed tool from a pipeline.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        pipeline: ToolPipeline,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            pipeline,
        }
    }
}

#[async_trait]
impl Tool for ComposedTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor::new(
            &self.name,
            &self.description,
            serde_json::json!({"type": "object"}),
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let (output, _intermediates) = self.pipeline.execute(input).await?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use serde_json::json;

    /// Tool A: takes input and doubles a "value" field
    struct DoubleTool;

    #[async_trait]
    impl Tool for DoubleTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("double", "Doubles the value", json!({"type": "object"}))
        }

        async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
            let value = input
                .get("value")
                .or_else(|| input.as_i64().map(|_| &input))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let result = json!({"value": value * 2});
            Ok(ToolOutput::success(result.to_string()))
        }
    }

    /// Tool B: takes input and adds 10 to "value" field
    struct AddTenTool;

    #[async_trait]
    impl Tool for AddTenTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("add_ten", "Adds 10 to the value", json!({"type": "object"}))
        }

        async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
            let value = input.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            let result = json!({"value": value + 10});
            Ok(ToolOutput::success(result.to_string()))
        }
    }

    /// Tool C: formats the value as a string
    struct FormatTool;

    #[async_trait]
    impl Tool for FormatTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("format", "Formats the value", json!({"type": "object"}))
        }

        async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
            let value = input.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(ToolOutput::success(format!("Result: {}", value)))
        }
    }

    /// Tool that always errors
    struct ErrorTool;

    #[async_trait]
    impl Tool for ErrorTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new("error_tool", "Always errors", json!({"type": "object"}))
        }

        async fn execute(&self, _input: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::error("Something went wrong"))
        }
    }

    // REQ-3.5: Unit: pipe(tool_a, tool_b) passes output of A as input to B
    #[tokio::test]
    async fn test_pipe_passes_output_as_input() {
        let pipeline = ToolPipeline::new().pipe(DoubleTool).pipe(AddTenTool);

        let (output, _intermediates) = pipeline.execute(json!({"value": 5})).await.unwrap();

        // 5 * 2 = 10, then 10 + 10 = 20
        let result: Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(result["value"], 20);
    }

    // REQ-3.5: Unit: intermediate result is captured and available for inspection
    #[tokio::test]
    async fn test_intermediate_results_captured() {
        let pipeline = ToolPipeline::new().pipe(DoubleTool).pipe(AddTenTool);

        let (_output, intermediates) = pipeline.execute(json!({"value": 5})).await.unwrap();

        assert_eq!(intermediates.len(), 2);
        assert_eq!(intermediates[0].tool_name, "double");
        assert_eq!(intermediates[1].tool_name, "add_ten");

        // First intermediate: 5 * 2 = 10
        let first_result: Value = serde_json::from_str(&intermediates[0].output.content).unwrap();
        assert_eq!(first_result["value"], 10);

        // Second intermediate: 10 + 10 = 20
        let second_result: Value = serde_json::from_str(&intermediates[1].output.content).unwrap();
        assert_eq!(second_result["value"], 20);
    }

    // REQ-3.5: Integration: chain of 3 tools produces correct final output
    #[tokio::test]
    async fn test_chain_of_three_tools() {
        let pipeline = ToolPipeline::new()
            .pipe(DoubleTool)
            .pipe(AddTenTool)
            .pipe(FormatTool);

        let (output, intermediates) = pipeline.execute(json!({"value": 7})).await.unwrap();

        // 7 * 2 = 14, 14 + 10 = 24, format = "Result: 24"
        assert_eq!(output.content, "Result: 24");
        assert_eq!(intermediates.len(), 3);
    }

    // REQ-3.5: pipe syntax via extension trait
    #[tokio::test]
    async fn test_pipe_extension_trait() {
        let pipeline = DoubleTool.pipe(AddTenTool);

        let (output, _) = pipeline.execute(json!({"value": 3})).await.unwrap();

        // 3 * 2 = 6, 6 + 10 = 16
        let result: Value = serde_json::from_str(&output.content).unwrap();
        assert_eq!(result["value"], 16);
    }

    // Test error propagation in pipeline
    #[tokio::test]
    async fn test_pipeline_error_stops_execution() {
        let pipeline = ToolPipeline::new()
            .pipe(DoubleTool)
            .pipe(ErrorTool)
            .pipe(AddTenTool);

        let result = pipeline.execute(json!({"value": 5})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("error_tool"));
    }

    // Test empty pipeline
    #[tokio::test]
    async fn test_empty_pipeline_errors() {
        let pipeline = ToolPipeline::new();
        let result = pipeline.execute(json!({})).await;
        assert!(result.is_err());
    }

    // Test ComposedTool
    #[tokio::test]
    async fn test_composed_tool() {
        let pipeline = ToolPipeline::new().pipe(DoubleTool).pipe(AddTenTool);
        let composed = ComposedTool::new("double_and_add", "Doubles then adds 10", pipeline);

        let descriptor = composed.descriptor();
        assert_eq!(descriptor.name, "double_and_add");

        let result = composed.execute(json!({"value": 4})).await.unwrap();
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["value"], 18); // 4*2=8, 8+10=18
    }
}
