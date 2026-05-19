// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent delegation support.
//!
//! This module provides functionality for agents to delegate tasks to other
//! agents, enabling hierarchical and peer-to-peer collaboration.

use ai_core::error::AgentError;
use ai_core::tool::{Tool, ToolDescriptor, ToolOutput};
use async_trait::async_trait;
use serde_json::Value;

/// A tool that enables an agent to delegate to another agent.
///
/// When an agent calls this tool, the specified agent is invoked with
/// the given input, and the result is returned as the tool output.
pub struct DelegateTool<P: ai_core::Provider + 'static> {
    agent: crate::Agent<P, ai_memory::InMemoryMemory>,
    name: String,
    description: String,
}

impl<P> DelegateTool<P>
where
    P: ai_core::Provider + Clone + Send + Sync + 'static,
{
    /// Create a new delegation tool.
    ///
    /// # Arguments
    ///
    /// * `name` — Name for this tool (e.g., "delegate_to_researcher")
    /// * `description` — Description of what this agent does
    /// * `agent` — The agent to delegate to
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: crate::Agent<P, ai_memory::InMemoryMemory>,
    ) -> Self {
        Self {
            agent,
            name: name.into(),
            description: description.into(),
        }
    }
}

#[async_trait::async_trait]
impl<P> Tool for DelegateTool<P>
where
    P: ai_core::Provider + Clone + Send + Sync + 'static,
{
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to delegate"
                    }
                },
                "required": ["task"]
            }),
            output_schema: None,
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ai_core::error::ToolError> {
        let task = input["task"]
            .as_str()
            .ok_or_else(|| ai_core::error::ToolError::InvalidInput("task required".to_string()))?;

        match self.agent.run(task).await {
            Ok(output) => Ok(ToolOutput::success(output.content)),
            Err(e) => Ok(ToolOutput::error(format!("Delegation failed: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::tool::ToolDescriptor;
    use ai_memory::InMemoryMemory;

    // Mock provider for testing
    struct MockProvider {
        response: String,
    }

    #[async_trait::async_trait]
    impl ai_core::Provider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<ai_core::types::Message>,
            _config: &ai_core::types::ModelConfig,
            _tools: &[ai_core::tool::ToolDescriptor],
        ) -> Result<ai_core::types::CompletionResponse, ai_core::error::ProviderError> {
            Ok(ai_core::types::CompletionResponse {
                content: self.response.clone(),
                tool_calls: vec![],
                usage: ai_core::types::Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                finish_reason: ai_core::types::FinishReason::Stop,
            })
        }

        fn stream(
            &self,
            _messages: Vec<ai_core::types::Message>,
            _config: &ai_core::types::ModelConfig,
            _tools: &[ai_core::tool::ToolDescriptor],
        ) -> futures::stream::BoxStream<'static, Result<ai_core::types::StreamChunk, ai_core::error::ProviderError>> {
            Box::pin(futures::stream::empty())
        }

        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ai_core::error::ProviderError> {
            Ok(vec![vec![0.0; 10]])
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    impl Clone for MockProvider {
        fn clone(&self) -> Self {
            Self {
                response: self.response.clone(),
            }
        }
    }

    #[test]
    fn test_delegate_tool_creation() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider {
            response: "Delegated response".to_string(),
        };

        let agent = crate::AgentBuilder::new(memory)
            .provider(provider)
            .role("You are a helper")
            .build()
            .unwrap();

        let delegate_tool = DelegateTool::new(
            "delegate_to_helper",
            "A helpful assistant agent",
            agent,
        );

        assert_eq!(delegate_tool.name, "delegate_to_helper");
        assert_eq!(delegate_tool.description, "A helpful assistant agent");
    }

    #[test]
    fn test_delegate_tool_descriptor() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider {
            response: "Response".to_string(),
        };

        let agent = crate::AgentBuilder::new(memory)
            .provider(provider)
            .build()
            .unwrap();

        let delegate_tool = DelegateTool::new(
            "delegate",
            "Description",
            agent,
        );

        let descriptor = delegate_tool.descriptor();

        assert_eq!(descriptor.name, "delegate");
        assert_eq!(descriptor.description, "Description");

        // Check the input schema
        let schema = &descriptor.input_schema;
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["task"].is_object());
        assert!(schema["required"].is_array());
        assert_eq!(schema["required"][0], "task");
    }

    #[tokio::test]
    async fn test_delegate_tool_execute_success() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider {
            response: "Task completed successfully".to_string(),
        };

        let agent = crate::AgentBuilder::new(memory)
            .provider(provider)
            .role("You are a worker")
            .build()
            .unwrap();

        let delegate_tool = DelegateTool::new(
            "delegate",
            "Worker agent",
            agent,
        );

        let input = serde_json::json!({"task": "Do something"});
        let result = delegate_tool.execute(input).await.unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Task completed successfully"));
    }

    #[tokio::test]
    async fn test_delegate_tool_execute_missing_task() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider {
            response: "Response".to_string(),
        };

        let agent = crate::AgentBuilder::new(memory)
            .provider(provider)
            .build()
            .unwrap();

        let delegate_tool = DelegateTool::new(
            "delegate",
            "Worker agent",
            agent,
        );

        let input = serde_json::json!({}); // Missing "task" field
        let result = delegate_tool.execute(input).await;

        assert!(result.is_err());
        match result {
            Err(ai_core::error::ToolError::InvalidInput(msg)) => {
                assert!(msg.contains("task required"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }
}
