// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent delegation support.
//!
//! This module provides functionality for agents to delegate tasks to other
//! agents, enabling hierarchical and peer-to-peer collaboration.

use ai_core::memory::Memory;
use ai_core::provider::Provider;
use ai_core::tool::{Tool, ToolDescriptor, ToolOutput};
use async_trait::async_trait;
use futures::future::join_all;
use serde_json::Value;

/// A named worker that can receive delegated tasks.
pub struct DelegationWorker<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    name: String,
    description: String,
    agent: crate::Agent<P, M>,
}

impl<P, M> DelegationWorker<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    /// Create a named worker entry for delegation pools.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: crate::Agent<P, M>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DelegationMode {
    Sequential,
    Parallel,
}

/// A tool that enables an agent to delegate to one or more worker agents.
pub struct DelegateTool<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    workers: Vec<DelegationWorker<P, M>>,
    name: String,
    description: String,
    mode: DelegationMode,
}

impl<P, M> DelegateTool<P, M>
where
    P: Provider + Send + Sync + 'static,
    M: Memory + Send + Sync + 'static,
{
    /// Create a new delegation tool backed by a single worker.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: crate::Agent<P, M>,
    ) -> Self {
        Self {
            workers: vec![DelegationWorker::new("default", "Default delegate", agent)],
            name: name.into(),
            description: description.into(),
            mode: DelegationMode::Sequential,
        }
    }

    /// Create a delegation tool backed by a named worker pool.
    pub fn pool(
        name: impl Into<String>,
        description: impl Into<String>,
        workers: Vec<DelegationWorker<P, M>>,
    ) -> Self {
        Self {
            workers,
            name: name.into(),
            description: description.into(),
            mode: DelegationMode::Sequential,
        }
    }

    /// Create a delegation tool that fans out to workers in parallel.
    pub fn parallel(
        name: impl Into<String>,
        description: impl Into<String>,
        workers: Vec<DelegationWorker<P, M>>,
    ) -> Self {
        Self {
            workers,
            name: name.into(),
            description: description.into(),
            mode: DelegationMode::Parallel,
        }
    }

    fn selected_workers<'a>(
        &'a self,
        input: &Value,
    ) -> Result<Vec<&'a DelegationWorker<P, M>>, ai_core::error::ToolError> {
        if self.workers.is_empty() {
            return Err(ai_core::error::ToolError::Execution(
                "delegate tool has no worker agents".to_string(),
            ));
        }

        let mut requested_names = Vec::new();
        if let Some(agent_name) = input.get("agent").and_then(Value::as_str) {
            requested_names.push(agent_name.to_string());
        }
        if let Some(agent_names) = input.get("agents").and_then(Value::as_array) {
            for value in agent_names {
                let name = value.as_str().ok_or_else(|| {
                    ai_core::error::ToolError::InvalidInput(
                        "agents must be an array of strings".to_string(),
                    )
                })?;
                requested_names.push(name.to_string());
            }
        }

        if requested_names.is_empty() {
            if self.workers.len() == 1 {
                return Ok(vec![&self.workers[0]]);
            }
            return Ok(self.workers.iter().collect());
        }

        let mut selected = Vec::new();
        for requested in requested_names {
            let worker = self
                .workers
                .iter()
                .find(|worker| worker.name == requested)
                .ok_or_else(|| {
                    ai_core::error::ToolError::InvalidInput(format!(
                        "unknown delegate agent: {requested}"
                    ))
                })?;
            selected.push(worker);
        }

        Ok(selected)
    }

    async fn run_worker(worker: &DelegationWorker<P, M>, task: String) -> ToolOutput {
        match worker.agent.run(task).await {
            Ok(output) => ToolOutput::success(output.content),
            Err(error) => ToolOutput::error(format!("Delegation failed: {error}")),
        }
    }

    fn aggregate_results(&self, results: Vec<(&DelegationWorker<P, M>, ToolOutput)>) -> ToolOutput {
        if results.len() == 1 {
            return results.into_iter().next().unwrap().1;
        }

        let mut is_error = false;
        let mut lines = Vec::new();
        for (worker, output) in results {
            if output.is_error {
                is_error = true;
            }
            lines.push(format!(
                "{} ({})\n{}",
                worker.name, worker.description, output.content
            ));
        }

        ToolOutput {
            content: lines.join("\n\n"),
            is_error,
        }
    }
}

#[async_trait]
impl<P, M> Tool for DelegateTool<P, M>
where
    P: Provider + Send + Sync + 'static,
    M: Memory + Send + Sync + 'static,
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
                    },
                    "agent": {
                        "type": "string",
                        "description": "Optional worker name for single-agent delegation"
                    },
                    "agents": {
                        "type": "array",
                        "description": "Optional worker names for pooled delegation",
                        "items": { "type": "string" }
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
        let workers = self.selected_workers(&input)?;

        let results = if self.mode == DelegationMode::Parallel && workers.len() > 1 {
            let futures = workers.iter().map(|worker| async move {
                (*worker, Self::run_worker(worker, task.to_string()).await)
            });
            join_all(futures).await
        } else {
            let mut results = Vec::with_capacity(workers.len());
            for worker in workers {
                results.push((worker, Self::run_worker(worker, task.to_string()).await));
            }
            results
        };

        Ok(self.aggregate_results(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::tool::{Tool, ToolDescriptor};
    use ai_core::types::{
        CompletionResponse, FinishReason, Message, ModelConfig, StreamChunk, Usage,
    };
    use ai_memory::InMemoryMemory;

    struct MockProvider {
        response: String,
    }

    #[async_trait::async_trait]
    impl ai_core::Provider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> Result<CompletionResponse, ai_core::error::ProviderError> {
            Ok(CompletionResponse {
                content: self.response.clone(),
                tool_calls: vec![],
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                finish_reason: FinishReason::Stop,
            })
        }

        fn stream(
            &self,
            _messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> futures::stream::BoxStream<'static, Result<StreamChunk, ai_core::error::ProviderError>>
        {
            Box::pin(futures::stream::empty())
        }

        async fn embed(
            &self,
            _texts: Vec<String>,
        ) -> Result<Vec<Vec<f32>>, ai_core::error::ProviderError> {
            Ok(vec![vec![0.0; 10]])
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn worker_agent(response: &str) -> crate::Agent<MockProvider, InMemoryMemory> {
        crate::AgentBuilder::new(InMemoryMemory::new(100))
            .provider(MockProvider {
                response: response.to_string(),
            })
            .build()
            .unwrap()
    }

    #[test]
    fn test_delegate_tool_creation() {
        let delegate_tool = DelegateTool::new(
            "delegate_to_helper",
            "A helpful assistant agent",
            worker_agent("Delegated response"),
        );

        assert_eq!(delegate_tool.name, "delegate_to_helper");
        assert_eq!(delegate_tool.description, "A helpful assistant agent");
        assert_eq!(delegate_tool.workers.len(), 1);
    }

    #[test]
    fn test_delegate_tool_descriptor() {
        let delegate_tool = DelegateTool::new("delegate", "Description", worker_agent("Response"));

        let descriptor = delegate_tool.descriptor();

        assert_eq!(descriptor.name, "delegate");
        assert_eq!(descriptor.description, "Description");
        assert_eq!(descriptor.input_schema["type"], "object");
        assert!(descriptor.input_schema["properties"]["task"].is_object());
        assert!(descriptor.input_schema["properties"]["agent"].is_object());
        assert!(descriptor.input_schema["properties"]["agents"].is_object());
        assert_eq!(descriptor.input_schema["required"][0], "task");
    }

    #[tokio::test]
    async fn test_delegate_tool_execute_success() {
        let delegate_tool = DelegateTool::new(
            "delegate",
            "Worker agent",
            worker_agent("Task completed successfully"),
        );

        let input = serde_json::json!({"task": "Do something"});
        let result = delegate_tool.execute(input).await.unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Task completed successfully"));
    }

    #[tokio::test]
    async fn test_delegate_tool_execute_missing_task() {
        let delegate_tool = DelegateTool::new("delegate", "Worker agent", worker_agent("Response"));

        let input = serde_json::json!({});
        let result = delegate_tool.execute(input).await;

        assert!(result.is_err());
        match result {
            Err(ai_core::error::ToolError::InvalidInput(msg)) => {
                assert!(msg.contains("task required"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_delegate_tool_selects_named_worker() {
        let delegate_tool = DelegateTool::pool(
            "delegate",
            "Worker pool",
            vec![
                DelegationWorker::new(
                    "researcher",
                    "Research specialist",
                    worker_agent("Research findings"),
                ),
                DelegationWorker::new(
                    "reviewer",
                    "Review specialist",
                    worker_agent("Review complete"),
                ),
            ],
        );

        let result = delegate_tool
            .execute(serde_json::json!({"task": "Investigate issue", "agent": "reviewer"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content, "Review complete");
    }

    #[tokio::test]
    async fn test_parallel_delegate_tool_aggregates_worker_results() {
        let delegate_tool = DelegateTool::parallel(
            "delegate",
            "Parallel worker pool",
            vec![
                DelegationWorker::new(
                    "researcher",
                    "Research specialist",
                    worker_agent("Collected sources"),
                ),
                DelegationWorker::new(
                    "reviewer",
                    "Review specialist",
                    worker_agent("Reviewed draft"),
                ),
            ],
        );

        let result = delegate_tool
            .execute(serde_json::json!({"task": "Prepare a summary"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("researcher (Research specialist)"));
        assert!(result.content.contains("Collected sources"));
        assert!(result.content.contains("reviewer (Review specialist)"));
        assert!(result.content.contains("Reviewed draft"));
    }
}
