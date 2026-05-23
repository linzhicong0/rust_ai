use ai_agent::{AgentBuilder, DelegateTool, DelegationWorker};
use ai_core::error::ProviderError;
use ai_core::provider::Provider;
use ai_core::tool::ToolDescriptor;
use ai_core::types::{CompletionResponse, FinishReason, Message, ModelConfig, StreamChunk, ToolCall, Usage};
use ai_memory::AgentMemory;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;

#[derive(Clone)]
struct WorkerProvider {
    name: &'static str,
    response: &'static str,
}

#[async_trait]
impl Provider for WorkerProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        _config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let task = messages
            .iter()
            .rev()
            .find_map(|message| match message.role {
                ai_core::types::Role::User => message.content.as_text().map(str::to_string),
                _ => None,
            })
            .unwrap_or_else(|| "No task provided".to_string());

        Ok(CompletionResponse {
            content: format!("{} handled: {}\n{}", self.name, task, self.response),
            tool_calls: vec![],
            usage: Usage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
            },
            finish_reason: FinishReason::Stop,
        })
    }

    fn stream(
        &self,
        _messages: Vec<Message>,
        _config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        Box::pin(futures::stream::empty())
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        Ok(texts.into_iter().map(|_| vec![0.0; 4]).collect())
    }

    fn name(&self) -> &str {
        self.name
    }
}

#[derive(Clone)]
struct ManagerProvider;

#[async_trait]
impl Provider for ManagerProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        _config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let delegated_result = messages.iter().rev().find_map(|message| match message.role {
            ai_core::types::Role::Tool => message.content.as_text().map(str::to_string),
            _ => None,
        });

        if let Some(result) = delegated_result {
            return Ok(CompletionResponse {
                content: format!("Manager summary:\n{}", result),
                tool_calls: vec![],
                usage: Usage {
                    prompt_tokens: 25,
                    completion_tokens: 15,
                    total_tokens: 40,
                },
                finish_reason: FinishReason::Stop,
            });
        }

        let task = messages
            .iter()
            .rev()
            .find_map(|message| match message.role {
                ai_core::types::Role::User => message.content.as_text().map(str::to_string),
                _ => None,
            })
            .unwrap_or_else(|| "Prepare a project summary".to_string());

        Ok(CompletionResponse {
            content: "Delegating to specialists".to_string(),
            tool_calls: vec![ToolCall {
                id: "delegate-1".to_string(),
                name: "delegate".to_string(),
                arguments: json!({
                    "task": task,
                    "agents": ["researcher", "reviewer"]
                }),
            }],
            usage: Usage {
                prompt_tokens: 18,
                completion_tokens: 8,
                total_tokens: 26,
            },
            finish_reason: FinishReason::ToolCalls,
        })
    }

    fn stream(
        &self,
        _messages: Vec<Message>,
        _config: &ModelConfig,
        _tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        Box::pin(futures::stream::empty())
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        Ok(texts.into_iter().map(|_| vec![0.0; 4]).collect())
    }

    fn name(&self) -> &str {
        "manager"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let researcher_memory = AgentMemory::new(16);
    researcher_memory.remember(
        "research-style",
        "Focus on concrete findings, risks, and dependency notes.",
    )?;
    let reviewer_memory = AgentMemory::new(16);
    reviewer_memory.remember(
        "review-style",
        "Summarize readiness, missing tests, and release concerns.",
    )?;

    let researcher = AgentBuilder::new(researcher_memory)
        .provider(WorkerProvider {
            name: "researcher",
            response: "Collected implementation facts and relevant risks.",
        })
        .name("researcher")
        .role("You are a research specialist.")
        .build()?;

    let reviewer = AgentBuilder::new(reviewer_memory)
        .provider(WorkerProvider {
            name: "reviewer",
            response: "Reviewed the plan and highlighted release-readiness concerns.",
        })
        .name("reviewer")
        .role("You are a review specialist.")
        .build()?;

    let manager_memory = AgentMemory::new(24);
    manager_memory.remember(
        "team-shape",
        "Use the researcher for fact gathering and the reviewer for release readiness.",
    )?;

    let delegate_tool = DelegateTool::parallel(
        "delegate",
        "Delegate work to specialist worker agents",
        vec![
            DelegationWorker::new("researcher", "Research specialist", researcher),
            DelegationWorker::new("reviewer", "Review specialist", reviewer),
        ],
    );

    let manager = AgentBuilder::new(manager_memory)
        .provider(ManagerProvider)
        .name("manager")
        .role("You are a manager coordinating specialist agents.")
        .tool(delegate_tool)
        .build()?;

    let output = manager
        .run("Prepare a launch brief for the Rust AI workspace.")
        .await?;

    println!("{}", output.content);
    Ok(())
}