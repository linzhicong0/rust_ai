// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent system with ReAct (Reasoning + Acting) loop.
//!
//! This module provides the [`Agent`] type which orchestrates LLM interactions
//! with tools and memory to accomplish tasks through iterative reasoning.

use std::sync::Arc;

use futures::stream::{BoxStream, StreamExt};
use serde_json::json;
use tracing::Instrument;

use ai_core::error::AgentError;
use ai_core::memory::{Memory, MemoryEntry};
use ai_core::provider::Provider;
use ai_core::tool::{Tool, ToolDescriptor};
use ai_core::types::{
    AgentEvent, AgentOutput, CompletionResponse, Content, FinishReason, Message, ModelConfig, Role,
    ToolCall, Usage,
};
use ai_core::{agent_scope, new_request_id, request_scope, CostTracker, GLOBAL_SCOPE};

// AgentInner is defined below in this module

/// Internal state of an Agent.
pub struct AgentInner<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    pub(crate) provider: P,
    pub(crate) memory: M,
    pub(crate) name: Option<String>,
    pub(crate) role: Option<String>,
    pub(crate) goal: Option<String>,
    pub(crate) backstory: Option<String>,
    pub(crate) tools: Vec<Box<dyn Tool>>,
    pub(crate) model_config: ModelConfig,
    pub(crate) max_iterations: u32,
    pub(crate) cost_tracker: CostTracker,
}

// Implement Clone for AgentInner where both P and M are Clone
// Note: Tools are not cloned, only the references
impl<P, M> Clone for AgentInner<P, M>
where
    P: Provider + Clone,
    M: Memory + Clone,
{
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            memory: self.memory.clone(),
            name: self.name.clone(),
            role: self.role.clone(),
            goal: self.goal.clone(),
            backstory: self.backstory.clone(),
            tools: Vec::new(), // Tools need to be re-added after clone
            model_config: self.model_config.clone(),
            max_iterations: self.max_iterations,
            cost_tracker: self.cost_tracker.clone(),
        }
    }
}

/// An AI agent that can reason, act, and observe using tools.
///
/// The agent implements the ReAct (Reasoning + Acting) loop:
/// 1. **Reason** — Generate a response or tool call based on context
/// 2. **Act** — Execute any requested tool calls
/// 3. **Observe** — Incorporate tool results into context
/// 4. **Repeat** — Continue until a final answer is produced
///
/// ## Type Parameters
///
/// * `P` — The LLM provider to use
/// * `M` — The memory backend for conversation history
///
/// ## Example
///
/// ```rust,no_run
/// use ai_agent::{Agent, AgentBuilder};
/// use ai_memory::InMemoryMemory;
/// use ai_provider_openai::OpenAiProvider;
/// use ai_core::{Tool, ToolDescriptor, ToolOutput};
/// use ai_core::types::Message;
/// # use async_trait::async_trait;
/// # struct MyTool;
/// # #[async_trait::async_trait]
/// # impl Tool for MyTool {
/// #     fn descriptor(&self) -> ToolDescriptor { todo!() }
/// #     async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ai_core::error::ToolError> { todo!() }
/// # }
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = InMemoryMemory::new(100);
/// let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
///
/// let agent = AgentBuilder::new(memory)
///     .provider(provider)
///     .role("You are a helpful assistant.")
///     .tool(MyTool)
///     .build()?;
///
/// let response = agent.run("What's the weather like?").await?;
/// println!("{}", response.content);
/// # Ok(())
/// # }
/// ```
pub struct Agent<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    pub(crate) inner: Arc<AgentInner<P, M>>,
}

impl<P, M> Agent<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    /// Get the provider this agent uses.
    pub fn provider(&self) -> &P {
        &self.inner.provider
    }

    /// Get the memory this agent uses.
    pub fn memory(&self) -> &M {
        &self.inner.memory
    }

    /// Get the agent name used for tracking.
    pub fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    /// Get the agent's role description.
    pub fn role(&self) -> Option<&str> {
        self.inner.role.as_deref()
    }

    /// Get the agent's goal.
    pub fn goal(&self) -> Option<&str> {
        self.inner.goal.as_deref()
    }

    /// Get the agent's backstory.
    pub fn backstory(&self) -> Option<&str> {
        self.inner.backstory.as_deref()
    }

    /// Get the tool descriptors for all registered tools.
    pub fn tool_descriptors(&self) -> Vec<ToolDescriptor> {
        self.inner.tools.iter().map(|t| t.descriptor()).collect()
    }

    /// Get the model config for this agent.
    pub fn model_config(&self) -> &ModelConfig {
        &self.inner.model_config
    }

    /// Get the maximum iterations before giving up.
    pub fn max_iterations(&self) -> u32 {
        self.inner.max_iterations
    }

    /// Get the cost tracker for this agent.
    pub fn cost_tracker(&self) -> &CostTracker {
        &self.inner.cost_tracker
    }

    /// Run the agent with the given input.
    ///
    /// This executes the ReAct loop until:
    /// - The agent produces a final response (no tool calls)
    /// - Maximum iterations are reached
    /// - An error occurs
    ///
    /// # Arguments
    ///
    /// * `input` — The user input or task description
    ///
    /// # Returns
    ///
    /// The agent's final response.
    pub async fn run(&self, input: impl Into<String>) -> Result<AgentOutput, AgentError> {
        let input = input.into();
        let mut messages = self.build_initial_messages(&input).await?;
        self.store_user_input(&input).await;
        let agent_scope_name = self.agent_scope_name();
        let mut total_usage = Usage::default();
        let mut total_cost = 0.0;
        let mut tracked_scopes = vec![agent_scope_name.clone(), GLOBAL_SCOPE.to_string()];

        for iteration in 0..self.inner.max_iterations {
            let agent_name = self.name().unwrap_or("unnamed");
            let turn_span = tracing::info_span!(
                "agent_turn",
                agent = agent_name,
                iteration,
            );

            let response = async {
                tracing::debug!(agent = agent_name, iteration, "Agent ReAct loop");

                self.inner
                    .provider
                    .complete(
                        messages.clone(),
                        &self.inner.model_config,
                        &self.tool_descriptors(),
                    )
                    .instrument(tracing::debug_span!(
                        "llm_call",
                        agent = agent_name,
                        iteration,
                        model = %self.inner.model_config.model,
                    ))
                    .await
            }
            .instrument(turn_span)
            .await?;

            let request_scope_name = self.track_response_cost(&response, &agent_scope_name).await;
            if !tracked_scopes
                .iter()
                .any(|scope| scope == &request_scope_name)
            {
                tracked_scopes.push(request_scope_name);
            }
            total_usage.prompt_tokens += response.usage.prompt_tokens;
            total_usage.completion_tokens += response.usage.completion_tokens;
            total_usage.total_tokens += response.usage.total_tokens;
            total_cost += self
                .inner
                .cost_tracker
                .estimate_cost(&self.inner.model_config.model, &response.usage);

            // Store assistant response in memory
            self.store_assistant_response(&response, iteration).await;

            if response.tool_calls.is_empty() {
                // No tool calls — agent is done
                return Ok(AgentOutput {
                    content: response.content,
                    usage: total_usage,
                    estimated_cost: total_cost,
                    tracked_scopes,
                });
            }

            // Execute tool calls
            for tool_call in &response.tool_calls {
                let tool_result = self.execute_tool_call(tool_call).await?;
                self.store_tool_result(tool_call, &tool_result).await;

                // Add tool result to messages
                messages.push(Message {
                    role: Role::Tool,
                    content: Content::Text(tool_result),
                });
            }
        }

        Err(AgentError::MaxIterationsExceeded)
    }

    /// Run the agent with streaming output.
    ///
    /// Returns a stream of [`AgentEvent`] values representing:
    /// - Text chunks as they're generated
    /// - Tool calls when initiated
    /// - Tool results when received
    ///
    /// # Arguments
    ///
    /// * `input` — The user input or task description
    pub fn stream(
        &self,
        input: impl Into<String>,
    ) -> BoxStream<'static, Result<AgentEvent, AgentError>> {
        let input = input.into();
        let agent = self.clone();

        Box::pin(async_stream::try_stream! {
            let mut messages = agent.build_initial_messages(&input).await?;
            agent.store_user_input(&input).await;

            for iteration in 0..agent.inner.max_iterations {
                let agent_name = agent.name().unwrap_or("unnamed").to_string();
                let stream_span = tracing::info_span!(
                    "agent_turn_stream",
                    agent = %agent_name,
                    iteration,
                    model = %agent.inner.model_config.model,
                );
                let mut stream = agent.inner.provider.stream(
                    messages.clone(),
                    &agent.inner.model_config,
                    &agent.tool_descriptors(),
                );

                let mut content_buffer = String::new();
                let tool_calls_buffer: Vec<ToolCall> = Vec::new();

                while let Some(chunk_result) = stream.next().instrument(stream_span.clone()).await {
                    let chunk = chunk_result?;

                    if let Some(delta) = chunk.delta {
                        content_buffer.push_str(&delta);
                        yield AgentEvent::Text(delta);
                    }

                    if let Some(finish_reason) = chunk.finish_reason {
                        // Store in memory
                        let entry = MemoryEntry::new(Role::Assistant, &content_buffer);
                        agent.inner.memory.add(entry).await?;

                        if !matches!(finish_reason, FinishReason::ToolCalls) {
                            // Done without tool calls
                            return;
                        }
                    }
                }

                // Handle tool calls
                for tool_call in &tool_calls_buffer {
                    yield AgentEvent::ToolCall(tool_call.clone());

                    let result = agent.execute_tool_call(tool_call).await?;
                    agent.store_tool_result(tool_call, &result).await;
                    yield AgentEvent::ToolResult {
                        call_id: tool_call.id.clone(),
                        content: result.clone(),
                    };

                    messages.push(Message {
                        role: Role::Tool,
                        content: Content::Text(result),
                    });
                }
            }
        })
    }

    /// Build the initial message list with system prompt and user input.
    async fn build_initial_messages(&self, input: &str) -> Result<Vec<Message>, AgentError> {
        let mut messages = Vec::new();

        // Add system prompt if configured
        if let Some(system_prompt) = self.build_system_prompt() {
            messages.push(Message::system(system_prompt));
        }

        let history = self.inner.memory.get(None).await.unwrap_or_default();
        if let Some(relevant_memory_prompt) = self.relevant_memory_prompt(input, &history).await {
            messages.push(Message::system(relevant_memory_prompt));
        }

        // Get conversation history from memory
        for entry in history {
            messages.push(Message {
                role: entry.role,
                content: Content::Text(entry.content),
            });
        }

        // Add current user input
        messages.push(Message::user(input));

        Ok(messages)
    }

    async fn relevant_memory_prompt(&self, input: &str, history: &[MemoryEntry]) -> Option<String> {
        let relevant_memories = self.inner.memory.search(input, 3).await.ok()?;
        let filtered: Vec<_> = relevant_memories
            .into_iter()
            .filter(|candidate| {
                !history.iter().any(|entry| {
                    std::mem::discriminant(&entry.role) == std::mem::discriminant(&candidate.role)
                        && entry.content == candidate.content
                })
            })
            .collect();

        if filtered.is_empty() {
            return None;
        }

        let mut prompt = String::from("Relevant memory for this task:\n");
        for entry in filtered {
            let label = entry
                .metadata
                .get("knowledge_key")
                .and_then(|value| value.as_str())
                .map(|value| format!("knowledge:{value}"))
                .unwrap_or_else(|| Self::role_label(&entry.role).to_string());
            prompt.push_str("- ");
            prompt.push_str(&label);
            prompt.push_str(": ");
            prompt.push_str(&entry.content);
            prompt.push('\n');
        }

        Some(prompt.trim_end().to_string())
    }

    /// Build the system prompt from agent configuration.
    fn build_system_prompt(&self) -> Option<String> {
        let parts = [
            self.inner.role.as_ref(),
            self.inner.goal.as_ref(),
            self.inner.backstory.as_ref(),
        ];

        let prompt: Vec<_> = parts.into_iter().flatten().cloned().collect();

        if prompt.is_empty() {
            None
        } else {
            Some(prompt.join("\n\n"))
        }
    }

    fn role_label(role: &Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    async fn store_user_input(&self, input: &str) {
        let entry = MemoryEntry::user(input).with_metadata("memory_kind", json!("conversation"));
        let _ = self.inner.memory.add(entry).await;
    }

    /// Store the assistant's response in memory.
    async fn store_assistant_response(&self, response: &CompletionResponse, iteration: u32) {
        let entry = MemoryEntry::assistant(&response.content)
            .with_metadata("memory_kind", json!("conversation"))
            .with_metadata("iteration", json!(iteration));
        let _ = self.inner.memory.add(entry).await;
    }

    async fn store_tool_result(&self, tool_call: &ToolCall, result: &str) {
        let entry = MemoryEntry::new(Role::Tool, result)
            .with_metadata("memory_kind", json!("tool_result"))
            .with_metadata("tool_call_id", json!(tool_call.id.clone()))
            .with_metadata("tool_name", json!(tool_call.name.clone()));
        let _ = self.inner.memory.add(entry).await;
    }

    fn agent_scope_name(&self) -> String {
        let name = self
            .inner
            .name
            .as_deref()
            .or(self.inner.role.as_deref())
            .unwrap_or("unnamed");
        agent_scope(name)
    }

    async fn track_response_cost(
        &self,
        response: &CompletionResponse,
        agent_scope_name: &str,
    ) -> String {
        let request_id = new_request_id();
        let request_scope_name = request_scope(&request_id);

        self.inner
            .cost_tracker
            .record_many(
                [
                    request_scope_name.clone(),
                    agent_scope_name.to_string(),
                    GLOBAL_SCOPE.to_string(),
                ],
                &self.inner.model_config.model,
                &response.usage,
            )
            .await;

        request_scope_name
    }

    /// Execute a single tool call.
    async fn execute_tool_call(&self, tool_call: &ToolCall) -> Result<String, AgentError> {
        let tool = self
            .inner
            .tools
            .iter()
            .find(|t| t.descriptor().name == tool_call.name)
            .ok_or_else(|| AgentError::ToolNotFound(tool_call.name.clone()))?;

        let output = tool
            .execute(tool_call.arguments.clone())
            .instrument(tracing::info_span!(
                "tool_execution",
                agent = self.name().unwrap_or("unnamed"),
                tool = %tool_call.name,
                call_id = %tool_call.id,
            ))
            .await?;

        Ok(if output.is_error {
            format!("Error: {}", output.content)
        } else {
            output.content
        })
    }

    /// Find a tool by name.
    pub fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.inner
            .tools
            .iter()
            .find(|t| t.descriptor().name == name)
            .map(|t| t.as_ref())
    }

    /// Add a tool to this agent.
    pub fn with_tool<T: Tool + 'static>(self, tool: T) -> Agent<P, M>
    where
        P: Provider + Clone,
        M: Memory + Clone,
    {
        let mut inner = (*self.inner).clone();
        inner.tools.push(Box::new(tool));
        Agent {
            inner: Arc::new(inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::builder::AgentBuilder;
    use super::*;
    use ai_core::error::ProviderError;
    use ai_core::tool::{ToolDescriptor, ToolOutput};
    use ai_core::types::{FinishReason, Message, Role, StreamChunk, ToolCall, Usage};
    use ai_memory::{AgentMemory, InMemoryMemory};
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    // Mock tool for testing
    struct MockTool {
        name: &'static str,
        response: String,
    }

    #[async_trait::async_trait]
    impl Tool for MockTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor::new(
                self.name,
                format!("A mock tool for {}", self.name),
                json!({"type": "object"}),
            )
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
        ) -> Result<ToolOutput, ai_core::error::ToolError> {
            Ok(ToolOutput::success(self.response.clone()))
        }
    }

    // Mock provider for testing
    #[derive(Clone)]
    struct MockProvider {
        response_content: String,
        tool_calls: Vec<ToolCall>,
    }

    impl MockProvider {
        fn new_response(content: &str) -> Self {
            Self {
                response_content: content.to_string(),
                tool_calls: Vec::new(),
            }
        }

        fn new_with_tool_call(tool_name: &str, tool_id: &str) -> Self {
            Self {
                response_content: "Let me call a tool.".to_string(),
                tool_calls: vec![ToolCall {
                    id: tool_id.to_string(),
                    name: tool_name.to_string(),
                    arguments: json!({}),
                }],
            }
        }
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(CompletionResponse {
                content: self.response_content.clone(),
                tool_calls: self.tool_calls.clone(),
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                finish_reason: if self.tool_calls.is_empty() {
                    FinishReason::Stop
                } else {
                    FinishReason::ToolCalls
                },
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

        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
            Ok(vec![vec![0.0; 10]])
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn create_test_agent() -> Agent<MockProvider, InMemoryMemory> {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider::new_response("Hello! How can I help?");

        AgentBuilder::new(memory)
            .provider(provider)
            .name("helper")
            .role("You are a helpful assistant.")
            .goal("Help users with their questions.")
            .backstory("You are an AI with general knowledge.")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_agent_properties() {
        let agent = create_test_agent();

        assert_eq!(agent.role(), Some("You are a helpful assistant."));
        assert_eq!(agent.goal(), Some("Help users with their questions."));
        assert_eq!(
            agent.backstory(),
            Some("You are an AI with general knowledge.")
        );
        assert_eq!(agent.name(), Some("helper"));
        assert_eq!(agent.max_iterations(), 10);
        assert_eq!(agent.model_config().model, "gpt-4");
    }

    #[tokio::test]
    async fn test_agent_tool_descriptors() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider::new_response("Hello!");

        let tool = MockTool {
            name: "test_tool",
            response: "Tool executed".to_string(),
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .tool(tool)
            .build()
            .unwrap();

        let descriptors = agent.tool_descriptors();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].name, "test_tool");
    }

    #[tokio::test]
    async fn test_agent_find_tool() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider::new_response("Hello!");

        let tool = MockTool {
            name: "search_tool",
            response: "Found results".to_string(),
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .tool(tool)
            .build()
            .unwrap();

        let found = agent.find_tool("search_tool");
        assert!(found.is_some());
        assert_eq!(found.unwrap().descriptor().name, "search_tool");

        let not_found = agent.find_tool("nonexistent");
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_agent_run_simple_response() {
        let agent = create_test_agent();
        let result = agent.run("Hello!").await.unwrap();

        assert_eq!(result.content, "Hello! How can I help?");
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 5);
        assert!(result.estimated_cost > 0.0);
        assert!(result.tracked_scopes.contains(&"agent:helper".to_string()));
        assert!(result.tracked_scopes.contains(&GLOBAL_SCOPE.to_string()));

        let memory_entries = agent.memory().get(None).await.unwrap();
        assert_eq!(memory_entries.len(), 2);
        assert!(matches!(memory_entries[0].role, Role::User));
        assert!(matches!(memory_entries[1].role, Role::Assistant));
    }

    #[derive(Clone)]
    struct InspectingProvider {
        seen_messages: Arc<Mutex<Vec<Message>>>,
    }

    #[async_trait::async_trait]
    impl Provider for InspectingProvider {
        async fn complete(
            &self,
            messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> Result<CompletionResponse, ProviderError> {
            *self.seen_messages.lock().unwrap() = messages;
            Ok(CompletionResponse {
                content: "Memory-aware response".to_string(),
                tool_calls: vec![],
                usage: Usage {
                    prompt_tokens: 12,
                    completion_tokens: 6,
                    total_tokens: 18,
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

        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
            Ok(vec![vec![0.0; 8]])
        }

        fn name(&self) -> &str {
            "inspecting"
        }
    }

    #[tokio::test]
    async fn test_agent_injects_relevant_memory_prompt() {
        let memory = AgentMemory::new(10);
        memory
            .remember(
                "favorite_language",
                "The user prefers Rust for systems and CLI work.",
            )
            .unwrap();

        let seen_messages = Arc::new(Mutex::new(Vec::new()));
        let provider = InspectingProvider {
            seen_messages: seen_messages.clone(),
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .role("You are a helpful assistant.")
            .build()
            .unwrap();

        agent
            .run("What language does the user usually prefer for systems work?")
            .await
            .unwrap();

        let messages = seen_messages.lock().unwrap().clone();
        assert!(messages.iter().any(|message| {
            matches!(message.role, Role::System)
                && message
                    .content
                    .as_text()
                    .unwrap_or_default()
                    .contains("favorite_language")
        }));
    }

    #[tokio::test]
    async fn test_agent_max_iterations() {
        let memory = InMemoryMemory::new(100);
        // Provider that always returns tool calls
        let provider = MockProvider::new_with_tool_call("loop_tool", "tool_123");

        let tool = MockTool {
            name: "loop_tool",
            response: "Tool result".to_string(),
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .tool(tool)
            .max_iterations(2)
            .build()
            .unwrap();

        let result = agent.run("Test").await;

        match result {
            Err(AgentError::MaxIterationsExceeded) => {
                // Expected outcome
            }
            _ => panic!("Expected MaxIterationsExceeded error"),
        }
    }

    #[tokio::test]
    async fn test_agent_with_tool() {
        let memory = InMemoryMemory::new(100);

        let tool = MockTool {
            name: "weather",
            response: "72°F and sunny".to_string(),
        };

        // First call: tool request, second call: final response
        let provider_calls = std::sync::Arc::new(std::sync::Mutex::new(0));
        let provider_calls_clone = provider_calls.clone();

        let provider = {
            let calls = provider_calls_clone;
            struct CountingProvider {
                calls: std::sync::Arc<std::sync::Mutex<usize>>,
            }
            #[async_trait::async_trait]
            impl Provider for CountingProvider {
                async fn complete(
                    &self,
                    _messages: Vec<Message>,
                    _config: &ModelConfig,
                    _tools: &[ToolDescriptor],
                ) -> Result<CompletionResponse, ProviderError> {
                    let mut count = self.calls.lock().unwrap();
                    *count += 1;
                    if *count == 1 {
                        Ok(CompletionResponse {
                            content: "I'll check the weather.".to_string(),
                            tool_calls: vec![ToolCall {
                                id: "tool_1".to_string(),
                                name: "weather".to_string(),
                                arguments: json!({"location": "SF"}),
                            }],
                            usage: Usage {
                                prompt_tokens: 10,
                                completion_tokens: 5,
                                total_tokens: 15,
                            },
                            finish_reason: FinishReason::ToolCalls,
                        })
                    } else {
                        Ok(CompletionResponse {
                            content: "The weather in SF is 72°F and sunny.".to_string(),
                            tool_calls: vec![],
                            usage: Usage {
                                prompt_tokens: 20,
                                completion_tokens: 10,
                                total_tokens: 30,
                            },
                            finish_reason: FinishReason::Stop,
                        })
                    }
                }
                fn stream(
                    &self,
                    _messages: Vec<Message>,
                    _config: &ModelConfig,
                    _tools: &[ToolDescriptor],
                ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
                    Box::pin(futures::stream::empty())
                }
                async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
                    Ok(vec![vec![0.0; 10]])
                }
                fn name(&self) -> &str {
                    "counting"
                }
            }
            CountingProvider { calls }
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .tool(tool)
            .max_iterations(5)
            .build()
            .unwrap();

        let result = agent.run("What's the weather in SF?").await.unwrap();

        assert!(result.content.contains("72°F"));
        assert_eq!(result.usage.prompt_tokens, 30);
        assert_eq!(result.usage.completion_tokens, 15);
        assert_eq!(*provider_calls.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn test_agent_cost_tracker_records_agent_and_global_scopes() {
        let agent = create_test_agent();

        let result = agent.run("Track this").await.unwrap();

        let agent_snapshot = agent.cost_tracker().get("agent:helper").await;
        assert_eq!(agent_snapshot.request_count, 1);
        assert_eq!(agent_snapshot.prompt_tokens, 10);

        let global_snapshot = agent.cost_tracker().get(GLOBAL_SCOPE).await;
        assert_eq!(global_snapshot.request_count, 1);
        assert_eq!(global_snapshot.completion_tokens, 5);

        assert!(result
            .tracked_scopes
            .iter()
            .any(|scope| scope.starts_with("request:")));
    }

    #[tokio::test]
    async fn test_agent_build_system_prompt() {
        let memory = InMemoryMemory::new(100);
        let provider = MockProvider::new_response("Hello!");

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .role("You are a researcher.")
            .goal("Find accurate information.")
            .backstory("You have access to academic databases.")
            .build()
            .unwrap();

        // The system prompt should be built from role, goal, and backstory
        let prompt = agent.build_system_prompt();
        assert!(prompt.is_some());
        let prompt = prompt.unwrap();
        assert!(prompt.contains("You are a researcher."));
        assert!(prompt.contains("Find accurate information."));
        assert!(prompt.contains("You have access to academic databases."));
    }

    #[tokio::test]
    async fn test_agent_clone() {
        let agent = create_test_agent();
        let cloned = agent.clone();

        assert_eq!(agent.role(), cloned.role());
        assert_eq!(agent.goal(), cloned.goal());
        assert_eq!(agent.max_iterations(), cloned.max_iterations());
    }

    #[tokio::test]
    async fn test_agent_with_tool_adds_tool() {
        use ai_memory::SharedMemory;
        let memory = SharedMemory::new(InMemoryMemory::new(100));
        let provider = MockProvider::new_response("Hello!");

        let tool1 = MockTool {
            name: "tool1",
            response: "Result 1".to_string(),
        };

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .build()
            .unwrap();

        let agent_with_tool = agent.with_tool(tool1);

        assert_eq!(agent_with_tool.tool_descriptors().len(), 1);
        assert_eq!(agent_with_tool.tool_descriptors()[0].name, "tool1");
    }
}

impl<P, M> Clone for Agent<P, M>
where
    P: Provider,
    M: Memory,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
