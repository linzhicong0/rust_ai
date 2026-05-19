// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent system with ReAct (Reasoning + Acting) loop.
//!
//! This module provides the [`Agent`] type which orchestrates LLM interactions
//! with tools and memory to accomplish tasks through iterative reasoning.

use std::sync::Arc;

use futures::stream::{BoxStream, StreamExt};
use futures::TryStreamExt;

use ai_core::error::AgentError;
use ai_core::memory::{Memory, MemoryEntry};
use ai_core::provider::Provider;
use ai_core::tool::{Tool, ToolDescriptor, ToolOutput};
use ai_core::types::{
    AgentEvent, AgentOutput, CompletionResponse, Content, FinishReason, Message,
    ModelConfig, Role, StreamChunk, ToolCall, Usage,
};

// AgentInner is defined below in this module

/// Internal state of an Agent.
pub struct AgentInner<P, M>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    pub(crate) provider: P,
    pub(crate) memory: M,
    pub(crate) role: Option<String>,
    pub(crate) goal: Option<String>,
    pub(crate) backstory: Option<String>,
    pub(crate) tools: Vec<Box<dyn Tool>>,
    pub(crate) model_config: ModelConfig,
    pub(crate) max_iterations: u32,
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
            role: self.role.clone(),
            goal: self.goal.clone(),
            backstory: self.backstory.clone(),
            tools: Vec::new(), // Tools need to be re-added after clone
            model_config: self.model_config.clone(),
            max_iterations: self.max_iterations,
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
        self.inner
            .tools
            .iter()
            .map(|t| t.descriptor())
            .collect()
    }

    /// Get the model config for this agent.
    pub fn model_config(&self) -> &ModelConfig {
        &self.inner.model_config
    }

    /// Get the maximum iterations before giving up.
    pub fn max_iterations(&self) -> u32 {
        self.inner.max_iterations
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

        for iteration in 0..self.inner.max_iterations {
            tracing::debug!(
                agent = self.inner.role.as_deref().unwrap_or("unnamed"),
                iteration,
                "Agent ReAct loop"
            );

            // Get response from provider
            let response = self
                .inner
                .provider
                .complete(
                    messages.clone(),
                    &self.inner.model_config,
                    &self.tool_descriptors(),
                )
                .await?;

            // Store assistant response in memory
            self.store_assistant_response(&response, iteration).await;

            if response.tool_calls.is_empty() {
                // No tool calls — agent is done
                return Ok(AgentOutput {
                    content: response.content,
                });
            }

            // Execute tool calls
            for tool_call in &response.tool_calls {
                let tool_result = self.execute_tool_call(tool_call).await?;

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
    pub fn stream(&self, input: impl Into<String>) -> BoxStream<'static, Result<AgentEvent, AgentError>> {
        let input = input.into();
        let agent = self.clone();

        Box::pin(async_stream::try_stream! {
            let mut messages = agent.build_initial_messages(&input).await?;

            for _iteration in 0..agent.inner.max_iterations {
                let mut stream = agent.inner.provider.stream(
                    messages.clone(),
                    &agent.inner.model_config,
                    &agent.tool_descriptors(),
                );

                let mut content_buffer = String::new();
                let tool_calls_buffer: Vec<ToolCall> = Vec::new();

                while let Some(chunk_result) = stream.next().await {
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

        // Get conversation history from memory
        if let Ok(history) = self.inner.memory.get(None).await {
            for entry in history {
                messages.push(Message {
                    role: entry.role,
                    content: Content::Text(entry.content),
                });
            }
        }

        // Add current user input
        messages.push(Message::user(input));

        Ok(messages)
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

    /// Store the assistant's response in memory.
    async fn store_assistant_response(&self, response: &CompletionResponse, _iteration: u32) {
        let entry = MemoryEntry::new(Role::Assistant, &response.content);
        let _ = self.inner.memory.add(entry).await;
    }

    /// Execute a single tool call.
    async fn execute_tool_call(&self, tool_call: &ToolCall) -> Result<String, AgentError> {
        let tool = self
            .inner
            .tools
            .iter()
            .find(|t| t.descriptor().name == tool_call.name)
            .ok_or_else(|| AgentError::ToolNotFound(tool_call.name.clone()))?;

        let output = tool.execute(tool_call.arguments.clone()).await?;

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
        Agent { inner: Arc::new(inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::builder::AgentBuilder;
    use ai_core::error::{ProviderError, ToolError};
    use ai_core::tool::{ToolDescriptor, ToolOutput};
    use ai_core::types::{FinishReason, Message, Role, ToolCall, Usage};
    use ai_memory::InMemoryMemory;
    use serde_json::json;

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

        async fn execute(&self, _input: serde_json::Value) -> Result<ToolOutput, ai_core::error::ToolError> {
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
        assert_eq!(agent.backstory(), Some("You are an AI with general knowledge."));
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
        assert_eq!(*provider_calls.lock().unwrap(), 2);
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
