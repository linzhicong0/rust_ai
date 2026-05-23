// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Builder for constructing [`Agent`] instances.
//!
//! The [`AgentBuilder`] uses a type-state pattern to ensure required fields
//! are provided before an agent can be built.

use std::sync::Arc;

use ai_core::error::AgentError;
use ai_core::memory::Memory;
use ai_core::provider::Provider;
use ai_core::tool::Tool;
use ai_core::types::ModelConfig;
use ai_core::CostTracker;

use super::agent::{Agent, AgentInner};

// Type-state markers
pub struct NoProvider;
pub struct HasProvider<P>(std::marker::PhantomData<P>);

/// Builder for constructing an [`Agent`].
///
/// The builder ensures that a provider is set before `build()` can be called.
///
/// ## Example
///
/// ```rust,no_run
/// use ai_agent::AgentBuilder;
/// use ai_memory::InMemoryMemory;
/// use ai_provider_openai::OpenAiProvider;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = InMemoryMemory::new(100);
/// let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
///
/// let agent = AgentBuilder::new(memory)
///     .provider(provider)
///     .role("You are a helpful research assistant.")
///     .goal("Help users find accurate information.")
///     .backstory("You have access to web search and academic databases.")
///     .max_iterations(15)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder<P, M, State = NoProvider>
where
    M: Memory + 'static,
{
    pub(crate) provider: Option<P>,
    pub(crate) memory: M,
    pub(crate) name: Option<String>,
    pub(crate) role: Option<String>,
    pub(crate) goal: Option<String>,
    pub(crate) backstory: Option<String>,
    pub(crate) tools: Vec<Box<dyn Tool>>,
    pub(crate) model_config: ModelConfig,
    pub(crate) max_iterations: u32,
    pub(crate) cost_tracker: CostTracker,
    pub(crate) _state: std::marker::PhantomData<(P, State)>,
}

impl<M> AgentBuilder<(), M, NoProvider>
where
    M: Memory + WithProvider,
{
    /// Create a new agent builder with the given memory backend.
    ///
    /// # Arguments
    ///
    /// * `memory` — The memory implementation to use
    pub fn new(memory: M) -> Self {
        Self {
            provider: None,
            memory,
            name: None,
            role: None,
            goal: None,
            backstory: None,
            tools: Vec::new(),
            model_config: ModelConfig::default(),
            max_iterations: 10,
            cost_tracker: CostTracker::new(),
            _state: std::marker::PhantomData,
        }
    }
}

impl<P, M, State> AgentBuilder<P, M, State>
where
    M: Memory + 'static,
{
    /// Set the agent name used for tracking and identification.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the agent's role description.
    ///
    /// This is included in the system prompt and tells the agent what
    /// role it should play.
    ///
    /// # Arguments
    ///
    /// * `role` — Role description (e.g., "You are a helpful coding assistant.")
    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Set the agent's goal.
    ///
    /// This describes what the agent should accomplish.
    ///
    /// # Arguments
    ///
    /// * `goal` — Goal description
    pub fn goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    /// Set the agent's backstory.
    ///
    /// This provides context about the agent's capabilities and history.
    ///
    /// # Arguments
    ///
    /// * `backstory` — Backstory description
    pub fn backstory(mut self, backstory: impl Into<String>) -> Self {
        self.backstory = Some(backstory.into());
        self
    }

    /// Add a tool to this agent.
    ///
    /// # Arguments
    ///
    /// * `tool` — The tool to add
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Set the model configuration for this agent.
    ///
    /// # Arguments
    ///
    /// * `config` — The model configuration
    pub fn model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = config;
        self
    }

    /// Set the maximum number of ReAct loop iterations.
    ///
    /// # Arguments
    ///
    /// * `max` — Maximum iterations before giving up
    pub fn max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the temperature for this agent.
    ///
    /// # Arguments
    ///
    /// * `temp` — Temperature (0.0 to 2.0)
    pub fn temperature(mut self, temp: f64) -> Self {
        self.model_config = self.model_config.with_temperature(temp);
        self
    }

    /// Override the cost tracker used by this agent.
    pub fn cost_tracker(mut self, cost_tracker: CostTracker) -> Self {
        self.cost_tracker = cost_tracker;
        self
    }
}

impl<P, M> AgentBuilder<P, M, NoProvider>
where
    M: Memory + 'static,
{
    /// Set the LLM provider for this agent.
    ///
    /// # Arguments
    ///
    /// * `provider` — The provider implementation
    pub fn provider<Pr: Provider + 'static>(
        self,
        provider: Pr,
    ) -> AgentBuilder<Pr, M, HasProvider<Pr>> {
        AgentBuilder {
            provider: Some(provider),
            memory: self.memory,
            name: self.name,
            role: self.role,
            goal: self.goal,
            backstory: self.backstory,
            tools: self.tools,
            model_config: self.model_config,
            max_iterations: self.max_iterations,
            cost_tracker: self.cost_tracker,
            _state: std::marker::PhantomData,
        }
    }
}

impl<P, M> AgentBuilder<P, M, HasProvider<P>>
where
    P: Provider + 'static,
    M: Memory + 'static,
{
    /// Build the agent.
    ///
    /// # Panics
    ///
    /// Panics if no provider was set (should be prevented by type system).
    pub fn build(self) -> Result<Agent<P, M>, AgentError> {
        let provider = self
            .provider
            .expect("Provider should be set in HasProvider state");

        let inner = AgentInner {
            provider,
            memory: self.memory,
            name: self.name,
            role: self.role,
            goal: self.goal,
            backstory: self.backstory,
            tools: self.tools,
            model_config: self.model_config,
            max_iterations: self.max_iterations,
            cost_tracker: self.cost_tracker,
        };

        Ok(Agent {
            inner: Arc::new(inner),
        })
    }
}

// Helper trait for type-state pattern - allows any memory with any provider
pub trait WithProvider: Memory + 'static {}

// Implement for all Memory types
impl<T> WithProvider for T where T: Memory + 'static {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use ai_core::types::Role;

    // Mock memory and provider for testing
    struct MockMemory;

    #[async_trait::async_trait]
    impl Memory for MockMemory {
        async fn add(
            &self,
            _entry: ai_core::memory::MemoryEntry,
        ) -> Result<(), ai_core::error::MemoryError> {
            Ok(())
        }
        async fn get(
            &self,
            _limit: Option<usize>,
        ) -> Result<Vec<ai_core::memory::MemoryEntry>, ai_core::error::MemoryError> {
            Ok(vec![])
        }
        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<ai_core::memory::MemoryEntry>, ai_core::error::MemoryError> {
            Ok(vec![])
        }
        async fn clear(&self) -> Result<(), ai_core::error::MemoryError> {
            Ok(())
        }
    }

    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<ai_core::types::Message>,
            _config: &ModelConfig,
            _tools: &[ai_core::tool::ToolDescriptor],
        ) -> Result<ai_core::types::CompletionResponse, ai_core::error::ProviderError> {
            Ok(ai_core::types::CompletionResponse {
                content: "Mock response".to_string(),
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
            _config: &ModelConfig,
            _tools: &[ai_core::tool::ToolDescriptor],
        ) -> futures::stream::BoxStream<
            'static,
            Result<ai_core::types::StreamChunk, ai_core::error::ProviderError>,
        > {
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

    #[test]
    fn test_builder_pattern() {
        let memory = MockMemory;
        let provider = MockProvider;

        let builder = AgentBuilder::new(memory)
            .role("Test role")
            .goal("Test goal")
            .max_iterations(5);

        assert_eq!(builder.role, Some("Test role".to_string()));
        assert_eq!(builder.goal, Some("Test goal".to_string()));
        assert_eq!(builder.max_iterations, 5);
    }

    #[test]
    fn test_build_with_provider() {
        let memory = MockMemory;
        let provider = MockProvider;

        let agent = AgentBuilder::new(memory)
            .provider(provider)
            .role("Test")
            .build()
            .unwrap();

        assert_eq!(agent.role(), Some("Test"));
        assert_eq!(agent.max_iterations(), 10);
    }
}
