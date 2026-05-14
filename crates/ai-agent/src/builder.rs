use ai_core::memory::Memory;
use ai_core::provider::Provider;
use ai_core::tool::Tool;
use ai_core::types::ModelConfig;

use super::agent::{Agent, AgentInner};
use std::sync::Arc;

pub struct NoProvider;
pub struct HasProvider<P>(std::marker::PhantomData<P>);

pub struct AgentBuilder<P, M, State = NoProvider>
where
    P: Provider,
    M: Memory,
{
    pub(crate) provider: Option<P>,
    pub(crate) memory: M,
    pub(crate) role: Option<String>,
    pub(crate) goal: Option<String>,
    pub(crate) backstory: Option<String>,
    pub(crate) tools: Vec<Box<dyn Tool>>,
    pub(crate) model_config: ModelConfig,
    pub(crate) max_iterations: u32,
    _state: std::marker::PhantomData<State>,
}

impl<M> AgentBuilder<ai_core::error::ProviderError, M, NoProvider>
where
    M: Memory,
{
    pub fn new(memory: M) -> Self {
        Self {
            provider: None,
            memory,
            role: None,
            goal: None,
            backstory: None,
            tools: Vec::new(),
            model_config: ModelConfig {
                model: "gpt-4".to_string(),
                temperature: Some(0.7),
                max_tokens: None,
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop_sequences: None,
            },
            max_iterations: 10,
            _state: std::marker::PhantomData,
        }
    }
}

// This is a workaround — a real type-state builder would use distinct type parameters.
// For now the builder is simplified to compile without the full generic dance.
