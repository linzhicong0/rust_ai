use std::sync::Arc;

use futures::stream::BoxStream;

use ai_core::error::AgentError;
use ai_core::memory::Memory;
use ai_core::provider::Provider;
use ai_core::tool::Tool;
use ai_core::types::{AgentEvent, AgentOutput, ModelConfig};

pub(crate) struct AgentInner<P, M>
where
    P: Provider,
    M: Memory,
{
    pub provider: P,
    pub memory: M,
    pub role: Option<String>,
    pub goal: Option<String>,
    pub backstory: Option<String>,
    pub tools: Vec<Box<dyn Tool>>,
    pub model_config: ModelConfig,
    pub max_iterations: u32,
}

pub struct Agent<P, M>
where
    P: Provider,
    M: Memory,
{
    pub(crate) inner: Arc<AgentInner<P, M>>,
}

impl<P, M> Agent<P, M>
where
    P: Provider,
    M: Memory,
{
    pub async fn run(&self, input: impl Into<String>) -> Result<AgentOutput, AgentError> {
        let _input = input.into();
        // ReAct loop: Reason -> Act (tool call) -> Observe -> repeat
        todo!("Implement ReAct loop")
    }

    pub fn stream(&self, input: impl Into<String>) -> BoxStream<'static, Result<AgentEvent, AgentError>> {
        let _input = input.into();
        // Streaming version of the ReAct loop
        todo!("Implement streaming ReAct loop")
    }
}
