use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::tool::ToolDescriptor;
use crate::types::{CompletionResponse, ModelConfig, StreamChunk};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<crate::types::Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError>;

    fn stream(
        &self,
        messages: Vec<crate::types::Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>>;

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError>;

    fn name(&self) -> &str;
}
