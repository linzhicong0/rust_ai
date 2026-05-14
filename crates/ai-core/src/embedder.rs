use async_trait::async_trait;

use crate::error::EmbedderError;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError>;
    fn name(&self) -> &str;
}
