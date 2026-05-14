use async_trait::async_trait;

use crate::error::GuardrailError;

#[derive(Debug)]
pub enum GuardrailAction {
    Allow,
    Block(String),
    Modify(String),
}

#[async_trait]
pub trait Guardrail: Send + Sync {
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError>;
    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError>;
    fn name(&self) -> &str;
}
