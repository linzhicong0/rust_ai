use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::Value;

use crate::error::ToolError;

pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn descriptor(&self) -> ToolDescriptor;
    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError>;
}
