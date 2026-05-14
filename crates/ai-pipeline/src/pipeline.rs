use ai_core::error::PipelineError;
use serde_json::Value;

use super::context::PipelineContext;
use super::step::Step;

pub struct Pipeline {
    pub steps: Vec<Step>,
}

impl Pipeline {
    pub fn new(steps: Vec<Step>) -> Self {
        Self { steps }
    }

    pub async fn execute(&self, input: Value) -> Result<PipelineContext, PipelineError> {
        let mut ctx = PipelineContext::new(input);
        for step in &self.steps {
            self.execute_step(step, &mut ctx).await?;
        }
        Ok(ctx)
    }

    async fn execute_step(
        &self,
        step: &Step,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        // TODO: Implement step execution
        Ok(())
    }
}
