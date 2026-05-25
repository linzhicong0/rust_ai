// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # AI Framework — Pipeline Library
//!
//! The pipeline library provides workflow orchestration for multi-step AI workflows.
//!
//! ## Features
//!
//! - **Sequential execution**: Run steps one after another
//! - **Parallel execution**: Run multiple steps concurrently
//! - **Conditional branching**: Execute different steps based on conditions
//! - **Loop execution**: Repeat steps until a condition is met
//! - **Pipeline context**: Flow data between steps
//! - **Error handling**: Configure error handling at pipeline level
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ai_pipeline::{Pipeline, Step};
//! use serde_json::json;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build a pipeline with multiple steps
//! let pipeline = Pipeline::builder("my_workflow")
//!     .step(Step::task("step1", "agent1", "input", "output1"))
//!     .parallel(vec![
//!         Step::task("step2a", "agent2a", "output1", "output2a"),
//!         Step::task("step2b", "agent2b", "output1", "output2b"),
//!     ])
//!     .conditional_value(
//!         "check_result",
//!         "output2a",
//!         json!("success"),
//!         Step::task("final", "agent3", "output2a", "final_output"),
//!         None
//!     )
//!     .build()?;
//!
//! // Execute the pipeline
//! let result = pipeline.execute(json!("initial input")).await?;
//!
//! // Get the final output
//! if let Some(final_output) = result.get("final_output") {
//!     println!("Result: {}", final_output);
//! }
//! # Ok(())
//! # }
//! ```

pub mod async_task;
pub mod context;
pub mod dag;
pub mod human_approval;
pub mod pipeline;
pub mod step;

// Re-export main types for convenience
pub use async_task::{spawn_task, AsyncTaskBuilder, TaskEvent, TaskHandle, TaskStatus};
pub use context::PipelineContext;
pub use human_approval::{
    ApprovalCallback, ApprovalDecision, ApprovalRequest, ApprovalResult, HumanApproval,
    TimeoutPolicy,
};
pub use pipeline::{Pipeline, PipelineBuilder, TrackedTaskOutput};
pub use step::{
    BackoffStrategy, Condition, LoopCondition, RetryPolicy, Step, StepKind, Task, TaskBuilder,
    TaskErrorPolicy, TaskValidationError,
};

// Re-export error from ai-core
pub use ai_core::error::PipelineError;

/// Result type alias for pipeline operations.
pub type Result<T> = std::result::Result<T, PipelineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_exports() {
        // Verify that re-exports compile
        let _ = PipelineContext::empty();
        let _step = Step::task("test", "agent", "in", "out");
    }
}
