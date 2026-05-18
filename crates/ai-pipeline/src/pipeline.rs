//! Pipeline execution and builder.
//!
//! The [`Pipeline`] struct represents a workflow that executes steps sequentially,
//! with support for parallel execution, conditional branching, and loops.

use serde_json::Value;
use std::time::Duration;
use tracing::{debug, error, info, instrument, warn};

use crate::context::PipelineContext;
use crate::step::Step;

/// Re-export PipelineError from ai-core for convenience.
pub use ai_core::error::PipelineError;

/// A pipeline that executes a series of steps.
///
/// Pipelines are the primary workflow orchestration mechanism in the framework.
/// They support sequential execution, parallel steps, conditional branching, and loops.
///
/// # Example
///
/// ```rust,no_run
/// use ai_pipeline::{Pipeline, Step};
/// use serde_json::json;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Build a pipeline with multiple steps
/// let pipeline = Pipeline::builder("my_pipeline")
///     .step(Step::task("task1", "agent1", "input", "output1"))
///     .parallel(vec![
///         Step::task("task2a", "agent2a", "output1", "output2a"),
///         Step::task("task2b", "agent2b", "output1", "output2b"),
///     ])
///     .conditional_value(
///         "check_result",
///         "output2a",
///         json!("success"),
///         Step::task("final", "agent3", "output2a", "final_output"),
///         None
///     )
///     .build()?;
///
/// // Execute the pipeline
/// let result = pipeline.execute(json!("initial input")).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Pipeline {
    /// Unique name for this pipeline.
    pub name: String,

    /// The steps to execute in order.
    pub steps: Vec<Step>,

    /// Optional timeout for the entire pipeline.
    pub timeout: Option<Duration>,

    /// Whether to continue executing steps after an error.
    pub continue_on_error: bool,
}

impl Pipeline {
    /// Create a new pipeline with the given name and steps.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::{Pipeline, Step};
    ///
    /// let steps = vec![
    ///     Step::task("step1", "agent1", "input", "output1"),
    ///     Step::task("step2", "agent2", "output1", "output2"),
    /// ];
    ///
    /// let pipeline = Pipeline::new("my_pipeline", steps);
    /// ```
    pub fn new(name: impl Into<String>, steps: Vec<Step>) -> Self {
        Self {
            name: name.into(),
            steps,
            timeout: None,
            continue_on_error: false,
        }
    }

    /// Create a pipeline builder for fluent construction.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::{Pipeline, Step};
    ///
    /// let builder = Pipeline::builder("my_pipeline");
    /// ```
    pub fn builder(name: impl Into<String>) -> PipelineBuilder {
        PipelineBuilder::new(name)
    }

    /// Execute the pipeline with the given input.
    ///
    /// # Arguments
    ///
    /// * `input` — The initial input value for the pipeline
    ///
    /// # Returns
    ///
    /// The final pipeline context after executing all steps.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ai_pipeline::{Pipeline, Step};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let pipeline = Pipeline::builder("test")
    ///     .step(Step::task("step", "agent", "input", "output"))
    ///     .build()?;
    ///
    /// let result = pipeline.execute(json!("input")).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, input), fields(pipeline = %self.name))]
    pub async fn execute(&self, input: Value) -> Result<PipelineContext, PipelineError> {
        info!("Starting pipeline execution");
        let start = std::time::Instant::now();

        let mut ctx = PipelineContext::new(input);

        // Apply timeout if configured
        let execute_future = async {
            for (idx, step) in self.steps.iter().enumerate() {
                debug!(
                    "Executing step {}/{}: {}",
                    idx + 1,
                    self.steps.len(),
                    step.name
                );

                match self.execute_step(step, &mut ctx).await {
                    Ok(_) => {
                        debug!("Step '{}' completed successfully", step.name);
                    }
                    Err(e) => {
                        error!("Step '{}' failed: {}", step.name, e);
                        if self.continue_on_error {
                            warn!("Continuing pipeline execution despite error");
                            ctx.set(format!("{}_error", step.name), Value::String(e.to_string()));
                        } else {
                            return Err(e);
                        }
                    }
                }
            }

            Ok::<_, PipelineError>(ctx)
        };

        let result = if let Some(timeout) = self.timeout {
            tokio::time::timeout(timeout, execute_future)
                .await
                .map_err(|_| PipelineError::StepFailed {
                    name: self.name.clone(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Pipeline execution timed out",
                    )),
                })?
        } else {
            execute_future.await
        };

        let duration = start.elapsed();
        info!(?duration, "Pipeline execution completed");

        result
    }

    /// Execute a single step.
    async fn execute_step(
        &self,
        step: &Step,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        match &step.kind {
            crate::step::StepKind::Task(task) => {
                self.execute_task_step(&step.name, task, ctx).await
            }

            crate::step::StepKind::Parallel(steps) => self.execute_parallel_step(steps, ctx).await,

            crate::step::StepKind::Conditional {
                condition,
                then_step,
                else_step,
            } => {
                self.execute_conditional_step(condition, then_step, else_step, ctx)
                    .await
            }

            crate::step::StepKind::Loop {
                body,
                condition,
                max_iterations,
            } => {
                self.execute_loop_step(body, condition, *max_iterations, ctx)
                    .await
            }
        }
    }

    /// Execute a task step by running an agent with retry and timeout support.
    ///
    /// NOTE: This is a placeholder implementation. In a full implementation,
    /// this would look up the agent by name and execute it with the input.
    async fn execute_task_step(
        &self,
        step_name: &str,
        task: &crate::step::Task,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        // Validate the task configuration
        task.validate()
            .map_err(|e| PipelineError::Context(e.to_string()))?;

        // Get input from context
        let input = ctx.require(&task.input_key)?.clone();

        // Execute with retry policy if configured
        let output = if let Some(retry_policy) = &task.retry_policy {
            self.execute_with_retry(step_name, &task.agent_name, &input, retry_policy)
                .await?
        } else {
            self.execute_agent(&task.agent_name, &input).await?
        };

        // Validate output against expected output if configured
        task.validate_output(&output)
            .map_err(|e| PipelineError::Context(e.to_string()))?;

        // Store output in context
        ctx.set(&task.output_key, output);

        Ok(())
    }

    /// Execute an agent (placeholder implementation).
    async fn execute_agent(
        &self,
        agent_name: &str,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value, PipelineError> {
        // TODO: Look up agent by name and execute
        // For now, just pass through the input as output
        debug!("Executing agent '{}' (placeholder)", agent_name);

        // Simulate agent execution
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        Ok(input.clone())
    }

    /// Execute an agent with retry policy.
    async fn execute_with_retry(
        &self,
        step_name: &str,
        agent_name: &str,
        input: &serde_json::Value,
        retry_policy: &crate::step::RetryPolicy,
    ) -> Result<serde_json::Value, PipelineError> {
        let mut last_error = None;

        for attempt in 0..=retry_policy.max_retries {
            if attempt > 0 {
                if let Some(delay) = retry_policy.delay_for(attempt - 1) {
                    debug!("Task '{}' retry {} in {:?}", step_name, attempt, delay);
                    tokio::time::sleep(delay).await;
                }
            }

            match self.execute_agent(agent_name, input).await {
                Ok(output) => {
                    if attempt > 0 {
                        info!("Task '{}' succeeded on attempt {}", step_name, attempt + 1);
                    }
                    return Ok(output);
                }
                Err(e) => {
                    warn!("Task '{}' attempt {} failed: {}", step_name, attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            PipelineError::Context(format!(
                "Task '{}' failed after {} retries",
                step_name, retry_policy.max_retries
            ))
        }))
    }

    /// Execute a parallel step by running multiple steps concurrently.
    async fn execute_parallel_step(
        &self,
        steps: &[Step],
        base_ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        use futures::future::try_join_all;

        // Clone the context for each parallel branch before creating futures,
        // so we don't hold multiple borrows of base_ctx simultaneously.
        let cloned_ctxs: Vec<PipelineContext> = steps.iter().map(|_| base_ctx.clone()).collect();

        let futures: Vec<_> = steps
            .iter()
            .zip(cloned_ctxs.into_iter())
            .map(|(step, mut ctx)| async move {
                self.execute_step(step, &mut ctx).await?;
                Ok::<_, PipelineError>(ctx)
            })
            .collect();

        let results = try_join_all(futures)
            .await
            .map_err(|e| PipelineError::StepFailed {
                name: "parallel".to_string(),
                source: Box::new(e),
            })?;

        // Merge all results back into the base context
        for result_ctx in results {
            base_ctx.merge(&result_ctx);
        }

        Ok(())
    }

    /// Execute a conditional step.
    async fn execute_conditional_step(
        &self,
        condition: &crate::step::Condition,
        then_step: &Step,
        else_step: &Option<Box<Step>>,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        let should_run_then = match condition {
            crate::step::Condition::Fn(f) => f(ctx),
            crate::step::Condition::Key { key, value } => {
                ctx.get(key).map_or(false, |v| v == value)
            }
        };

        let step_to_run = if should_run_then {
            then_step
        } else {
            match else_step {
                Some(s) => s.as_ref(),
                None => {
                    debug!("Conditional condition false, no else step - skipping");
                    return Ok(());
                }
            }
        };

        Box::pin(self.execute_step(step_to_run, ctx)).await
    }

    /// Execute a loop step.
    async fn execute_loop_step(
        &self,
        body: &Step,
        condition: &crate::step::LoopCondition,
        max_iterations: u32,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        let mut iterations = 0;

        loop {
            if iterations >= max_iterations {
                return Err(PipelineError::StepFailed {
                    name: "loop".to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Max iterations ({}) exceeded", max_iterations),
                    )),
                });
            }

            let should_continue = match condition {
                crate::step::LoopCondition::Fn(f) => f(ctx),
                crate::step::LoopCondition::Key { key, value } => {
                    ctx.get(key).map_or(false, |v| v == value)
                }
            };

            if !should_continue {
                debug!(
                    "Loop condition false, exiting after {} iterations",
                    iterations
                );
                break;
            }

            iterations += 1;
            debug!("Loop iteration {}", iterations);

            Box::pin(self.execute_step(body, ctx)).await?;
        }

        Ok(())
    }
}

/// Builder for creating pipelines with a fluent API.
///
/// # Example
///
/// ```rust
/// use ai_pipeline::{Pipeline, Step};
/// use serde_json::json;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = Pipeline::builder("my_pipeline")
///     .step(Step::task("step1", "agent1", "input", "out1"))
///     .parallel(vec![
///         Step::task("step2a", "agent2a", "out1", "out2a"),
///         Step::task("step2b", "agent2b", "out1", "out2b"),
///     ])
///     .conditional_value(
///         "check",
///         "out2a",
///         json!("success"),
///         Step::task("final", "agent3", "out2a", "result"),
///         None
///     )
///     .timeout(std::time::Duration::from_secs(60))
///     .continue_on_error(false)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct PipelineBuilder {
    name: String,
    steps: Vec<Step>,
    timeout: Option<Duration>,
    continue_on_error: bool,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            timeout: None,
            continue_on_error: false,
        }
    }

    /// Add a sequential step to the pipeline.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// let builder = Pipeline::builder("test")
    ///     .step(Step::task("step1", "agent", "in", "out"));
    /// ```
    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// Add multiple sequential steps to the pipeline.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// let builder = Pipeline::builder("test")
    ///     .steps(vec![
    ///         Step::task("step1", "agent1", "in", "out1"),
    ///         Step::task("step2", "agent2", "out1", "out2"),
    ///     ]);
    /// ```
    pub fn steps(mut self, steps: Vec<Step>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Add a parallel step to the pipeline.
    ///
    /// All steps in the vector will be executed concurrently.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// let builder = Pipeline::builder("test")
    ///     .parallel(vec![
    ///         Step::task("task1", "agent1", "in", "out1"),
    ///         Step::task("task2", "agent2", "in", "out2"),
    ///     ]);
    /// ```
    pub fn parallel(mut self, steps: Vec<Step>) -> Self {
        self.steps.push(Step::parallel(
            format!("parallel_{}", self.steps.len()),
            steps,
        ));
        self
    }

    /// Add a conditional step with a function-based condition.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// # use ai_pipeline::PipelineContext;
    /// let then_step = Step::task("then", "agent", "in", "out");
    ///
    /// let builder = Pipeline::builder("test")
    ///     .conditional_fn(
    ///         "check",
    ///         |ctx: &PipelineContext| true,
    ///         then_step,
    ///         None
    ///     );
    /// ```
    pub fn conditional_fn(
        mut self,
        name: impl Into<String>,
        condition: impl Fn(&PipelineContext) -> bool + Send + Sync + 'static,
        then_step: Step,
        else_step: Option<Step>,
    ) -> Self {
        self.steps
            .push(Step::conditional_fn(name, condition, then_step, else_step));
        self
    }

    /// Add a conditional step that checks a context key against a value.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// # use serde_json::json;
    /// let then_step = Step::task("then", "agent", "in", "out");
    ///
    /// let builder = Pipeline::builder("test")
    ///     .conditional_value(
    ///         "check",
    ///         "status",
    ///         json!("approved"),
    ///         then_step,
    ///         None
    ///     );
    /// ```
    pub fn conditional_value(
        mut self,
        name: impl Into<String>,
        key: impl Into<String>,
        value: Value,
        then_step: Step,
        else_step: Option<Step>,
    ) -> Self {
        self.steps.push(Step::conditional_value(
            name, key, value, then_step, else_step,
        ));
        self
    }

    /// Add a loop step with a function-based condition.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// # use ai_pipeline::PipelineContext;
    /// let body = Step::task("retry", "agent", "in", "out");
    ///
    /// let builder = Pipeline::builder("test")
    ///     .loop_fn("retry_loop", body, |ctx: &PipelineContext| true, 10);
    /// ```
    pub fn loop_fn(
        mut self,
        name: impl Into<String>,
        body: Step,
        condition: impl Fn(&PipelineContext) -> bool + Send + Sync + 'static,
        max_iterations: u32,
    ) -> Self {
        self.steps
            .push(Step::loop_fn(name, body, condition, max_iterations));
        self
    }

    /// Add a loop step that checks a context key against a value.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// # use serde_json::json;
    /// let body = Step::task("process", "agent", "in", "out");
    ///
    /// let builder = Pipeline::builder("test")
    ///     .loop_value("process_loop", body, "continue", json!(true), 100);
    /// ```
    pub fn loop_value(
        mut self,
        name: impl Into<String>,
        body: Step,
        key: impl Into<String>,
        value: Value,
        max_iterations: u32,
    ) -> Self {
        self.steps
            .push(Step::loop_value(name, body, key, value, max_iterations));
        self
    }

    /// Set the timeout for the entire pipeline.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::Pipeline;
    /// use std::time::Duration;
    ///
    /// let builder = Pipeline::builder("test")
    ///     .timeout(Duration::from_secs(60));
    /// ```
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set whether to continue execution after an error.
    ///
    /// When `true`, errors are stored in the context and execution continues.
    /// When `false` (default), the pipeline stops on the first error.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::Pipeline;
    ///
    /// let builder = Pipeline::builder("test")
    ///     .continue_on_error(true);
    /// ```
    pub fn continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Build the pipeline.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_pipeline::{Pipeline, Step};
    /// let pipeline = Pipeline::builder("my_pipeline")
    ///     .step(Step::task("step", "agent", "in", "out"))
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(self) -> Result<Pipeline, PipelineError> {
        if self.steps.is_empty() {
            return Err(PipelineError::Context(
                "Pipeline must have at least one step".to_string(),
            ));
        }

        Ok(Pipeline {
            name: self.name,
            steps: self.steps,
            timeout: self.timeout,
            continue_on_error: self.continue_on_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_pipeline_new() {
        let steps = vec![Step::task("s1", "a", "i", "o")];
        let pipeline = Pipeline::new("test", steps);
        assert_eq!(pipeline.name, "test");
        assert_eq!(pipeline.steps.len(), 1);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "a", "i", "o"))
            .step(Step::task("s2", "b", "i", "o"))
            .build()
            .unwrap();

        assert_eq!(pipeline.name, "test");
        assert_eq!(pipeline.steps.len(), 2);
    }

    #[test]
    fn test_pipeline_builder_empty() {
        let result = Pipeline::builder("test").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_builder_with_timeout() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "a", "i", "o"))
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        assert_eq!(pipeline.timeout, Some(Duration::from_secs(10)));
    }

    #[test]
    fn test_pipeline_builder_continue_on_error() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "a", "i", "o"))
            .continue_on_error(true)
            .build()
            .unwrap();

        assert!(pipeline.continue_on_error);
    }

    #[tokio::test]
    async fn test_pipeline_execute_simple() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "agent", "input", "output"))
            .build()
            .unwrap();

        let result = pipeline.execute(json!("test")).await.unwrap();
        assert_eq!(result.get("input"), Some(&json!("test")));
        // The placeholder implementation passes input through to output
        assert_eq!(result.get("output"), Some(&json!("test")));
    }

    #[tokio::test]
    async fn test_pipeline_execute_parallel() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "agent1", "input", "out1"))
            .parallel(vec![
                Step::task("s2a", "agent2a", "out1", "out2a"),
                Step::task("s2b", "agent2b", "out1", "out2b"),
            ])
            .build()
            .unwrap();

        let result = pipeline.execute(json!("test")).await.unwrap();
        // All outputs should exist from parallel execution
        assert!(result.has("out1"));
        assert!(result.has("out2a"));
        assert!(result.has("out2b"));
    }

    #[tokio::test]
    async fn test_pipeline_execute_conditional() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "agent", "input", "decision"))
            .conditional_value(
                "check",
                "decision",
                json!("input"), // step s1 sets decision to "input"
                Step::task("then_branch", "agent", "decision", "result"),
                Some(Step::task("else_branch", "agent", "decision", "result")),
            )
            .build()
            .unwrap();

        let result = pipeline.execute(json!("input")).await.unwrap();
        // s1 sets decision to "input", so then_branch should run
        assert_eq!(result.get("decision"), Some(&json!("input")));
        // then_branch sets result to the value of decision
        assert_eq!(result.get("result"), Some(&json!("input")));
    }
}
