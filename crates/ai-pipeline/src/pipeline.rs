//! Pipeline execution and builder.
//!
//! The [`Pipeline`] struct represents a workflow that executes steps sequentially,
//! with support for parallel execution, conditional branching, and loops.

use ai_core::{
    agent_scope, new_request_id, request_scope, workflow_scope, CostTracker, Usage, GLOBAL_SCOPE,
};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{debug, error, info, instrument, warn, Instrument};

use std::sync::Arc;

use crate::context::PipelineContext;
use crate::dag::Dag;
use crate::step::{Step, StepKind, TaskErrorPolicy};

/// Re-export PipelineError from ai-core for convenience.
pub use ai_core::error::PipelineError;

/// Type alias for an async agent executor callback.
///
/// The callback receives the agent name and input value, and returns
/// the agent's output or a [`PipelineError`].
pub type AgentExecutorFn = Arc<
    dyn Fn(
            String,
            serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value, PipelineError>> + Send>,
        > + Send
        + Sync,
>;

/// Result returned by a cost-aware task executor.
#[derive(Debug, Clone)]
pub struct TrackedTaskOutput {
    /// Task output to write into the pipeline context.
    pub output: serde_json::Value,

    /// Model name used for the task execution.
    pub model: String,

    /// Token usage reported by the task execution.
    pub usage: Usage,
}

/// Type alias for an agent executor that also returns usage metadata.
pub type TrackedAgentExecutorFn = Arc<
    dyn Fn(
            String,
            serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<TrackedTaskOutput, PipelineError>> + Send>,
        > + Send
        + Sync,
>;

enum AgentExecutionResult {
    Untracked(Value),
    Tracked(TrackedTaskOutput),
}

impl AgentExecutionResult {
    fn output(&self) -> &Value {
        match self {
            Self::Untracked(value) => value,
            Self::Tracked(tracked) => &tracked.output,
        }
    }

    fn tracked(&self) -> Option<&TrackedTaskOutput> {
        match self {
            Self::Tracked(tracked) => Some(tracked),
            Self::Untracked(_) => None,
        }
    }

    fn into_output(self) -> Value {
        match self {
            Self::Untracked(value) => value,
            Self::Tracked(tracked) => tracked.output,
        }
    }
}

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

    /// Default error policy to apply to task steps when they do not define one.
    pub default_task_error_policy: Option<TaskErrorPolicy>,

    /// Maximum number of parallel branches that may run at once.
    pub parallel_step_concurrency_limit: Option<usize>,

    /// Maximum number of concurrent workflow executions for this pipeline instance.
    pub workflow_concurrency_limit: Option<usize>,

    /// Optional agent executor to run `Task` steps.
    ///
    /// Set via [`Pipeline::with_agent_executor`] or
    /// [`PipelineBuilder::with_agent_executor`]. If `None`, task steps that
    /// invoke agents will return a [`PipelineError::Context`] error.
    pub agent_executor: Option<AgentExecutorFn>,

    /// Optional agent executor that returns cost metadata.
    pub tracked_agent_executor: Option<TrackedAgentExecutorFn>,

    /// Cost tracker for workflow-level accounting.
    pub cost_tracker: CostTracker,

    workflow_semaphore: Option<Arc<Semaphore>>,
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
            default_task_error_policy: None,
            parallel_step_concurrency_limit: None,
            workflow_concurrency_limit: None,
            agent_executor: None,
            tracked_agent_executor: None,
            cost_tracker: CostTracker::new(),
            workflow_semaphore: None,
        }
    }

    /// Attach an agent executor to this pipeline.
    ///
    /// The executor is called for every [`Step::task`] step. It receives the
    /// agent name and the task input, and must return the agent output.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_pipeline::{Pipeline, Step};
    /// use serde_json::json;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pipeline = Pipeline::builder("demo")
    ///     .step(Step::task("greet", "greeter", "input", "output"))
    ///     .build()?
    ///     .with_agent_executor(|agent, input| Box::pin(async move {
    ///         Ok(json!(format!("[{}] processed: {}", agent, input)))
    ///     }));
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_agent_executor<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut:
            std::future::Future<Output = Result<serde_json::Value, PipelineError>> + Send + 'static,
    {
        self.agent_executor = Some(Arc::new(move |name, input| Box::pin(f(name, input))));
        self
    }

    /// Attach an agent executor that also returns usage metadata for cost tracking.
    pub fn with_tracked_agent_executor<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut:
            std::future::Future<Output = Result<TrackedTaskOutput, PipelineError>> + Send + 'static,
    {
        self.tracked_agent_executor = Some(Arc::new(move |name, input| Box::pin(f(name, input))));
        self
    }

    /// Override the pipeline cost tracker.
    pub fn with_cost_tracker(mut self, cost_tracker: CostTracker) -> Self {
        self.cost_tracker = cost_tracker;
        self
    }

    /// Set the maximum number of parallel branches that may run at once.
    pub fn with_parallel_step_concurrency_limit(mut self, limit: usize) -> Self {
        assert!(
            limit > 0,
            "parallel step concurrency limit must be greater than zero"
        );
        self.parallel_step_concurrency_limit = Some(limit);
        self
    }

    /// Set the maximum number of concurrent workflow executions for this pipeline instance.
    pub fn with_workflow_concurrency_limit(mut self, limit: usize) -> Self {
        assert!(
            limit > 0,
            "workflow concurrency limit must be greater than zero"
        );
        self.workflow_concurrency_limit = Some(limit);
        self.workflow_semaphore = Some(Arc::new(Semaphore::new(limit)));
        self
    }

    /// Return the pipeline cost tracker.
    pub fn cost_tracker(&self) -> &CostTracker {
        &self.cost_tracker
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
        let _workflow_permit = self.acquire_workflow_permit().await?;

        let mut ctx = PipelineContext::new(input);

        // Apply timeout if configured
        let execute_future = async {
            let total_steps = self.steps.len();
            let mut pending_steps: Vec<&Step> = self.steps.iter().collect();
            let mut completed_tasks = HashSet::new();
            let mut executed_steps = 0usize;

            while !pending_steps.is_empty() {
                let mut remaining_steps = Vec::new();
                let mut progressed = false;

                for step in pending_steps {
                    if !self.step_dependencies_satisfied(step, &completed_tasks) {
                        remaining_steps.push(step);
                        continue;
                    }

                    executed_steps += 1;
                    progressed = true;

                    debug!(
                        "Executing step {}/{}: {}",
                        executed_steps, total_steps, step.name
                    );

                    match self.execute_step(step, &mut ctx).await {
                        Ok(_) => {
                            debug!("Step '{}' completed successfully", step.name);
                            if matches!(&step.kind, StepKind::Task(_)) {
                                completed_tasks.insert(step.name.clone());
                            }
                        }
                        Err(e) => {
                            error!("Step '{}' failed: {}", step.name, e);
                            if self.continue_on_error && !matches!(&step.kind, StepKind::Task(_)) {
                                warn!("Continuing pipeline execution despite error");
                                ctx.set(
                                    format!("{}_error", step.name),
                                    Value::String(e.to_string()),
                                );
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }

                if !progressed {
                    let blocked_steps: Vec<String> = remaining_steps
                        .iter()
                        .map(|step| step.name.clone())
                        .collect();
                    return Err(PipelineError::Context(format!(
                        "No executable steps remain in pipeline '{}' because task dependencies are unresolved: {}",
                        self.name,
                        blocked_steps.join(", ")
                    )));
                }

                pending_steps = remaining_steps;
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

    async fn acquire_workflow_permit(&self) -> Result<Option<OwnedSemaphorePermit>, PipelineError> {
        match &self.workflow_semaphore {
            Some(semaphore) => semaphore
                .clone()
                .acquire_owned()
                .await
                .map(Some)
                .map_err(|_| {
                    PipelineError::Context(format!(
                        "Workflow concurrency limiter for pipeline '{}' has been closed",
                        self.name
                    ))
                }),
            None => Ok(None),
        }
    }

    fn step_dependencies_satisfied(&self, step: &Step, completed_tasks: &HashSet<String>) -> bool {
        match &step.kind {
            StepKind::Task(task) => task
                .dependencies
                .iter()
                .all(|dependency| completed_tasks.contains(dependency)),
            _ => true,
        }
    }

    /// Execute a single step.
    async fn execute_step(
        &self,
        step: &Step,
        ctx: &mut PipelineContext,
    ) -> Result<(), PipelineError> {
        let step_span = tracing::info_span!(
            "pipeline_step",
            pipeline = %self.name,
            step = %step.name,
            kind = Self::step_kind_name(step),
        );

        async move {
            match &step.kind {
                crate::step::StepKind::Task(task) => {
                    self.execute_task_step(&step.name, task, ctx).await
                }

                crate::step::StepKind::Parallel(steps) => {
                    self.execute_parallel_step(steps, ctx).await
                }

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
        .instrument(step_span)
        .await
    }

    fn step_kind_name(step: &Step) -> &'static str {
        match &step.kind {
            StepKind::Task(_) => "task",
            StepKind::Parallel(_) => "parallel",
            StepKind::Conditional { .. } => "conditional",
            StepKind::Loop { .. } => "loop",
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
        let (output, executed_agent_name) = match self
            .execute_primary_task(step_name, task, &input)
            .await
        {
            Ok(output) => (output, task.agent_name.clone()),
            Err(primary_error) => {
                let policy = self.resolve_task_error_policy(task);

                match policy {
                    TaskErrorPolicy::Halt => {
                        self.store_task_error_context(
                            ctx,
                            step_name,
                            &task.agent_name,
                            "halt",
                            &primary_error.to_string(),
                            None,
                            false,
                        );
                        return Err(primary_error);
                    }
                    TaskErrorPolicy::Skip => {
                        self.store_task_error_context(
                            ctx,
                            step_name,
                            &task.agent_name,
                            "skip",
                            &primary_error.to_string(),
                            None,
                            false,
                        );
                        return Ok(());
                    }
                    TaskErrorPolicy::FallbackAgent(fallback_agent) => {
                        let fallback_result = self
                            .execute_task_for_agent(step_name, &fallback_agent, task, &input)
                            .await;

                        match fallback_result {
                            Ok(output) => {
                                self.store_task_error_context(
                                    ctx,
                                    step_name,
                                    &task.agent_name,
                                    "fallback",
                                    &primary_error.to_string(),
                                    Some(&fallback_agent),
                                    true,
                                );
                                (output, fallback_agent)
                            }
                            Err(fallback_error) => {
                                self.store_task_error_context(
                                    ctx,
                                    step_name,
                                    &task.agent_name,
                                    "fallback",
                                    &primary_error.to_string(),
                                    Some(&fallback_agent),
                                    false,
                                );
                                return Err(PipelineError::Context(format!(
                                    "Task '{}' failed with primary agent '{}' and fallback agent '{}': {}; fallback: {}",
                                    step_name,
                                    task.agent_name,
                                    fallback_agent,
                                    primary_error,
                                    fallback_error,
                                )));
                            }
                        }
                    }
                }
            }
        };

        // Validate output against expected output if configured
        task.validate_output(output.output())
            .map_err(|e| PipelineError::Context(e.to_string()))?;

        if let Some(tracked) = output.tracked() {
            self.cost_tracker
                .record_many(
                    [
                        request_scope(&new_request_id()),
                        agent_scope(&executed_agent_name),
                        workflow_scope(&self.name),
                        GLOBAL_SCOPE.to_string(),
                    ],
                    &tracked.model,
                    &tracked.usage,
                )
                .await;
        }

        // Store output in context
        ctx.set(&task.output_key, output.into_output());

        Ok(())
    }

    async fn execute_primary_task(
        &self,
        step_name: &str,
        task: &crate::step::Task,
        input: &serde_json::Value,
    ) -> Result<AgentExecutionResult, PipelineError> {
        self.execute_task_for_agent(step_name, &task.agent_name, task, input)
            .await
    }

    async fn execute_task_for_agent(
        &self,
        step_name: &str,
        agent_name: &str,
        task: &crate::step::Task,
        input: &serde_json::Value,
    ) -> Result<AgentExecutionResult, PipelineError> {
        if let Some(retry_policy) = &task.retry_policy {
            self.execute_with_retry(step_name, agent_name, input, retry_policy, task.timeout)
                .await
        } else {
            self.execute_agent_with_timeout(step_name, agent_name, input, task.timeout)
                .await
        }
    }

    fn resolve_task_error_policy(&self, task: &crate::step::Task) -> TaskErrorPolicy {
        task.error_policy
            .clone()
            .or_else(|| self.default_task_error_policy.clone())
            .unwrap_or_else(|| {
                if self.continue_on_error {
                    TaskErrorPolicy::Skip
                } else {
                    TaskErrorPolicy::Halt
                }
            })
    }

    fn store_task_error_context(
        &self,
        ctx: &mut PipelineContext,
        step_name: &str,
        agent_name: &str,
        action: &str,
        error_message: &str,
        fallback_agent: Option<&str>,
        recovered: bool,
    ) {
        ctx.set(
            format!("{}_error", step_name),
            json!({
                "step": step_name,
                "agent_name": agent_name,
                "action": action,
                "error": error_message,
                "fallback_agent": fallback_agent,
                "recovered": recovered,
            }),
        );
    }

    /// Execute an agent via the configured [`AgentExecutorFn`].
    ///
    /// Returns `PipelineError::Context` if no executor has been configured.
    async fn execute_agent(
        &self,
        agent_name: &str,
        input: &serde_json::Value,
    ) -> Result<AgentExecutionResult, PipelineError> {
        let agent_span = tracing::debug_span!(
            "pipeline_agent_call",
            pipeline = %self.name,
            agent = agent_name,
        );

        if let Some(executor) = &self.tracked_agent_executor {
            debug!("Executing agent '{}' via tracked executor", agent_name);
            let output = executor(agent_name.to_string(), input.clone())
                .instrument(agent_span.clone())
                .await?;
            return Ok(AgentExecutionResult::Tracked(output));
        }

        match &self.agent_executor {
            Some(executor) => {
                debug!("Executing agent '{}' via configured executor", agent_name);
                executor(agent_name.to_string(), input.clone())
                    .instrument(agent_span)
                    .await
                    .map(AgentExecutionResult::Untracked)
            }
            None => Err(PipelineError::Context(format!(
                "No agent executor configured for pipeline '{}'. \
                 Call Pipeline::with_agent_executor() or Pipeline::with_tracked_agent_executor() before executing agent step '{}'",
                self.name, agent_name
            ))),
        }
    }

    async fn execute_agent_with_timeout(
        &self,
        step_name: &str,
        agent_name: &str,
        input: &serde_json::Value,
        timeout: Option<Duration>,
    ) -> Result<AgentExecutionResult, PipelineError> {
        if let Some(timeout) = timeout {
            tokio::time::timeout(timeout, self.execute_agent(agent_name, input))
                .await
                .map_err(|_| PipelineError::StepFailed {
                    name: step_name.to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("Task '{}' timed out after {:?}", step_name, timeout),
                    )),
                })?
        } else {
            self.execute_agent(agent_name, input).await
        }
    }

    /// Execute an agent with retry policy.
    async fn execute_with_retry(
        &self,
        step_name: &str,
        agent_name: &str,
        input: &serde_json::Value,
        retry_policy: &crate::step::RetryPolicy,
        timeout: Option<Duration>,
    ) -> Result<AgentExecutionResult, PipelineError> {
        let mut last_error = None;

        for attempt in 0..=retry_policy.max_retries {
            if attempt > 0 {
                if let Some(delay) = retry_policy.delay_for(attempt - 1) {
                    debug!("Task '{}' retry {} in {:?}", step_name, attempt, delay);
                    tokio::time::sleep(delay).await;
                }
            }

            match self
                .execute_agent_with_timeout(step_name, agent_name, input, timeout)
                .await
            {
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
        let concurrency_limit = self
            .parallel_step_concurrency_limit
            .unwrap_or_else(|| steps.len().max(1));
        let semaphore = Arc::new(Semaphore::new(concurrency_limit));

        let futures: Vec<_> = steps
            .iter()
            .zip(cloned_ctxs.into_iter())
            .map(|(step, mut ctx)| {
                let semaphore = semaphore.clone();
                async move {
                    let _permit = semaphore.acquire_owned().await.map_err(|_| {
                        PipelineError::Context(
                            "Parallel step concurrency limiter has been closed".to_string(),
                        )
                    })?;
                    self.execute_step(step, &mut ctx).await?;
                    Ok::<_, PipelineError>(ctx)
                }
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
    default_task_error_policy: Option<TaskErrorPolicy>,
    parallel_step_concurrency_limit: Option<usize>,
    workflow_concurrency_limit: Option<usize>,
    agent_executor: Option<AgentExecutorFn>,
    tracked_agent_executor: Option<TrackedAgentExecutorFn>,
    cost_tracker: CostTracker,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            timeout: None,
            continue_on_error: false,
            default_task_error_policy: None,
            parallel_step_concurrency_limit: None,
            workflow_concurrency_limit: None,
            agent_executor: None,
            tracked_agent_executor: None,
            cost_tracker: CostTracker::new(),
        }
    }

    /// Attach an agent executor — see [`Pipeline::with_agent_executor`].
    pub fn with_agent_executor<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut:
            std::future::Future<Output = Result<serde_json::Value, PipelineError>> + Send + 'static,
    {
        self.agent_executor = Some(Arc::new(move |name, input| Box::pin(f(name, input))));
        self
    }

    /// Attach a tracked agent executor — see [`Pipeline::with_tracked_agent_executor`].
    pub fn with_tracked_agent_executor<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut:
            std::future::Future<Output = Result<TrackedTaskOutput, PipelineError>> + Send + 'static,
    {
        self.tracked_agent_executor = Some(Arc::new(move |name, input| Box::pin(f(name, input))));
        self
    }

    /// Override the pipeline cost tracker.
    pub fn cost_tracker(mut self, cost_tracker: CostTracker) -> Self {
        self.cost_tracker = cost_tracker;
        self
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

    /// Set the default error policy for task steps.
    pub fn default_task_error_policy(mut self, policy: TaskErrorPolicy) -> Self {
        self.default_task_error_policy = Some(policy);
        self
    }

    /// Set the maximum number of branches that may run concurrently within a parallel step.
    pub fn parallel_step_concurrency_limit(mut self, limit: usize) -> Self {
        self.parallel_step_concurrency_limit = Some(limit);
        self
    }

    /// Set the maximum number of concurrent executions for this pipeline instance.
    pub fn workflow_concurrency_limit(mut self, limit: usize) -> Self {
        self.workflow_concurrency_limit = Some(limit);
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

        if self.parallel_step_concurrency_limit == Some(0) {
            return Err(PipelineError::Context(
                "Parallel step concurrency limit must be greater than zero".to_string(),
            ));
        }

        if self.workflow_concurrency_limit == Some(0) {
            return Err(PipelineError::Context(
                "Workflow concurrency limit must be greater than zero".to_string(),
            ));
        }

        Dag::from_steps(&self.steps).map_err(|e| PipelineError::Context(e.to_string()))?;

        let workflow_limit = self.workflow_concurrency_limit;

        Ok(Pipeline {
            name: self.name,
            steps: self.steps,
            timeout: self.timeout,
            continue_on_error: self.continue_on_error,
            default_task_error_policy: self.default_task_error_policy,
            parallel_step_concurrency_limit: self.parallel_step_concurrency_limit,
            workflow_concurrency_limit: workflow_limit,
            agent_executor: self.agent_executor,
            tracked_agent_executor: self.tracked_agent_executor,
            cost_tracker: self.cost_tracker,
            workflow_semaphore: workflow_limit.map(|limit| Arc::new(Semaphore::new(limit))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::TaskErrorPolicy;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn test_pipeline_builder_concurrency_limits() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "a", "i", "o"))
            .parallel_step_concurrency_limit(2)
            .workflow_concurrency_limit(3)
            .build()
            .unwrap();

        assert_eq!(pipeline.parallel_step_concurrency_limit, Some(2));
        assert_eq!(pipeline.workflow_concurrency_limit, Some(3));
    }

    #[tokio::test]
    async fn test_pipeline_execute_simple() {
        let pipeline = Pipeline::builder("test")
            .step(Step::task("s1", "agent", "input", "output"))
            .build()
            .unwrap()
            .with_agent_executor(|_agent, input| Box::pin(async move { Ok(input) }));

        let result = pipeline.execute(json!("test")).await.unwrap();
        assert_eq!(result.get("input"), Some(&json!("test")));
        // Executor passes input through to output
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
            .unwrap()
            .with_agent_executor(|_agent, input| Box::pin(async move { Ok(input) }));

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
            .unwrap()
            .with_agent_executor(|_agent, input| Box::pin(async move { Ok(input) }));

        let result = pipeline.execute(json!("input")).await.unwrap();
        // s1 sets decision to "input", so then_branch should run
        assert_eq!(result.get("decision"), Some(&json!("input")));
        // then_branch sets result to the value of decision
        assert_eq!(result.get("result"), Some(&json!("input")));
    }

    #[tokio::test]
    async fn test_pipeline_tracked_executor_records_cost_scopes() {
        let pipeline = Pipeline::builder("workflow")
            .step(Step::task("s1", "agent_a", "input", "output"))
            .with_tracked_agent_executor(|_agent, input| {
                Box::pin(async move {
                    Ok(TrackedTaskOutput {
                        output: input,
                        model: "gpt-4".to_string(),
                        usage: Usage {
                            prompt_tokens: 12,
                            completion_tokens: 6,
                            total_tokens: 18,
                        },
                    })
                })
            })
            .build()
            .unwrap();

        let result = pipeline.execute(json!("tracked")).await.unwrap();
        assert_eq!(result.get("output").cloned(), Some(json!("tracked")));

        let workflow_snapshot = pipeline.cost_tracker().get("workflow:workflow").await;
        assert_eq!(workflow_snapshot.request_count, 1);
        assert_eq!(workflow_snapshot.prompt_tokens, 12);

        let agent_snapshot = pipeline.cost_tracker().get("agent:agent_a").await;
        assert_eq!(agent_snapshot.completion_tokens, 6);

        let global_snapshot = pipeline.cost_tracker().get(GLOBAL_SCOPE).await;
        assert_eq!(global_snapshot.request_count, 1);
    }

    #[test]
    fn test_pipeline_builder_rejects_unknown_dependency() {
        let task = crate::step::Task::new("agent", "input", "output").depends_on("missing_step");

        let result = Pipeline::builder("workflow")
            .step(Step::task_with("dependent", task))
            .build();

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("unknown task 'missing_step'"));
    }

    #[tokio::test]
    async fn test_pipeline_execute_respects_task_dependencies_out_of_order() {
        let call_order = Arc::new(Mutex::new(Vec::new()));
        let order_ref = call_order.clone();

        let prepare = crate::step::Task::new("planner", "input", "prepared");
        let report =
            crate::step::Task::new("reporter", "prepared", "reported").depends_on("prepare");

        let pipeline = Pipeline::builder("workflow")
            .step(Step::task_with("report", report))
            .step(Step::task_with("prepare", prepare))
            .build()
            .unwrap()
            .with_agent_executor(move |agent, input| {
                let order_ref = order_ref.clone();
                Box::pin(async move {
                    order_ref.lock().unwrap().push(agent.clone());
                    Ok(json!({
                        "agent": agent,
                        "input": input,
                    }))
                })
            });

        let result = pipeline.execute(json!("seed")).await.unwrap();

        assert_eq!(
            call_order.lock().unwrap().clone(),
            vec!["planner".to_string(), "reporter".to_string()]
        );
        assert!(result.has("prepared"));
        assert!(result.has("reported"));
    }

    #[tokio::test]
    async fn test_pipeline_skips_failed_task_with_default_policy_and_propagates_error_context() {
        let pipeline = Pipeline::builder("workflow")
            .default_task_error_policy(TaskErrorPolicy::Skip)
            .step(Step::task("unstable", "broken", "input", "unused"))
            .step(Step::task_with(
                "recover",
                crate::step::Task::new("recovery", "unstable_error", "handled")
                    .depends_on("unstable"),
            ))
            .build()
            .unwrap()
            .with_agent_executor(|agent, input| {
                Box::pin(async move {
                    if agent == "broken" {
                        Err(PipelineError::Context("boom".to_string()))
                    } else {
                        Ok(input)
                    }
                })
            });

        let result = pipeline.execute(json!("seed")).await.unwrap();

        assert!(result.get("unstable_error").is_some());
        assert_eq!(result.get("handled"), result.get("unstable_error"));
    }

    #[tokio::test]
    async fn test_pipeline_uses_fallback_agent_on_task_failure() {
        let pipeline = Pipeline::builder("workflow")
            .step(Step::task_with(
                "primary",
                crate::step::Task::new("primary_agent", "input", "output")
                    .with_error_policy(TaskErrorPolicy::fallback("backup_agent")),
            ))
            .build()
            .unwrap()
            .with_agent_executor(|agent, input| {
                Box::pin(async move {
                    if agent == "primary_agent" {
                        Err(PipelineError::Context("primary failed".to_string()))
                    } else {
                        Ok(json!({"agent": agent, "input": input}))
                    }
                })
            });

        let result = pipeline.execute(json!("seed")).await.unwrap();

        assert_eq!(
            result.get("output"),
            Some(&json!({"agent": "backup_agent", "input": "seed"}))
        );
        assert_eq!(
            result.get("primary_error"),
            Some(&json!({
                "step": "primary",
                "agent_name": "primary_agent",
                "action": "fallback",
                "error": "Context error: primary failed",
                "fallback_agent": "backup_agent",
                "recovered": true,
            }))
        );
    }

    #[tokio::test]
    async fn test_pipeline_task_timeout_returns_error() {
        let pipeline = Pipeline::builder("workflow")
            .step(Step::task_with(
                "slow",
                crate::step::Task::new("slow_agent", "input", "output")
                    .with_timeout(Duration::from_millis(10)),
            ))
            .build()
            .unwrap()
            .with_agent_executor(|_agent, input| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                    Ok(input)
                })
            });

        let result = pipeline.execute(json!("seed")).await;
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn test_parallel_step_respects_concurrency_limit() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let pipeline = Pipeline::builder("workflow")
            .parallel_step_concurrency_limit(2)
            .parallel(vec![
                Step::task("a", "agent_a", "input", "out_a"),
                Step::task("b", "agent_b", "input", "out_b"),
                Step::task("c", "agent_c", "input", "out_c"),
            ])
            .build()
            .unwrap()
            .with_agent_executor({
                let active = active.clone();
                let max_active = max_active.clone();
                move |_agent, input| {
                    let active = active.clone();
                    let max_active = max_active.clone();
                    Box::pin(async move {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(current, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(input)
                    })
                }
            });

        let result = pipeline.execute(json!("seed")).await.unwrap();

        assert!(result.has("out_a"));
        assert!(result.has("out_b"));
        assert!(result.has("out_c"));
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_workflow_concurrency_limit_serializes_execute_calls() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let pipeline = Pipeline::builder("workflow")
            .workflow_concurrency_limit(1)
            .step(Step::task("task", "agent", "input", "output"))
            .build()
            .unwrap()
            .with_agent_executor({
                let active = active.clone();
                let max_active = max_active.clone();
                move |_agent, input| {
                    let active = active.clone();
                    let max_active = max_active.clone();
                    Box::pin(async move {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(current, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(input)
                    })
                }
            });

        let pipeline_clone = pipeline.clone();
        let (first, second) = tokio::join!(
            pipeline.execute(json!("first")),
            pipeline_clone.execute(json!("second"))
        );

        assert_eq!(first.unwrap().get("output"), Some(&json!("first")));
        assert_eq!(second.unwrap().get("output"), Some(&json!("second")));
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }
}
