//! Pipeline step definitions.
//!
//! Steps are the building blocks of pipelines. Each step can be a task,
//! parallel steps, a conditional branch, or a loop.

use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Retry strategy for task execution.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Fixed delay between retries.
    Fixed(Duration),

    /// Exponential backoff with a maximum delay.
    Exponential {
        /// Base delay between retries.
        base: Duration,
        /// Maximum delay cap.
        max: Duration,
    },
}

impl BackoffStrategy {
    /// Calculate the delay for a given retry attempt (0-indexed).
    pub fn delay(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Exponential { base, max } => {
                // Calculate exponential delay: base * 2^attempt
                let millis = base.as_millis() as u64;
                let exponential = millis.saturating_mul(2u64.saturating_pow(attempt.min(20)));
                let capped = exponential.min(max.as_millis() as u64);
                Duration::from_millis(capped)
            }
        }
    }

    /// Create a fixed backoff strategy.
    pub fn fixed(millis: u64) -> Self {
        Self::Fixed(Duration::from_millis(millis))
    }

    /// Create an exponential backoff strategy.
    pub fn exponential(base_millis: u64, max_millis: u64) -> Self {
        Self::Exponential {
            base: Duration::from_millis(base_millis),
            max: Duration::from_millis(max_millis),
        }
    }
}

/// Retry policy for task execution.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_retries: u32,

    /// Backoff strategy between retries.
    pub backoff: BackoffStrategy,
}

impl RetryPolicy {
    /// Create a new retry policy.
    pub fn new(max_retries: u32, backoff: BackoffStrategy) -> Self {
        Self {
            max_retries,
            backoff,
        }
    }

    /// Create a fixed-delay retry policy.
    pub fn fixed(max_retries: u32, delay_millis: u64) -> Self {
        Self {
            max_retries,
            backoff: BackoffStrategy::fixed(delay_millis),
        }
    }

    /// Create an exponential-backoff retry policy.
    pub fn exponential(max_retries: u32, base_millis: u64, max_millis: u64) -> Self {
        Self {
            max_retries,
            backoff: BackoffStrategy::exponential(base_millis, max_millis),
        }
    }

    /// Calculate the delay for a given retry attempt.
    pub fn delay_for(&self, attempt: u32) -> Option<Duration> {
        if attempt < self.max_retries {
            Some(self.backoff.delay(attempt))
        } else {
            None
        }
    }
}

/// A task definition with full configuration.
///
/// Tasks represent the basic unit of work in a pipeline, executing
/// an agent with specific input/output bindings and optional
/// validation, retry, and timeout policies.
#[derive(Debug, Clone)]
pub struct Task {
    /// Human-readable description of what this task does.
    pub description: Option<String>,

    /// Name of the agent to run.
    pub agent_name: String,

    /// Key in the context to read input from.
    pub input_key: String,

    /// Key in the context to write output to.
    pub output_key: String,

    /// Expected output schema/value for validation.
    pub expected_output: Option<Value>,

    /// Names of tasks this task depends on (must complete first).
    pub dependencies: Vec<String>,

    /// Optional timeout for this task.
    pub timeout: Option<Duration>,

    /// Optional retry policy for this task.
    pub retry_policy: Option<RetryPolicy>,
}

impl Task {
    /// Create a new task with minimal configuration.
    pub fn new(
        agent_name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            description: None,
            agent_name: agent_name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            expected_output: None,
            dependencies: Vec::new(),
            timeout: None,
            retry_policy: None,
        }
    }

    /// Set the task description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the expected output schema/value for validation.
    pub fn with_expected_output(mut self, expected: Value) -> Self {
        self.expected_output = Some(expected);
        self
    }

    /// Add a dependency on another task.
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Set multiple dependencies.
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Set the timeout for this task.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the retry policy for this task.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    /// Validate the task configuration.
    pub fn validate(&self) -> Result<(), TaskValidationError> {
        if self.agent_name.is_empty() {
            return Err(TaskValidationError::EmptyAgentName);
        }
        if self.input_key.is_empty() {
            return Err(TaskValidationError::EmptyInputKey);
        }
        if self.output_key.is_empty() {
            return Err(TaskValidationError::EmptyOutputKey);
        }
        Ok(())
    }

    /// Validate output against the expected output schema.
    pub fn validate_output(&self, actual: &Value) -> Result<(), TaskValidationError> {
        if let Some(expected) = &self.expected_output {
            if actual != expected {
                return Err(TaskValidationError::OutputMismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Errors that can occur during task validation.
#[derive(Debug, thiserror::Error)]
pub enum TaskValidationError {
    /// Agent name is empty.
    #[error("Agent name cannot be empty")]
    EmptyAgentName,

    /// Input key is empty.
    #[error("Input key cannot be empty")]
    EmptyInputKey,

    /// Output key is empty.
    #[error("Output key cannot be empty")]
    EmptyOutputKey,

    /// Output does not match expected value.
    #[error("Output mismatch: expected {expected}, got {actual}")]
    OutputMismatch { expected: Value, actual: Value },

    /// Circular dependency detected.
    #[error("Circular dependency detected: {0}")]
    CircularDependency(String),
}

/// A single step in a pipeline.
///
/// Steps define what work should be done during pipeline execution.
/// Each step has a name for identification and a kind that determines
/// how it executes.
#[derive(Clone)]
pub struct Step {
    /// Unique name for this step.
    pub name: String,

    /// The kind of step and its configuration.
    pub kind: StepKind,
}

impl Step {
    /// Create a new task step with minimal configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    /// use serde_json::json;
    ///
    /// let step = Step::task(
    ///     "my_task",
    ///     "my_agent",
    ///     "input",
    ///     "output"
    /// );
    /// ```
    pub fn task(
        name: impl Into<String>,
        agent_name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Task(Task::new(agent_name, input_key, output_key)),
        }
    }

    /// Create a new task step with a full Task configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::{Step, Task, RetryPolicy};
    /// use std::time::Duration;
    ///
    /// let task = Task::new("agent", "input", "output")
    ///     .with_description("Process the data")
    ///     .with_timeout(Duration::from_secs(30))
    ///     .with_retry_policy(RetryPolicy::fixed(3, 1000));
    ///
    /// let step = Step::task_with("my_task", task);
    /// ```
    pub fn task_with(name: impl Into<String>, task: Task) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Task(task),
        }
    }

    /// Create a parallel step that runs multiple steps concurrently.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    ///
    /// let step1 = Step::task("task1", "agent1", "in", "out1");
    /// let step2 = Step::task("task2", "agent2", "in", "out2");
    /// let parallel = Step::parallel("parallel_steps", vec![step1, step2]);
    /// ```
    pub fn parallel(name: impl Into<String>, steps: Vec<Step>) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Parallel(steps),
        }
    }

    /// Create a conditional step with a closure-based condition.
    ///
    /// The condition function receives the pipeline context and returns
    /// `true` to execute the `then` step, or `false` to execute the `else` step (if provided).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// # fn main() {
    /// let then_step = Step::task("then_task", "agent", "in", "out");
    /// let else_step = Step::task("else_task", "agent", "in", "out");
    ///
    /// let conditional = Step::conditional_fn(
    ///     "check_condition",
    ///     |ctx: &PipelineContext| {
    ///         ctx.get("should_proceed")
    ///             .and_then(|v| v.as_bool())
    ///             .unwrap_or(false)
    ///     },
    ///     then_step,
    ///     Some(else_step)
    /// );
    /// # }
    /// ```
    pub fn conditional_fn(
        name: impl Into<String>,
        condition: impl Fn(&crate::PipelineContext) -> bool + Send + Sync + 'static,
        then_step: Step,
        else_step: Option<Step>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Conditional {
                condition: Condition::Fn(Arc::new(condition)),
                then_step: Box::new(then_step),
                else_step: else_step.map(Box::new),
            },
        }
    }

    /// Create a conditional step that checks if a context key matches a value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    /// use serde_json::json;
    ///
    /// let then_step = Step::task("then_task", "agent", "in", "out");
    /// let else_step = Step::task("else_task", "agent", "in", "out");
    ///
    /// let conditional = Step::conditional_value(
    ///     "check_value",
    ///     "status",
    ///     json!("approved"),
    ///     then_step,
    ///     Some(else_step)
    /// );
    /// ```
    pub fn conditional_value(
        name: impl Into<String>,
        key: impl Into<String>,
        value: Value,
        then_step: Step,
        else_step: Option<Step>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Conditional {
                condition: Condition::Key {
                    key: key.into(),
                    value,
                },
                then_step: Box::new(then_step),
                else_step: else_step.map(Box::new),
            },
        }
    }

    /// Create a loop step with a closure-based condition.
    ///
    /// The loop continues while the condition function returns `true`,
    /// or until `max_iterations` is reached.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    /// use ai_pipeline::PipelineContext;
    ///
    /// # fn main() {
    /// let body = Step::task("loop_task", "agent", "in", "out");
    ///
    /// let loop_step = Step::loop_fn(
    ///     "retry_loop",
    ///     body,
    ///     |ctx: &PipelineContext| {
    ///         // Continue while retry_count < 3
    ///         ctx.get("retry_count")
    ///             .and_then(|v| v.as_u64())
    ///             .map(|c| c < 3)
    ///             .unwrap_or(false)
    ///     },
    ///     10
    /// );
    /// # }
    /// ```
    pub fn loop_fn(
        name: impl Into<String>,
        body: Step,
        condition: impl Fn(&crate::PipelineContext) -> bool + Send + Sync + 'static,
        max_iterations: u32,
    ) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Loop {
                body: Box::new(body),
                condition: LoopCondition::Fn(Arc::new(condition)),
                max_iterations,
            },
        }
    }

    /// Create a loop step that continues while a context key matches a value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::Step;
    /// use serde_json::json;
    ///
    /// let body = Step::task("process_item", "agent", "item", "result");
    ///
    /// let loop_step = Step::loop_value(
    ///     "process_all",
    ///     body,
    ///     "has_more",
    ///     json!(true),
    ///     100
    /// );
    /// ```
    pub fn loop_value(
        name: impl Into<String>,
        body: Step,
        key: impl Into<String>,
        value: Value,
        max_iterations: u32,
    ) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Loop {
                body: Box::new(body),
                condition: LoopCondition::Key {
                    key: key.into(),
                    value,
                },
                max_iterations,
            },
        }
    }
}

impl fmt::Debug for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Step")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .finish()
    }
}

/// The kind of step and its configuration.
#[derive(Clone)]
pub enum StepKind {
    /// A task step that runs an agent.
    Task(Task),

    /// Runs multiple steps concurrently.
    Parallel(Vec<Step>),

    /// Conditionally executes one of two branches.
    Conditional {
        /// The condition to evaluate.
        condition: Condition,
        /// Step to execute if condition is true.
        then_step: Box<Step>,
        /// Optional step to execute if condition is false.
        else_step: Option<Box<Step>>,
    },

    /// Repeats a step until a condition is met or max iterations reached.
    Loop {
        /// The step to repeat.
        body: Box<Step>,
        /// The loop condition.
        condition: LoopCondition,
        /// Maximum number of iterations before giving up.
        max_iterations: u32,
    },
}

impl fmt::Debug for StepKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(task) => f
                .debug_struct("Task")
                .field("agent_name", &task.agent_name)
                .field("input_key", &task.input_key)
                .field("output_key", &task.output_key)
                .finish(),
            Self::Parallel(steps) => f.debug_tuple("Parallel").field(&steps.len()).finish(),
            Self::Conditional { .. } => f.debug_tuple("Conditional").finish(),
            Self::Loop { .. } => f.debug_tuple("Loop").finish(),
        }
    }
}

/// Condition for conditional steps.
pub enum Condition {
    /// Function-based condition.
    Fn(Arc<dyn Fn(&crate::PipelineContext) -> bool + Send + Sync>),

    /// Key-value comparison condition.
    Key {
        /// The key to look up in the context.
        key: String,
        /// The value to compare against.
        value: Value,
    },
}

impl Clone for Condition {
    fn clone(&self) -> Self {
        match self {
            Self::Fn(f) => Self::Fn(Arc::clone(f)),
            Self::Key { key, value } => Self::Key {
                key: key.clone(),
                value: value.clone(),
            },
        }
    }
}

/// Condition for loop steps.
pub enum LoopCondition {
    /// Function-based condition.
    Fn(Arc<dyn Fn(&crate::PipelineContext) -> bool + Send + Sync>),

    /// Key-value comparison condition.
    Key {
        /// The key to look up in the context.
        key: String,
        /// The value to compare against (loop continues while equal).
        value: Value,
    },
}

impl Clone for LoopCondition {
    fn clone(&self) -> Self {
        match self {
            Self::Fn(f) => Self::Fn(Arc::clone(f)),
            Self::Key { key, value } => Self::Key {
                key: key.clone(),
                value: value.clone(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_creation() {
        let step = Step::task("test", "agent", "in", "out");
        assert_eq!(step.name, "test");
        match &step.kind {
            StepKind::Task(task) => {
                assert_eq!(task.agent_name, "agent");
                assert_eq!(task.input_key, "in");
                assert_eq!(task.output_key, "out");
            }
            _ => panic!("Expected Task step"),
        }
    }

    #[test]
    fn test_parallel_step() {
        let step1 = Step::task("t1", "a", "i", "o");
        let step2 = Step::task("t2", "b", "i", "o");
        let parallel = Step::parallel("parallel", vec![step1, step2]);
        assert_eq!(parallel.name, "parallel");
        match &parallel.kind {
            StepKind::Parallel(steps) => assert_eq!(steps.len(), 2),
            _ => panic!("Expected Parallel step"),
        }
    }

    #[test]
    fn test_conditional_fn() {
        let then_step = Step::task("then", "a", "i", "o");
        let else_step = Step::task("else", "b", "i", "o");
        let cond = Step::conditional_fn("cond", |_| true, then_step, Some(else_step));
        assert_eq!(cond.name, "cond");
        match &cond.kind {
            StepKind::Conditional { .. } => {}
            _ => panic!("Expected Conditional step"),
        }
    }

    #[test]
    fn test_conditional_value() {
        let then_step = Step::task("then", "a", "i", "o");
        let cond = Step::conditional_value("cond", "key", serde_json::json!(true), then_step, None);
        assert_eq!(cond.name, "cond");
        match &cond.kind {
            StepKind::Conditional { .. } => {}
            _ => panic!("Expected Conditional step"),
        }
    }

    #[test]
    fn test_loop_fn() {
        let body = Step::task("body", "a", "i", "o");
        let loop_step = Step::loop_fn("loop_step", body, |_| false, 10);
        assert_eq!(loop_step.name, "loop_step");
        match &loop_step.kind {
            StepKind::Loop { max_iterations, .. } => assert_eq!(*max_iterations, 10),
            _ => panic!("Expected Loop step"),
        }
    }

    #[test]
    fn test_loop_value() {
        let body = Step::task("body", "a", "i", "o");
        let loop_step = Step::loop_value("loop_step", body, "key", serde_json::json!(true), 100);
        assert_eq!(loop_step.name, "loop_step");
        match &loop_step.kind {
            StepKind::Loop { max_iterations, .. } => assert_eq!(*max_iterations, 100),
            _ => panic!("Expected Loop step"),
        }
    }
}
