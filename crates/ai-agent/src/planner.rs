// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent planning capabilities.
//!
//! This module provides support for agents to create and execute multi-step plans.

use ai_core::error::AgentError;
use serde::{Deserialize, Serialize};

/// A single step in an agent's plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Unique identifier for this step.
    pub id: String,

    /// Description of what this step does.
    pub description: String,

    /// IDs of steps this step depends on.
    pub dependencies: Vec<String>,

    /// Current status of this step.
    pub status: PlanStepStatus,

    /// Result of this step (once completed).
    pub result: Option<String>,
}

impl PlanStep {
    /// Create a new plan step.
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            dependencies,
            status: PlanStepStatus::Pending,
            result: None,
        }
    }
}

/// Status of a plan step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanStepStatus {
    /// Step is pending execution.
    Pending,

    /// Step is currently running.
    InProgress,

    /// Step completed successfully.
    Completed,

    /// Step failed.
    Failed,
}

/// A plan that an agent can execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Unique identifier for this plan.
    pub id: String,

    /// Description of the plan's goal.
    pub goal: String,

    /// Steps in the plan.
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Create a new plan.
    pub fn new(id: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            goal: goal.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step to this plan.
    pub fn add_step(mut self, step: PlanStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Check if this plan has circular dependencies.
    ///
    /// Returns `true` if circular dependencies are detected.
    pub fn has_circular_dependencies(&self) -> bool {
        // Simple DFS-based cycle detection
        for step in &self.steps {
            if self.has_cycle_from(&step.id, &step.id) {
                return true;
            }
        }
        false
    }

    fn has_cycle_from(&self, start: &str, current: &str) -> bool {
        if let Some(step) = self.steps.iter().find(|s| s.id == current) {
            for dep in &step.dependencies {
                if dep == start {
                    return true;
                }
                if self.has_cycle_from(start, dep) {
                    return true;
                }
            }
        }
        false
    }

    /// Get steps in topological order (dependencies before dependents).
    ///
    /// Returns `None` if circular dependencies are detected.
    pub fn execution_order(&self) -> Option<Vec<PlanStep>> {
        if self.has_circular_dependencies() {
            return None;
        }

        let mut result = Vec::new();
        let mut remaining = self.steps.clone();

        while !remaining.is_empty() {
            // Find steps with no unmet dependencies
            let ready_ids: Vec<String> = remaining
                .iter()
                .filter(|step| {
                    step.dependencies
                        .iter()
                        .all(|dep| result.iter().any(|s: &PlanStep| s.id == *dep))
                })
                .map(|s| s.id.clone())
                .collect();

            if ready_ids.is_empty() {
                // Circular dependency or other issue
                return None;
            }

            for id in ready_ids {
                if let Some(pos) = remaining.iter().position(|s| s.id == id) {
                    result.push(remaining.remove(pos));
                }
            }
        }

        Some(result)
    }
}

/// Result of a plan execution.
#[derive(Debug)]
pub struct PlanResult {
    /// Whether the plan completed successfully.
    pub success: bool,

    /// Results from each step.
    pub step_results: Vec<(String, String)>,

    /// Steps that failed and were re-planned.
    pub replanned_steps: Vec<String>,
}

/// Executor that runs plans step by step with re-planning support.
pub struct PlanExecutor {
    /// Maximum number of re-plan attempts per step.
    pub max_retries: u32,
}

impl PlanExecutor {
    /// Create a new plan executor with a maximum retry count.
    pub fn new(max_retries: u32) -> Self {
        Self { max_retries }
    }

    /// Execute a plan using the provided step executor function.
    ///
    /// The `step_fn` receives a step description and returns a result string or error.
    /// On failure, the executor will retry up to `max_retries` times.
    pub async fn execute<F, Fut>(
        &self,
        plan: &mut Plan,
        step_fn: F,
    ) -> Result<PlanResult, AgentError>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<String, String>> + Send,
    {
        let order = plan
            .execution_order()
            .ok_or_else(|| AgentError::PlanError("Circular dependencies detected".to_string()))?;

        let mut step_results = Vec::new();
        let mut replanned_steps = Vec::new();

        for step_template in &order {
            let step = plan
                .steps
                .iter_mut()
                .find(|s| s.id == step_template.id)
                .unwrap();

            step.status = PlanStepStatus::InProgress;

            let mut attempts = 0;
            let mut last_error = String::new();
            let mut success = false;

            while attempts <= self.max_retries {
                match step_fn(step.description.clone()).await {
                    Ok(result) => {
                        step.status = PlanStepStatus::Completed;
                        step.result = Some(result.clone());
                        step_results.push((step.id.clone(), result));
                        success = true;
                        break;
                    }
                    Err(err) => {
                        last_error = err;
                        attempts += 1;
                        if attempts <= self.max_retries {
                            replanned_steps.push(step.id.clone());
                        }
                    }
                }
            }

            if !success {
                step.status = PlanStepStatus::Failed;
                step.result = Some(last_error.clone());
                step_results.push((step.id.clone(), format!("FAILED: {}", last_error)));

                return Ok(PlanResult {
                    success: false,
                    step_results,
                    replanned_steps,
                });
            }
        }

        Ok(PlanResult {
            success: true,
            step_results,
            replanned_steps,
        })
    }
}

impl Default for PlanExecutor {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_step_creation() {
        let step = PlanStep::new("step1", "First step", vec![]);

        assert_eq!(step.id, "step1");
        assert_eq!(step.description, "First step");
        assert_eq!(step.dependencies, Vec::<String>::new());
        assert_eq!(step.status, PlanStepStatus::Pending);
        assert!(step.result.is_none());
    }

    #[test]
    fn test_plan_step_with_dependencies() {
        let step = PlanStep::new("step2", "Second step", vec!["step1".to_string()]);

        assert_eq!(step.dependencies.len(), 1);
        assert_eq!(step.dependencies[0], "step1");
    }

    #[test]
    fn test_plan_step_status_changes() {
        let mut step = PlanStep::new("step1", "Test", vec![]);

        assert_eq!(step.status, PlanStepStatus::Pending);

        step.status = PlanStepStatus::InProgress;
        assert_eq!(step.status, PlanStepStatus::InProgress);

        step.status = PlanStepStatus::Completed;
        step.result = Some("Done".to_string());
        assert_eq!(step.status, PlanStepStatus::Completed);
        assert_eq!(step.result.as_ref().unwrap(), "Done");
    }

    #[test]
    fn test_plan_creation() {
        let plan = Plan::new("plan1", "Test plan");

        assert_eq!(plan.id, "plan1");
        assert_eq!(plan.goal, "Test plan");
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_plan_add_step() {
        let plan =
            Plan::new("plan1", "Test plan").add_step(PlanStep::new("step1", "First step", vec![]));

        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].id, "step1");
    }

    #[test]
    fn test_plan_no_circular_dependencies() {
        let plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("step1", "First", vec![]))
            .add_step(PlanStep::new("step2", "Second", vec!["step1".to_string()]))
            .add_step(PlanStep::new("step3", "Third", vec!["step2".to_string()]));

        assert!(!plan.has_circular_dependencies());
    }

    #[test]
    fn test_plan_detects_circular_dependencies() {
        let plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("step1", "First", vec!["step2".to_string()]))
            .add_step(PlanStep::new("step2", "Second", vec!["step1".to_string()]));

        assert!(plan.has_circular_dependencies());
    }

    #[test]
    fn test_plan_execution_order_simple() {
        let plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("step1", "First", vec![]))
            .add_step(PlanStep::new("step2", "Second", vec!["step1".to_string()]));

        let order = plan.execution_order();

        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].id, "step1");
        assert_eq!(order[1].id, "step2");
    }

    #[test]
    fn test_plan_execution_order_complex() {
        let plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("a", "Task A", vec![]))
            .add_step(PlanStep::new("b", "Task B", vec![]))
            .add_step(PlanStep::new(
                "c",
                "Task C",
                vec!["a".to_string(), "b".to_string()],
            ))
            .add_step(PlanStep::new("d", "Task D", vec!["c".to_string()]));

        let order = plan.execution_order();

        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.len(), 4);

        // A and B should come before C
        let a_pos = order.iter().position(|s| s.id == "a").unwrap();
        let b_pos = order.iter().position(|s| s.id == "b").unwrap();
        let c_pos = order.iter().position(|s| s.id == "c").unwrap();
        let d_pos = order.iter().position(|s| s.id == "d").unwrap();

        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
        assert!(c_pos < d_pos);
    }

    #[test]
    fn test_plan_execution_order_with_cycle_returns_none() {
        let plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("a", "Task A", vec!["b".to_string()]))
            .add_step(PlanStep::new("b", "Task B", vec!["a".to_string()]));

        let order = plan.execution_order();
        assert!(order.is_none());
    }

    #[test]
    fn test_plan_step_status_serialization() {
        // Verify that status variants can be compared
        assert_eq!(PlanStepStatus::Pending, PlanStepStatus::Pending);
        assert_ne!(PlanStepStatus::Pending, PlanStepStatus::Completed);
        assert_ne!(PlanStepStatus::InProgress, PlanStepStatus::Failed);
    }

    #[tokio::test]
    async fn test_plan_executor_success() {
        let executor = PlanExecutor::new(2);
        let mut plan = Plan::new("plan1", "Test plan")
            .add_step(PlanStep::new("s1", "Step 1", vec![]))
            .add_step(PlanStep::new("s2", "Step 2", vec!["s1".to_string()]));

        let result = executor
            .execute(
                &mut plan,
                |desc| async move { Ok(format!("done: {}", desc)) },
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.step_results.len(), 2);
        assert!(result.replanned_steps.is_empty());

        // Verify steps are marked completed
        assert_eq!(plan.steps[0].status, PlanStepStatus::Completed);
        assert_eq!(plan.steps[1].status, PlanStepStatus::Completed);
    }

    #[tokio::test]
    async fn test_plan_executor_failure_after_retries() {
        let executor = PlanExecutor::new(1);
        let mut plan =
            Plan::new("plan1", "Test plan").add_step(PlanStep::new("s1", "Failing step", vec![]));

        let result = executor
            .execute(&mut plan, |_desc| async move {
                Err::<String, String>("always fails".to_string())
            })
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(plan.steps[0].status, PlanStepStatus::Failed);
        assert!(!result.replanned_steps.is_empty());
    }

    #[tokio::test]
    async fn test_plan_executor_retry_then_succeed() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let executor = PlanExecutor::new(2);
        let mut plan =
            Plan::new("plan1", "Test").add_step(PlanStep::new("s1", "Flaky step", vec![]));

        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = executor
            .execute(&mut plan, move |_desc| {
                let ac = attempt_count_clone.clone();
                async move {
                    let attempt = ac.fetch_add(1, Ordering::SeqCst);
                    if attempt < 2 {
                        Err("transient error".to_string())
                    } else {
                        Ok("success on retry".to_string())
                    }
                }
            })
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
        assert!(!result.replanned_steps.is_empty()); // Steps were replanned
    }

    #[tokio::test]
    async fn test_plan_executor_circular_dependency_error() {
        let executor = PlanExecutor::new(0);
        let mut plan = Plan::new("plan1", "Test")
            .add_step(PlanStep::new("a", "A", vec!["b".to_string()]))
            .add_step(PlanStep::new("b", "B", vec!["a".to_string()]));

        let result = executor
            .execute(&mut plan, |_| async { Ok("done".to_string()) })
            .await;

        assert!(result.is_err());
    }
}
