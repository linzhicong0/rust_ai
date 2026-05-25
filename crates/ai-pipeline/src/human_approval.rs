//! Human-in-the-Loop approval step for pipelines (REQ-7.5).
//!
//! Provides a [`HumanApproval`] step type that pauses pipeline execution
//! until a human approves or rejects the request, with configurable timeout handling.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// What to do when approval times out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutPolicy {
    /// Automatically reject the request on timeout.
    AutoReject,
    /// Automatically approve the request on timeout.
    AutoApprove,
}

/// Decision made by a human reviewer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// The request was approved.
    Approved,
    /// The request was rejected with an optional reason.
    Rejected(Option<String>),
}

impl ApprovalDecision {
    /// Returns true if the decision is approval.
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// Returns true if the decision is rejection.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }

    /// Get the rejection reason, if any.
    pub fn rejection_reason(&self) -> Option<&str> {
        match self {
            Self::Rejected(reason) => reason.as_deref(),
            _ => None,
        }
    }
}

/// Request sent to the approval callback.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The step name requesting approval.
    pub step_name: String,
    /// Description of what is being approved.
    pub description: String,
    /// The data to be reviewed (from pipeline context).
    pub payload: serde_json::Value,
}

/// Callback interface for UI integration.
///
/// Implementations receive an [`ApprovalRequest`] and must return an [`ApprovalDecision`].
/// This trait is object-safe and can be used with dynamic dispatch.
#[async_trait::async_trait]
pub trait ApprovalCallback: Send + Sync {
    /// Called when the pipeline needs human approval.
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalDecision;
}

/// Function-based approval callback adapter.
pub struct FnApprovalCallback<F>(pub F)
where
    F: Fn(
            ApprovalRequest,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApprovalDecision> + Send>>
        + Send
        + Sync;

#[async_trait::async_trait]
impl<F> ApprovalCallback for FnApprovalCallback<F>
where
    F: Fn(
            ApprovalRequest,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ApprovalDecision> + Send>>
        + Send
        + Sync,
{
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalDecision {
        (self.0)(request).await
    }
}

/// A human approval step in a pipeline.
///
/// This step pauses pipeline execution and sends an approval request
/// to the configured callback. Execution continues based on the human's
/// decision or the timeout policy.
#[derive(Clone)]
pub struct HumanApproval {
    /// Name of this approval step.
    pub name: String,

    /// Description shown to the reviewer.
    pub description: String,

    /// Key in context to include in the approval payload.
    pub context_key: String,

    /// Timeout before the timeout policy is applied.
    pub timeout: Duration,

    /// What to do when the timeout expires.
    pub timeout_policy: TimeoutPolicy,

    /// The callback to invoke for approval decisions.
    callback: Arc<dyn ApprovalCallback>,
}

impl fmt::Debug for HumanApproval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HumanApproval")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("context_key", &self.context_key)
            .field("timeout", &self.timeout)
            .field("timeout_policy", &self.timeout_policy)
            .finish()
    }
}

/// Result of executing a human approval step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalResult {
    /// Pipeline should continue (approved).
    Continue,
    /// Pipeline should stop (rejected).
    Halt(String),
    /// Timeout occurred and policy was applied.
    TimedOut(TimeoutPolicy),
}

impl HumanApproval {
    /// Create a new human approval step.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        context_key: impl Into<String>,
        timeout: Duration,
        timeout_policy: TimeoutPolicy,
        callback: Arc<dyn ApprovalCallback>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            context_key: context_key.into(),
            timeout,
            timeout_policy,
            callback,
        }
    }

    /// Execute the approval step.
    ///
    /// Sends an approval request to the callback and waits for a decision
    /// or timeout. Returns an [`ApprovalResult`] indicating whether the
    /// pipeline should continue or halt.
    pub async fn execute(&self, context: &crate::PipelineContext) -> ApprovalResult {
        let payload = context
            .get(&self.context_key)
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let request = ApprovalRequest {
            step_name: self.name.clone(),
            description: self.description.clone(),
            payload,
        };

        let callback = self.callback.clone();
        let timeout = self.timeout;

        let decision = tokio::time::timeout(timeout, callback.request_approval(request)).await;

        match decision {
            Ok(ApprovalDecision::Approved) => ApprovalResult::Continue,
            Ok(ApprovalDecision::Rejected(reason)) => {
                ApprovalResult::Halt(reason.unwrap_or_else(|| "Rejected by reviewer".to_string()))
            }
            Err(_timeout) => match self.timeout_policy {
                TimeoutPolicy::AutoReject => ApprovalResult::TimedOut(TimeoutPolicy::AutoReject),
                TimeoutPolicy::AutoApprove => ApprovalResult::TimedOut(TimeoutPolicy::AutoApprove),
            },
        }
    }

    /// Returns whether the approval result allows the pipeline to continue.
    pub fn should_continue(result: &ApprovalResult) -> bool {
        matches!(
            result,
            ApprovalResult::Continue | ApprovalResult::TimedOut(TimeoutPolicy::AutoApprove)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelineContext;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Simple callback that always approves.
    struct AlwaysApprove;

    #[async_trait::async_trait]
    impl ApprovalCallback for AlwaysApprove {
        async fn request_approval(&self, _request: ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Approved
        }
    }

    /// Callback that always rejects with a reason.
    struct AlwaysReject {
        reason: String,
    }

    #[async_trait::async_trait]
    impl ApprovalCallback for AlwaysReject {
        async fn request_approval(&self, _request: ApprovalRequest) -> ApprovalDecision {
            ApprovalDecision::Rejected(Some(self.reason.clone()))
        }
    }

    /// Callback that never responds (hangs forever).
    struct NeverRespond;

    #[async_trait::async_trait]
    impl ApprovalCallback for NeverRespond {
        async fn request_approval(&self, _request: ApprovalRequest) -> ApprovalDecision {
            // Simulate a callback that never returns
            tokio::time::sleep(Duration::from_secs(3600)).await;
            ApprovalDecision::Approved
        }
    }

    /// Callback that tracks if it was called.
    struct TrackingCallback {
        called: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl ApprovalCallback for TrackingCallback {
        async fn request_approval(&self, _request: ApprovalRequest) -> ApprovalDecision {
            self.called.store(true, Ordering::SeqCst);
            ApprovalDecision::Approved
        }
    }

    // REQ-7.5: Unit: HumanApproval step pauses pipeline until approval is received
    #[tokio::test]
    async fn test_human_approval_pauses_until_approval() {
        let approval = HumanApproval::new(
            "review_step",
            "Please review this output",
            "output_data",
            Duration::from_secs(60),
            TimeoutPolicy::AutoReject,
            Arc::new(AlwaysApprove),
        );

        let mut ctx = PipelineContext::empty();
        ctx.set("output_data", serde_json::json!({"result": "test"}));

        let result = approval.execute(&ctx).await;
        assert_eq!(result, ApprovalResult::Continue);
    }

    // REQ-7.5: Unit: approval callback resumes pipeline execution
    #[tokio::test]
    async fn test_approval_callback_resumes_pipeline() {
        let called = Arc::new(AtomicBool::new(false));
        let callback = TrackingCallback {
            called: called.clone(),
        };

        let approval = HumanApproval::new(
            "review_step",
            "Approve deployment",
            "deploy_config",
            Duration::from_secs(60),
            TimeoutPolicy::AutoReject,
            Arc::new(callback),
        );

        let ctx = PipelineContext::empty();
        let result = approval.execute(&ctx).await;

        assert!(
            called.load(Ordering::SeqCst),
            "Callback should have been called"
        );
        assert_eq!(result, ApprovalResult::Continue);
        assert!(HumanApproval::should_continue(&result));
    }

    // REQ-7.5: Unit: rejection callback stops pipeline with rejection reason
    #[tokio::test]
    async fn test_rejection_stops_pipeline_with_reason() {
        let approval = HumanApproval::new(
            "review_step",
            "Approve action",
            "action_data",
            Duration::from_secs(60),
            TimeoutPolicy::AutoReject,
            Arc::new(AlwaysReject {
                reason: "Content violates policy".to_string(),
            }),
        );

        let ctx = PipelineContext::empty();
        let result = approval.execute(&ctx).await;

        assert_eq!(
            result,
            ApprovalResult::Halt("Content violates policy".to_string())
        );
        assert!(!HumanApproval::should_continue(&result));
    }

    // REQ-7.5: Unit: timeout of 5 minutes with auto-reject policy rejects after timeout
    #[tokio::test]
    async fn test_timeout_with_auto_reject_policy() {
        let approval = HumanApproval::new(
            "review_step",
            "Approve action",
            "action_data",
            Duration::from_millis(50), // Short timeout for test
            TimeoutPolicy::AutoReject,
            Arc::new(NeverRespond),
        );

        let ctx = PipelineContext::empty();
        let result = approval.execute(&ctx).await;

        assert_eq!(result, ApprovalResult::TimedOut(TimeoutPolicy::AutoReject));
        assert!(!HumanApproval::should_continue(&result));
    }

    // REQ-7.5: Test auto-approve timeout policy
    #[tokio::test]
    async fn test_timeout_with_auto_approve_policy() {
        let approval = HumanApproval::new(
            "review_step",
            "Approve action",
            "action_data",
            Duration::from_millis(50), // Short timeout for test
            TimeoutPolicy::AutoApprove,
            Arc::new(NeverRespond),
        );

        let ctx = PipelineContext::empty();
        let result = approval.execute(&ctx).await;

        assert_eq!(result, ApprovalResult::TimedOut(TimeoutPolicy::AutoApprove));
        assert!(HumanApproval::should_continue(&result));
    }

    // REQ-7.5: Integration: UI callback receives approval request and returns decision
    #[tokio::test]
    async fn test_ui_callback_receives_request_and_returns_decision() {
        /// A callback simulating a UI that inspects the request.
        struct UiCallback;

        #[async_trait::async_trait]
        impl ApprovalCallback for UiCallback {
            async fn request_approval(&self, request: ApprovalRequest) -> ApprovalDecision {
                // UI inspects the request
                assert_eq!(request.step_name, "deploy_review");
                assert_eq!(request.description, "Approve production deployment");
                assert_eq!(
                    request.payload,
                    serde_json::json!({"env": "prod", "version": "1.2.3"})
                );

                // UI returns approval
                ApprovalDecision::Approved
            }
        }

        let approval = HumanApproval::new(
            "deploy_review",
            "Approve production deployment",
            "deploy_info",
            Duration::from_secs(300), // 5 minute timeout
            TimeoutPolicy::AutoReject,
            Arc::new(UiCallback),
        );

        let mut ctx = PipelineContext::empty();
        ctx.set(
            "deploy_info",
            serde_json::json!({"env": "prod", "version": "1.2.3"}),
        );

        let result = approval.execute(&ctx).await;
        assert_eq!(result, ApprovalResult::Continue);
    }
}
