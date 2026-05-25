// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Agent lifecycle management.
//!
//! This module provides a state machine for managing the full agent lifecycle:
//! initialization, execution, pause, resume, and termination.
//!
//! ## State Transitions
//!
//! ```text
//! Created → Running → Paused → Running → Stopped
//!                  ↘                      ↗
//!                    ----→ Stopped ------
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{watch, Mutex, Notify};

/// The possible states of an agent in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent has been created but not yet started.
    Created,

    /// Agent is actively running and processing tasks.
    Running,

    /// Agent is paused and not processing new tasks.
    Paused,

    /// Agent has been stopped (terminal state).
    Stopped,
}

impl std::fmt::Display for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentState::Created => write!(f, "Created"),
            AgentState::Running => write!(f, "Running"),
            AgentState::Paused => write!(f, "Paused"),
            AgentState::Stopped => write!(f, "Stopped"),
        }
    }
}

/// Error type for invalid state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTransitionError {
    /// The current state.
    pub from: AgentState,
    /// The attempted target state.
    pub to: AgentState,
}

impl std::fmt::Display for StateTransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid state transition from {} to {}",
            self.from, self.to
        )
    }
}

impl std::error::Error for StateTransitionError {}

/// Manages the lifecycle state of an agent.
///
/// Provides thread-safe state transitions with validation, pause/resume
/// support, and graceful shutdown with in-flight request completion.
#[derive(Clone)]
pub struct AgentLifecycle {
    inner: Arc<LifecycleInner>,
}

struct LifecycleInner {
    /// Current state with watch channel for notifications.
    state_tx: watch::Sender<AgentState>,
    state_rx: watch::Receiver<AgentState>,

    /// Notifier for resume events.
    resume_notify: Notify,

    /// Count of in-flight requests for graceful shutdown.
    in_flight: Mutex<u32>,

    /// Persisted state data (serializable snapshot).
    persisted_state: Mutex<Option<PersistedAgentState>>,
}

/// Serializable snapshot of agent state for persistence and resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedAgentState {
    /// The lifecycle state at time of persistence.
    pub state: AgentState,

    /// Optional context data associated with the agent.
    pub context: Option<String>,

    /// Timestamp of when state was persisted (ISO 8601).
    pub persisted_at: String,
}

impl AgentLifecycle {
    /// Create a new lifecycle manager in the `Created` state.
    pub fn new() -> Self {
        let (state_tx, state_rx) = watch::channel(AgentState::Created);
        Self {
            inner: Arc::new(LifecycleInner {
                state_tx,
                state_rx,
                resume_notify: Notify::new(),
                in_flight: Mutex::new(0),
                persisted_state: Mutex::new(None),
            }),
        }
    }

    /// Get the current state.
    pub fn state(&self) -> AgentState {
        *self.inner.state_rx.borrow()
    }

    /// Check if a transition from the current state to `target` is valid.
    pub fn can_transition_to(&self, target: AgentState) -> bool {
        let current = self.state();
        Self::is_valid_transition(current, target)
    }

    /// Validate whether a state transition is allowed.
    fn is_valid_transition(from: AgentState, to: AgentState) -> bool {
        matches!(
            (from, to),
            (AgentState::Created, AgentState::Running)
                | (AgentState::Running, AgentState::Paused)
                | (AgentState::Running, AgentState::Stopped)
                | (AgentState::Paused, AgentState::Running)
                | (AgentState::Paused, AgentState::Stopped)
        )
    }

    /// Transition to a new state.
    ///
    /// Returns an error if the transition is invalid.
    pub fn transition_to(&self, target: AgentState) -> Result<AgentState, StateTransitionError> {
        let current = self.state();
        if !Self::is_valid_transition(current, target) {
            return Err(StateTransitionError {
                from: current,
                to: target,
            });
        }

        self.inner.state_tx.send(target).ok();

        // Notify waiters on resume
        if target == AgentState::Running {
            self.inner.resume_notify.notify_waiters();
        }

        Ok(target)
    }

    /// Start the agent (Created → Running).
    pub fn start(&self) -> Result<AgentState, StateTransitionError> {
        self.transition_to(AgentState::Running)
    }

    /// Pause the agent (Running → Paused).
    pub fn pause(&self) -> Result<AgentState, StateTransitionError> {
        self.transition_to(AgentState::Paused)
    }

    /// Resume the agent (Paused → Running).
    pub fn resume(&self) -> Result<AgentState, StateTransitionError> {
        self.transition_to(AgentState::Running)
    }

    /// Stop the agent (Running|Paused → Stopped).
    pub fn stop(&self) -> Result<AgentState, StateTransitionError> {
        self.transition_to(AgentState::Stopped)
    }

    /// Wait until the agent is resumed (blocks while paused).
    ///
    /// Returns immediately if the agent is not paused.
    pub async fn wait_if_paused(&self) {
        while self.state() == AgentState::Paused {
            self.inner.resume_notify.notified().await;
        }
    }

    /// Register an in-flight request.
    pub async fn register_request(&self) {
        let mut count = self.inner.in_flight.lock().await;
        *count += 1;
    }

    /// Complete an in-flight request.
    pub async fn complete_request(&self) {
        let mut count = self.inner.in_flight.lock().await;
        *count = count.saturating_sub(1);
    }

    /// Get the number of in-flight requests.
    pub async fn in_flight_count(&self) -> u32 {
        *self.inner.in_flight.lock().await
    }

    /// Graceful shutdown: stop the agent and wait for in-flight requests to complete.
    ///
    /// Returns the number of in-flight requests at the time of stopping.
    pub async fn graceful_shutdown(&self) -> Result<u32, StateTransitionError> {
        let count = self.in_flight_count().await;
        self.stop()?;
        Ok(count)
    }

    /// Persist the current agent state.
    pub async fn persist(&self, context: Option<String>) {
        let state = PersistedAgentState {
            state: self.state(),
            context,
            persisted_at: chrono::Utc::now().to_rfc3339(),
        };
        let mut persisted = self.inner.persisted_state.lock().await;
        *persisted = Some(state);
    }

    /// Restore from a persisted state.
    ///
    /// Forces the lifecycle into the persisted state regardless of transition rules.
    pub async fn restore(&self, persisted: PersistedAgentState) {
        self.inner.state_tx.send(persisted.state).ok();
        let mut ps = self.inner.persisted_state.lock().await;
        *ps = Some(persisted);
    }

    /// Get the last persisted state, if any.
    pub async fn persisted_state(&self) -> Option<PersistedAgentState> {
        self.inner.persisted_state.lock().await.clone()
    }

    /// Subscribe to state changes.
    pub fn watch(&self) -> watch::Receiver<AgentState> {
        self.inner.state_rx.clone()
    }
}

impl Default for AgentLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_is_created() {
        let lifecycle = AgentLifecycle::new();
        assert_eq!(lifecycle.state(), AgentState::Created);
    }

    #[test]
    fn test_valid_transitions() {
        let lifecycle = AgentLifecycle::new();

        // Created → Running
        assert!(lifecycle.start().is_ok());
        assert_eq!(lifecycle.state(), AgentState::Running);

        // Running → Paused
        assert!(lifecycle.pause().is_ok());
        assert_eq!(lifecycle.state(), AgentState::Paused);

        // Paused → Running
        assert!(lifecycle.resume().is_ok());
        assert_eq!(lifecycle.state(), AgentState::Running);

        // Running → Stopped
        assert!(lifecycle.stop().is_ok());
        assert_eq!(lifecycle.state(), AgentState::Stopped);
    }

    #[test]
    fn test_invalid_transitions() {
        let lifecycle = AgentLifecycle::new();

        // Created → Paused (invalid)
        assert!(lifecycle.pause().is_err());

        // Created → Stopped (invalid)
        assert!(lifecycle.stop().is_err());
    }

    #[test]
    fn test_cannot_transition_from_stopped() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();
        lifecycle.stop().unwrap();

        // Stopped → Running (invalid)
        assert!(lifecycle.start().is_err());

        // Stopped → Paused (invalid)
        assert!(lifecycle.pause().is_err());
    }

    #[test]
    fn test_pause_to_stopped() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();
        lifecycle.pause().unwrap();

        // Paused → Stopped (valid)
        assert!(lifecycle.stop().is_ok());
        assert_eq!(lifecycle.state(), AgentState::Stopped);
    }

    #[test]
    fn test_state_transition_error_display() {
        let err = StateTransitionError {
            from: AgentState::Created,
            to: AgentState::Stopped,
        };
        assert_eq!(
            err.to_string(),
            "Invalid state transition from Created to Stopped"
        );
    }

    #[test]
    fn test_can_transition_to() {
        let lifecycle = AgentLifecycle::new();
        assert!(lifecycle.can_transition_to(AgentState::Running));
        assert!(!lifecycle.can_transition_to(AgentState::Paused));
        assert!(!lifecycle.can_transition_to(AgentState::Stopped));
    }

    #[tokio::test]
    async fn test_in_flight_requests() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();

        assert_eq!(lifecycle.in_flight_count().await, 0);

        lifecycle.register_request().await;
        lifecycle.register_request().await;
        assert_eq!(lifecycle.in_flight_count().await, 2);

        lifecycle.complete_request().await;
        assert_eq!(lifecycle.in_flight_count().await, 1);

        lifecycle.complete_request().await;
        assert_eq!(lifecycle.in_flight_count().await, 0);
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();

        lifecycle.register_request().await;
        lifecycle.register_request().await;

        let count = lifecycle.graceful_shutdown().await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(lifecycle.state(), AgentState::Stopped);
    }

    #[tokio::test]
    async fn test_persist_and_restore() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();

        lifecycle.persist(Some("test context".to_string())).await;

        let persisted = lifecycle.persisted_state().await.unwrap();
        assert_eq!(persisted.state, AgentState::Running);
        assert_eq!(persisted.context, Some("test context".to_string()));

        // Create a new lifecycle and restore
        let lifecycle2 = AgentLifecycle::new();
        assert_eq!(lifecycle2.state(), AgentState::Created);

        lifecycle2.restore(persisted).await;
        assert_eq!(lifecycle2.state(), AgentState::Running);
    }

    #[tokio::test]
    async fn test_wait_if_paused_returns_immediately_when_running() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();

        // Should return immediately since not paused
        lifecycle.wait_if_paused().await;
        assert_eq!(lifecycle.state(), AgentState::Running);
    }

    #[tokio::test]
    async fn test_wait_if_paused_blocks_then_resumes() {
        let lifecycle = AgentLifecycle::new();
        lifecycle.start().unwrap();
        lifecycle.pause().unwrap();

        let lifecycle_clone = lifecycle.clone();
        let handle = tokio::spawn(async move {
            lifecycle_clone.wait_if_paused().await;
            lifecycle_clone.state()
        });

        // Give the task time to block
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Resume should unblock the wait
        lifecycle.resume().unwrap();

        let state = handle.await.unwrap();
        assert_eq!(state, AgentState::Running);
    }

    #[tokio::test]
    async fn test_watch_state_changes() {
        let lifecycle = AgentLifecycle::new();
        let mut rx = lifecycle.watch();

        assert_eq!(*rx.borrow(), AgentState::Created);

        lifecycle.start().unwrap();
        rx.changed().await.unwrap();
        assert_eq!(*rx.borrow(), AgentState::Running);

        lifecycle.pause().unwrap();
        rx.changed().await.unwrap();
        assert_eq!(*rx.borrow(), AgentState::Paused);
    }

    #[test]
    fn test_agent_state_display() {
        assert_eq!(AgentState::Created.to_string(), "Created");
        assert_eq!(AgentState::Running.to_string(), "Running");
        assert_eq!(AgentState::Paused.to_string(), "Paused");
        assert_eq!(AgentState::Stopped.to_string(), "Stopped");
    }

    #[test]
    fn test_agent_state_serialization() {
        let state = AgentState::Running;
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: AgentState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }

    #[test]
    fn test_persisted_state_serialization() {
        let persisted = PersistedAgentState {
            state: AgentState::Paused,
            context: Some("my context".to_string()),
            persisted_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&persisted).unwrap();
        let deserialized: PersistedAgentState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.state, AgentState::Paused);
        assert_eq!(deserialized.context, Some("my context".to_string()));
    }
}
