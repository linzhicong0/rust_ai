// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Asynchronous task execution with callbacks and event streams.
//!
//! This module supports spawning async tasks with join handles,
//! success/failure callbacks, and event-driven progress reporting.

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};
use tokio::task::JoinHandle;

/// Status of an async task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending execution.
    Pending,
    /// Task is currently running.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed with an error.
    Failed(String),
    /// Task was cancelled.
    Cancelled,
}

/// An event emitted by an async task during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvent {
    /// The task ID that emitted this event.
    pub task_id: String,
    /// Event type/name.
    pub event_type: String,
    /// Optional event data.
    pub data: Option<serde_json::Value>,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
}

impl TaskEvent {
    /// Create a new task event.
    pub fn new(
        task_id: impl Into<String>,
        event_type: impl Into<String>,
        data: Option<serde_json::Value>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            event_type: event_type.into(),
            data,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Type for success callbacks.
pub type OnSuccessFn = Box<dyn FnOnce(serde_json::Value) + Send + 'static>;

/// Type for failure callbacks.
pub type OnFailureFn = Box<dyn FnOnce(String) + Send + 'static>;

/// A handle to a spawned async task.
pub struct TaskHandle {
    /// The unique task ID.
    pub task_id: String,

    /// Join handle for awaiting the task result.
    join_handle: JoinHandle<Result<serde_json::Value, String>>,

    /// Watch receiver for status updates.
    status_rx: watch::Receiver<TaskStatus>,
}

impl TaskHandle {
    /// Get the current task status.
    pub fn status(&self) -> TaskStatus {
        self.status_rx.borrow().clone()
    }

    /// Wait for the task to complete and return its result.
    pub async fn await_result(self) -> Result<serde_json::Value, String> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(e) => Err(format!("Task panicked: {}", e)),
        }
    }

    /// Cancel the task.
    pub fn cancel(self) {
        self.join_handle.abort();
    }

    /// Check if the task is finished.
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    /// Wait for status changes.
    pub async fn wait_for_status_change(&mut self) -> TaskStatus {
        let _ = self.status_rx.changed().await;
        self.status_rx.borrow().clone()
    }
}

/// Builder for async tasks with callbacks and event streams.
pub struct AsyncTaskBuilder {
    task_id: String,
    on_success: Option<OnSuccessFn>,
    on_failure: Option<OnFailureFn>,
    event_buffer_size: usize,
}

impl AsyncTaskBuilder {
    /// Create a new task builder with an ID.
    pub fn new(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            on_success: None,
            on_failure: None,
            event_buffer_size: 100,
        }
    }

    /// Set a callback to invoke on successful completion.
    pub fn on_success(mut self, callback: impl FnOnce(serde_json::Value) + Send + 'static) -> Self {
        self.on_success = Some(Box::new(callback));
        self
    }

    /// Set a callback to invoke on failure.
    pub fn on_failure(mut self, callback: impl FnOnce(String) + Send + 'static) -> Self {
        self.on_failure = Some(Box::new(callback));
        self
    }

    /// Set the event buffer size.
    pub fn event_buffer_size(mut self, size: usize) -> Self {
        self.event_buffer_size = size;
        self
    }

    /// Spawn the async task.
    ///
    /// The `task_fn` receives an event sender that can be used to emit progress events.
    /// Returns a `TaskHandle` for monitoring and an event receiver for the event stream.
    pub fn spawn<F, Fut>(self, task_fn: F) -> (TaskHandle, mpsc::Receiver<TaskEvent>)
    where
        F: FnOnce(mpsc::Sender<TaskEvent>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<serde_json::Value, String>> + Send + 'static,
    {
        let (status_tx, status_rx) = watch::channel(TaskStatus::Pending);
        let (event_tx, event_rx) = mpsc::channel(self.event_buffer_size);
        let task_id = self.task_id.clone();
        let on_success = self.on_success;
        let on_failure = self.on_failure;

        let task_id_clone = task_id.clone();
        let event_tx_clone = event_tx.clone();

        let join_handle = tokio::spawn(async move {
            // Update status to Running
            let _ = status_tx.send(TaskStatus::Running);
            let _ = event_tx_clone
                .send(TaskEvent::new(&task_id_clone, "started", None))
                .await;

            // Execute the task
            let result = task_fn(event_tx_clone.clone()).await;

            match &result {
                Ok(value) => {
                    let _ = status_tx.send(TaskStatus::Completed);
                    let _ = event_tx_clone
                        .send(TaskEvent::new(
                            &task_id_clone,
                            "completed",
                            Some(value.clone()),
                        ))
                        .await;
                    if let Some(cb) = on_success {
                        cb(value.clone());
                    }
                }
                Err(err) => {
                    let _ = status_tx.send(TaskStatus::Failed(err.clone()));
                    let _ = event_tx_clone
                        .send(TaskEvent::new(
                            &task_id_clone,
                            "failed",
                            Some(serde_json::Value::String(err.clone())),
                        ))
                        .await;
                    if let Some(cb) = on_failure {
                        cb(err.clone());
                    }
                }
            }

            result
        });

        let handle = TaskHandle {
            task_id,
            join_handle,
            status_rx,
        };

        (handle, event_rx)
    }
}

/// Convenience function to spawn a simple async task.
pub fn spawn_task<F, Fut>(
    task_id: impl Into<String>,
    task_fn: F,
) -> (TaskHandle, mpsc::Receiver<TaskEvent>)
where
    F: FnOnce(mpsc::Sender<TaskEvent>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<serde_json::Value, String>> + Send + 'static,
{
    AsyncTaskBuilder::new(task_id).spawn(task_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_spawn_successful_task() {
        let (handle, mut events) = spawn_task("task1", |_event_tx| async {
            Ok(json!({"result": "success"}))
        });

        let result = handle.await_result().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"result": "success"}));

        // Should have started and completed events
        let event1 = events.recv().await.unwrap();
        assert_eq!(event1.event_type, "started");

        let event2 = events.recv().await.unwrap();
        assert_eq!(event2.event_type, "completed");
    }

    #[tokio::test]
    async fn test_spawn_failing_task() {
        let (handle, mut events) = spawn_task("task2", |_event_tx| async {
            Err::<serde_json::Value, String>("something went wrong".to_string())
        });

        let result = handle.await_result().await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "something went wrong");

        let event1 = events.recv().await.unwrap();
        assert_eq!(event1.event_type, "started");

        let event2 = events.recv().await.unwrap();
        assert_eq!(event2.event_type, "failed");
    }

    #[tokio::test]
    async fn test_task_status_transitions() {
        let (status_tx, status_rx) = watch::channel(TaskStatus::Pending);

        // Simulate status transitions
        assert_eq!(*status_rx.borrow(), TaskStatus::Pending);

        status_tx.send(TaskStatus::Running).unwrap();
        assert_eq!(*status_rx.borrow(), TaskStatus::Running);

        status_tx.send(TaskStatus::Completed).unwrap();
        assert_eq!(*status_rx.borrow(), TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_with_progress_events() {
        let (handle, mut events) = spawn_task("task3", |event_tx| async move {
            event_tx
                .send(TaskEvent::new(
                    "task3",
                    "progress",
                    Some(json!({"step": 1})),
                ))
                .await
                .unwrap();
            event_tx
                .send(TaskEvent::new(
                    "task3",
                    "progress",
                    Some(json!({"step": 2})),
                ))
                .await
                .unwrap();
            Ok(json!({"steps_completed": 2}))
        });

        let result = handle.await_result().await;
        assert!(result.is_ok());

        // Collect events
        let mut event_types = Vec::new();
        while let Ok(event) = events.try_recv() {
            event_types.push(event.event_type);
        }

        assert!(event_types.contains(&"started".to_string()));
        assert!(event_types.contains(&"progress".to_string()));
        assert!(event_types.contains(&"completed".to_string()));
    }

    #[tokio::test]
    async fn test_on_success_callback() {
        let success_value = Arc::new(Mutex::new(None));
        let success_value_clone = success_value.clone();

        let (handle, _events) = AsyncTaskBuilder::new("task4")
            .on_success(move |val| {
                let sv = success_value_clone.clone();
                // Note: callback is sync, so we just store directly
                // In practice, we need a different approach for async callbacks
                let _ = sv.try_lock().map(|mut guard| *guard = Some(val));
            })
            .spawn(|_| async { Ok(json!(42)) });

        let result = handle.await_result().await;
        assert!(result.is_ok());

        // Give callback time to execute
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let val = success_value.lock().await;
        assert_eq!(*val, Some(json!(42)));
    }

    #[tokio::test]
    async fn test_on_failure_callback() {
        let error_msg = Arc::new(Mutex::new(None));
        let error_msg_clone = error_msg.clone();

        let (handle, _events) = AsyncTaskBuilder::new("task5")
            .on_failure(move |err| {
                let em = error_msg_clone.clone();
                let _ = em.try_lock().map(|mut guard| *guard = Some(err));
            })
            .spawn(|_| async { Err::<serde_json::Value, _>("oops".to_string()) });

        let result = handle.await_result().await;
        assert!(result.is_err());

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let val = error_msg.lock().await;
        assert_eq!(*val, Some("oops".to_string()));
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let (handle, _events) = spawn_task("task6", |_event_tx| async {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            Ok(json!("should not reach"))
        });

        // Cancel before it completes
        handle.cancel();

        // Give tokio time to abort
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn test_task_event_creation() {
        let event = TaskEvent::new("t1", "progress", Some(json!({"pct": 50})));
        assert_eq!(event.task_id, "t1");
        assert_eq!(event.event_type, "progress");
        assert_eq!(event.data, Some(json!({"pct": 50})));
        assert!(!event.timestamp.is_empty());
    }

    #[tokio::test]
    async fn test_task_status_serialization() {
        let status = TaskStatus::Failed("error msg".to_string());
        let json_str = serde_json::to_string(&status).unwrap();
        let deserialized: TaskStatus = serde_json::from_str(&json_str).unwrap();
        assert_eq!(status, deserialized);
    }

    #[tokio::test]
    async fn test_is_finished() {
        let (handle, _events) = spawn_task("task7", |_| async { Ok(json!("done")) });

        // Wait for completion
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(handle.is_finished());
    }

    #[tokio::test]
    async fn test_task_event_serialization() {
        let event = TaskEvent::new("t1", "completed", Some(json!({"value": true})));
        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: TaskEvent = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.task_id, "t1");
        assert_eq!(deserialized.event_type, "completed");
    }
}
