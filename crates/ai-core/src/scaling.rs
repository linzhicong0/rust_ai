// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Horizontal Scaling (REQ-14.2)
//!
//! Provides abstractions for horizontal scaling through stateless worker nodes
//! and distributed task queues.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Errors that can occur in scaling operations.
#[derive(Debug, thiserror::Error)]
pub enum ScalingError {
    #[error("Queue error: {0}")]
    Queue(String),
    #[error("Worker error: {0}")]
    Worker(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// A task to be distributed across workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    pub id: String,
    /// Task type/name for routing.
    pub task_type: String,
    /// Serialized payload.
    pub payload: serde_json::Value,
    /// Task priority (higher = more urgent).
    pub priority: u32,
    /// Maximum time allowed for execution.
    pub timeout: Duration,
    /// When the task was created.
    #[serde(skip)]
    pub created_at: Option<Instant>,
}

impl Task {
    /// Create a new task with the given type and payload.
    pub fn new(task_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            task_type: task_type.into(),
            payload,
            priority: 0,
            timeout: Duration::from_secs(300),
            created_at: Some(Instant::now()),
        }
    }

    /// Set task priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set task timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Result of a completed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task ID this result corresponds to.
    pub task_id: String,
    /// Whether the task succeeded.
    pub success: bool,
    /// The result payload (if successful).
    pub output: Option<serde_json::Value>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// How long the task took to execute.
    pub duration: Duration,
}

/// Status of a task in the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is queued and waiting for a worker.
    Pending,
    /// Task is currently being processed.
    InProgress,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
}

/// Trait for distributed task queue backends (Redis, RabbitMQ, etc.).
#[async_trait]
pub trait TaskQueue: Send + Sync {
    /// Enqueue a task for processing.
    async fn enqueue(&self, task: Task) -> Result<String, ScalingError>;

    /// Dequeue the next available task (blocks until one is available or timeout).
    async fn dequeue(&self, timeout: Duration) -> Result<Option<Task>, ScalingError>;

    /// Report the result of a completed task.
    async fn complete(&self, result: TaskResult) -> Result<(), ScalingError>;

    /// Get the status of a task.
    async fn status(&self, task_id: &str) -> Result<TaskStatus, ScalingError>;

    /// Get the current queue length.
    async fn queue_length(&self) -> Result<usize, ScalingError>;
}

/// Health status of a worker node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    /// Worker identifier.
    pub worker_id: String,
    /// Whether the worker is healthy.
    pub healthy: bool,
    /// Current load (0.0 to 1.0).
    pub load: f64,
    /// Number of tasks currently being processed.
    pub active_tasks: usize,
    /// Total tasks processed since start.
    pub total_processed: u64,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

/// Information about a registered worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique worker ID.
    pub id: String,
    /// Worker address (host:port).
    pub address: String,
    /// Task types this worker can handle.
    pub capabilities: Vec<String>,
    /// Maximum concurrent tasks.
    pub max_concurrency: usize,
    /// Current health status.
    pub health: WorkerHealth,
}

/// Trait for worker registration and health monitoring.
#[async_trait]
pub trait WorkerRegistry: Send + Sync {
    /// Register a new worker.
    async fn register(&self, info: WorkerInfo) -> Result<(), ScalingError>;

    /// Deregister a worker.
    async fn deregister(&self, worker_id: &str) -> Result<(), ScalingError>;

    /// Send a heartbeat/health update.
    async fn heartbeat(&self, health: WorkerHealth) -> Result<(), ScalingError>;

    /// Get all registered workers.
    async fn list_workers(&self) -> Result<Vec<WorkerInfo>, ScalingError>;

    /// Get healthy workers that can handle a given task type.
    async fn available_workers(&self, task_type: &str) -> Result<Vec<WorkerInfo>, ScalingError>;
}

/// In-memory task queue implementation for testing and single-node deployments.
pub struct InMemoryTaskQueue {
    tasks: std::sync::Arc<tokio::sync::Mutex<Vec<Task>>>,
    results: std::sync::Arc<tokio::sync::Mutex<HashMap<String, TaskResult>>>,
    statuses: std::sync::Arc<tokio::sync::Mutex<HashMap<String, TaskStatus>>>,
}

impl InMemoryTaskQueue {
    /// Create a new in-memory task queue.
    pub fn new() -> Self {
        Self {
            tasks: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            results: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            statuses: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TaskQueue for InMemoryTaskQueue {
    async fn enqueue(&self, task: Task) -> Result<String, ScalingError> {
        let id = task.id.clone();
        let mut statuses = self.statuses.lock().await;
        statuses.insert(id.clone(), TaskStatus::Pending);
        let mut tasks = self.tasks.lock().await;
        tasks.push(task);
        // Sort by priority (highest first)
        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(id)
    }

    async fn dequeue(&self, _timeout: Duration) -> Result<Option<Task>, ScalingError> {
        let mut tasks = self.tasks.lock().await;
        if let Some(task) = tasks.first().cloned() {
            tasks.remove(0);
            let mut statuses = self.statuses.lock().await;
            statuses.insert(task.id.clone(), TaskStatus::InProgress);
            Ok(Some(task))
        } else {
            Ok(None)
        }
    }

    async fn complete(&self, result: TaskResult) -> Result<(), ScalingError> {
        let task_id = result.task_id.clone();
        let status = if result.success {
            TaskStatus::Completed
        } else {
            TaskStatus::Failed
        };
        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.clone(), status);
        let mut results = self.results.lock().await;
        results.insert(task_id, result);
        Ok(())
    }

    async fn status(&self, task_id: &str) -> Result<TaskStatus, ScalingError> {
        let statuses = self.statuses.lock().await;
        statuses
            .get(task_id)
            .copied()
            .ok_or_else(|| ScalingError::Queue(format!("Task not found: {task_id}")))
    }

    async fn queue_length(&self) -> Result<usize, ScalingError> {
        let tasks = self.tasks.lock().await;
        Ok(tasks.len())
    }
}

/// In-memory worker registry for testing.
pub struct InMemoryWorkerRegistry {
    workers: std::sync::Arc<tokio::sync::Mutex<HashMap<String, WorkerInfo>>>,
}

impl InMemoryWorkerRegistry {
    /// Create a new in-memory worker registry.
    pub fn new() -> Self {
        Self {
            workers: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryWorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkerRegistry for InMemoryWorkerRegistry {
    async fn register(&self, info: WorkerInfo) -> Result<(), ScalingError> {
        let mut workers = self.workers.lock().await;
        workers.insert(info.id.clone(), info);
        Ok(())
    }

    async fn deregister(&self, worker_id: &str) -> Result<(), ScalingError> {
        let mut workers = self.workers.lock().await;
        workers.remove(worker_id);
        Ok(())
    }

    async fn heartbeat(&self, health: WorkerHealth) -> Result<(), ScalingError> {
        let mut workers = self.workers.lock().await;
        if let Some(worker) = workers.get_mut(&health.worker_id) {
            worker.health = health;
            Ok(())
        } else {
            Err(ScalingError::Worker(format!(
                "Worker not found: {}",
                health.worker_id
            )))
        }
    }

    async fn list_workers(&self) -> Result<Vec<WorkerInfo>, ScalingError> {
        let workers = self.workers.lock().await;
        Ok(workers.values().cloned().collect())
    }

    async fn available_workers(&self, task_type: &str) -> Result<Vec<WorkerInfo>, ScalingError> {
        let workers = self.workers.lock().await;
        Ok(workers
            .values()
            .filter(|w| w.health.healthy && w.capabilities.contains(&task_type.to_string()))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_creation() {
        let task = Task::new("inference", serde_json::json!({"prompt": "hello"}));
        assert_eq!(task.task_type, "inference");
        assert_eq!(task.priority, 0);
        assert_eq!(task.timeout, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn test_task_with_priority() {
        let task = Task::new("inference", serde_json::json!({})).with_priority(10);
        assert_eq!(task.priority, 10);
    }

    #[tokio::test]
    async fn test_in_memory_task_queue_enqueue_dequeue() {
        let queue = InMemoryTaskQueue::new();

        let task = Task::new("inference", serde_json::json!({"prompt": "test"}));
        let task_id = queue.enqueue(task).await.unwrap();

        assert_eq!(queue.queue_length().await.unwrap(), 1);
        assert_eq!(queue.status(&task_id).await.unwrap(), TaskStatus::Pending);

        let dequeued = queue.dequeue(Duration::from_secs(1)).await.unwrap();
        assert!(dequeued.is_some());
        let dequeued = dequeued.unwrap();
        assert_eq!(dequeued.id, task_id);
        assert_eq!(
            queue.status(&task_id).await.unwrap(),
            TaskStatus::InProgress
        );
        assert_eq!(queue.queue_length().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_in_memory_task_queue_priority_ordering() {
        let queue = InMemoryTaskQueue::new();

        let low = Task::new("task", serde_json::json!({"name": "low"})).with_priority(1);
        let high = Task::new("task", serde_json::json!({"name": "high"})).with_priority(10);
        let medium = Task::new("task", serde_json::json!({"name": "medium"})).with_priority(5);

        queue.enqueue(low).await.unwrap();
        queue.enqueue(high).await.unwrap();
        queue.enqueue(medium).await.unwrap();

        let first = queue
            .dequeue(Duration::from_secs(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.priority, 10);

        let second = queue
            .dequeue(Duration::from_secs(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.priority, 5);

        let third = queue
            .dequeue(Duration::from_secs(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(third.priority, 1);
    }

    #[tokio::test]
    async fn test_in_memory_task_queue_complete() {
        let queue = InMemoryTaskQueue::new();

        let task = Task::new("inference", serde_json::json!({}));
        let task_id = queue.enqueue(task).await.unwrap();
        let _ = queue.dequeue(Duration::from_secs(1)).await.unwrap();

        let result = TaskResult {
            task_id: task_id.clone(),
            success: true,
            output: Some(serde_json::json!({"result": "done"})),
            error: None,
            duration: Duration::from_millis(100),
        };
        queue.complete(result).await.unwrap();

        assert_eq!(queue.status(&task_id).await.unwrap(), TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_in_memory_task_queue_empty_dequeue() {
        let queue = InMemoryTaskQueue::new();
        let result = queue.dequeue(Duration::from_secs(1)).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_worker_registry_register_and_list() {
        let registry = InMemoryWorkerRegistry::new();

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            address: "localhost:8001".to_string(),
            capabilities: vec!["inference".to_string(), "embedding".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-1".to_string(),
                healthy: true,
                load: 0.2,
                active_tasks: 1,
                total_processed: 100,
                metadata: HashMap::new(),
            },
        };

        registry.register(worker).await.unwrap();

        let workers = registry.list_workers().await.unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, "worker-1");
        assert_eq!(workers[0].capabilities.len(), 2);
    }

    #[tokio::test]
    async fn test_worker_registry_deregister() {
        let registry = InMemoryWorkerRegistry::new();

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            address: "localhost:8001".to_string(),
            capabilities: vec!["inference".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-1".to_string(),
                healthy: true,
                load: 0.0,
                active_tasks: 0,
                total_processed: 0,
                metadata: HashMap::new(),
            },
        };

        registry.register(worker).await.unwrap();
        assert_eq!(registry.list_workers().await.unwrap().len(), 1);

        registry.deregister("worker-1").await.unwrap();
        assert_eq!(registry.list_workers().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_worker_registry_heartbeat() {
        let registry = InMemoryWorkerRegistry::new();

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            address: "localhost:8001".to_string(),
            capabilities: vec!["inference".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-1".to_string(),
                healthy: true,
                load: 0.2,
                active_tasks: 1,
                total_processed: 50,
                metadata: HashMap::new(),
            },
        };

        registry.register(worker).await.unwrap();

        let updated_health = WorkerHealth {
            worker_id: "worker-1".to_string(),
            healthy: true,
            load: 0.8,
            active_tasks: 3,
            total_processed: 55,
            metadata: HashMap::new(),
        };

        registry.heartbeat(updated_health).await.unwrap();

        let workers = registry.list_workers().await.unwrap();
        assert_eq!(workers[0].health.load, 0.8);
        assert_eq!(workers[0].health.active_tasks, 3);
    }

    #[tokio::test]
    async fn test_worker_registry_available_workers() {
        let registry = InMemoryWorkerRegistry::new();

        let worker1 = WorkerInfo {
            id: "worker-1".to_string(),
            address: "localhost:8001".to_string(),
            capabilities: vec!["inference".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-1".to_string(),
                healthy: true,
                load: 0.2,
                active_tasks: 1,
                total_processed: 0,
                metadata: HashMap::new(),
            },
        };

        let worker2 = WorkerInfo {
            id: "worker-2".to_string(),
            address: "localhost:8002".to_string(),
            capabilities: vec!["embedding".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-2".to_string(),
                healthy: true,
                load: 0.5,
                active_tasks: 2,
                total_processed: 0,
                metadata: HashMap::new(),
            },
        };

        let worker3 = WorkerInfo {
            id: "worker-3".to_string(),
            address: "localhost:8003".to_string(),
            capabilities: vec!["inference".to_string()],
            max_concurrency: 4,
            health: WorkerHealth {
                worker_id: "worker-3".to_string(),
                healthy: false, // unhealthy
                load: 0.9,
                active_tasks: 4,
                total_processed: 0,
                metadata: HashMap::new(),
            },
        };

        registry.register(worker1).await.unwrap();
        registry.register(worker2).await.unwrap();
        registry.register(worker3).await.unwrap();

        let available = registry.available_workers("inference").await.unwrap();
        assert_eq!(available.len(), 1); // only worker-1 is healthy + supports inference
        assert_eq!(available[0].id, "worker-1");

        let available = registry.available_workers("embedding").await.unwrap();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, "worker-2");
    }
}
