// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Resource Management (REQ-14.3)
//!
//! Manages computational resources (GPU memory, API quotas, database connections)
//! with pooling, checkout/checkin, quota tracking, and automatic cleanup.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::resource_management::{ResourcePool, ResourceConfig, QuotaTracker};
//!
//! let mut pool = ResourcePool::new(ResourceConfig {
//!     max_resources: 5,
//!     idle_timeout_secs: 300,
//!     resource_type: "api_connection".into(),
//! });
//!
//! // Checkout a resource
//! let handle = pool.checkout("worker-1").unwrap();
//! assert_eq!(pool.active_count(), 1);
//!
//! // Return the resource
//! pool.checkin(handle);
//! assert_eq!(pool.active_count(), 0);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── ResourceConfig ────────────────────────────────────────────────────────────

/// Configuration for a resource pool.
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// Maximum number of resources in the pool.
    pub max_resources: usize,
    /// Idle timeout in seconds before resources are cleaned up.
    pub idle_timeout_secs: u64,
    /// Type of resource managed by this pool.
    pub resource_type: String,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_resources: 10,
            idle_timeout_secs: 300,
            resource_type: "generic".into(),
        }
    }
}

// ── ResourceHandle ────────────────────────────────────────────────────────────

/// A handle to a checked-out resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceHandle {
    /// Unique resource identifier.
    pub id: String,
    /// Who checked out this resource.
    pub owner: String,
    /// When it was checked out.
    pub checked_out_at: Instant,
}

// ── ResourceError ─────────────────────────────────────────────────────────────

/// Errors that can occur during resource management.
#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    /// No resources available in the pool.
    #[error("Pool exhausted: no resources available (max: {max})")]
    PoolExhausted { max: usize },
    /// Quota exceeded for the provider.
    #[error("Quota exceeded for '{provider}': used {used}/{limit}")]
    QuotaExceeded {
        provider: String,
        used: u64,
        limit: u64,
    },
    /// Resource not found.
    #[error("Resource not found: {0}")]
    NotFound(String),
    /// Resource already returned.
    #[error("Resource already returned: {0}")]
    AlreadyReturned(String),
}

// ── ResourcePool ──────────────────────────────────────────────────────────────

/// A pool of resources with checkout/checkin semantics.
#[derive(Debug)]
pub struct ResourcePool {
    config: ResourceConfig,
    /// Currently checked-out resources.
    active: HashMap<String, ResourceHandle>,
    /// Available resource IDs (idle pool).
    available: Vec<IdleResource>,
    /// Counter for generating unique resource IDs.
    next_id: u64,
    /// Total checkouts (lifetime).
    total_checkouts: u64,
}

/// An idle resource waiting to be checked out.
#[derive(Debug)]
struct IdleResource {
    id: String,
    idle_since: Instant,
}

impl ResourcePool {
    /// Create a new resource pool with the given configuration.
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            active: HashMap::new(),
            available: Vec::new(),
            next_id: 0,
            total_checkouts: 0,
        }
    }

    /// Checkout a resource from the pool.
    pub fn checkout(&mut self, owner: impl Into<String>) -> Result<ResourceHandle, ResourceError> {
        // First try to recycle an idle resource
        if let Some(idle) = self.available.pop() {
            let handle = ResourceHandle {
                id: idle.id,
                owner: owner.into(),
                checked_out_at: Instant::now(),
            };
            self.active.insert(handle.id.clone(), handle.clone());
            self.total_checkouts += 1;
            return Ok(handle);
        }

        // Check if we can create a new resource
        if self.active.len() >= self.config.max_resources {
            return Err(ResourceError::PoolExhausted {
                max: self.config.max_resources,
            });
        }

        // Create a new resource
        self.next_id += 1;
        let id = format!("{}-{}", self.config.resource_type, self.next_id);
        let handle = ResourceHandle {
            id: id.clone(),
            owner: owner.into(),
            checked_out_at: Instant::now(),
        };
        self.active.insert(id, handle.clone());
        self.total_checkouts += 1;
        Ok(handle)
    }

    /// Return a resource to the pool.
    pub fn checkin(&mut self, handle: ResourceHandle) {
        if self.active.remove(&handle.id).is_some() {
            self.available.push(IdleResource {
                id: handle.id,
                idle_since: Instant::now(),
            });
        }
    }

    /// Clean up idle resources that have exceeded the idle timeout.
    pub fn cleanup_idle(&mut self) -> usize {
        let timeout = Duration::from_secs(self.config.idle_timeout_secs);
        let before = self.available.len();
        self.available.retain(|r| r.idle_since.elapsed() < timeout);
        before - self.available.len()
    }

    /// Get the number of currently active (checked-out) resources.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get the number of idle (available) resources.
    pub fn idle_count(&self) -> usize {
        self.available.len()
    }

    /// Get total number of resources (active + idle).
    pub fn total_count(&self) -> usize {
        self.active.len() + self.available.len()
    }

    /// Get total checkout count (lifetime).
    pub fn total_checkouts(&self) -> u64 {
        self.total_checkouts
    }

    /// Get the pool configuration.
    pub fn config(&self) -> &ResourceConfig {
        &self.config
    }
}

// ── QuotaConfig ───────────────────────────────────────────────────────────────

/// Quota configuration for a provider.
#[derive(Debug, Clone)]
pub struct QuotaConfig {
    /// Maximum requests per period.
    pub max_requests: u64,
    /// Period duration.
    pub period: Duration,
    /// Provider name.
    pub provider: String,
}

// ── QuotaTracker ──────────────────────────────────────────────────────────────

/// Tracks API quota usage per provider.
#[derive(Debug)]
pub struct QuotaTracker {
    quotas: HashMap<String, QuotaState>,
}

#[derive(Debug)]
struct QuotaState {
    config: QuotaConfig,
    used: u64,
    period_start: Instant,
}

impl QuotaTracker {
    /// Create a new quota tracker.
    pub fn new() -> Self {
        Self {
            quotas: HashMap::new(),
        }
    }

    /// Register a quota for a provider.
    pub fn register(&mut self, config: QuotaConfig) {
        self.quotas.insert(
            config.provider.clone(),
            QuotaState {
                config,
                used: 0,
                period_start: Instant::now(),
            },
        );
    }

    /// Record a usage against a provider's quota.
    pub fn record_usage(&mut self, provider: &str, count: u64) -> Result<(), ResourceError> {
        let state = self
            .quotas
            .get_mut(provider)
            .ok_or_else(|| ResourceError::NotFound(provider.to_string()))?;

        // Reset if period has elapsed
        if state.period_start.elapsed() >= state.config.period {
            state.used = 0;
            state.period_start = Instant::now();
        }

        // Check if usage would exceed quota
        if state.used + count > state.config.max_requests {
            return Err(ResourceError::QuotaExceeded {
                provider: provider.to_string(),
                used: state.used,
                limit: state.config.max_requests,
            });
        }

        state.used += count;
        Ok(())
    }

    /// Get current usage for a provider.
    pub fn get_usage(&self, provider: &str) -> Option<(u64, u64)> {
        self.quotas
            .get(provider)
            .map(|s| (s.used, s.config.max_requests))
    }

    /// Get remaining quota for a provider.
    pub fn remaining(&self, provider: &str) -> Option<u64> {
        self.quotas
            .get(provider)
            .map(|s| s.config.max_requests.saturating_sub(s.used))
    }

    /// Check if a provider has quota available.
    pub fn has_quota(&self, provider: &str) -> bool {
        self.quotas
            .get(provider)
            .map(|s| s.used < s.config.max_requests)
            .unwrap_or(false)
    }
}

impl Default for QuotaTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-14.3: Resource pool with checkout/checkin
    #[test]
    fn test_checkout_and_checkin() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 3,
            idle_timeout_secs: 300,
            resource_type: "connection".into(),
        });

        let handle = pool.checkout("worker-1").unwrap();
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.idle_count(), 0);

        pool.checkin(handle);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 1);
    }

    // REQ-14.3: Pool exhaustion
    #[test]
    fn test_pool_exhaustion() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 2,
            idle_timeout_secs: 300,
            resource_type: "gpu".into(),
        });

        let _h1 = pool.checkout("w1").unwrap();
        let _h2 = pool.checkout("w2").unwrap();
        let result = pool.checkout("w3");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResourceError::PoolExhausted { max: 2 }
        ));
    }

    // REQ-14.3: Resource recycling
    #[test]
    fn test_resource_recycling() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 2,
            idle_timeout_secs: 300,
            resource_type: "conn".into(),
        });

        let h1 = pool.checkout("w1").unwrap();
        let id = h1.id.clone();
        pool.checkin(h1);

        // Checking out again should reuse the idle resource
        let h2 = pool.checkout("w2").unwrap();
        assert_eq!(h2.id, id); // Same resource recycled
    }

    // REQ-14.3: Quota tracking per provider
    #[test]
    fn test_quota_tracking() {
        let mut tracker = QuotaTracker::new();
        tracker.register(QuotaConfig {
            max_requests: 100,
            period: Duration::from_secs(3600),
            provider: "openai".into(),
        });

        // Record usage
        assert!(tracker.record_usage("openai", 50).is_ok());
        assert_eq!(tracker.get_usage("openai"), Some((50, 100)));
        assert_eq!(tracker.remaining("openai"), Some(50));
    }

    // REQ-14.3: Quota exceeded
    #[test]
    fn test_quota_exceeded() {
        let mut tracker = QuotaTracker::new();
        tracker.register(QuotaConfig {
            max_requests: 10,
            period: Duration::from_secs(3600),
            provider: "anthropic".into(),
        });

        assert!(tracker.record_usage("anthropic", 8).is_ok());
        let result = tracker.record_usage("anthropic", 5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResourceError::QuotaExceeded { .. }
        ));
    }

    // REQ-14.3: Multiple checkouts
    #[test]
    fn test_multiple_checkouts() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 5,
            idle_timeout_secs: 300,
            resource_type: "api".into(),
        });

        let h1 = pool.checkout("w1").unwrap();
        let h2 = pool.checkout("w2").unwrap();
        let h3 = pool.checkout("w3").unwrap();
        assert_eq!(pool.active_count(), 3);
        assert_eq!(pool.total_checkouts(), 3);

        pool.checkin(h2);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(pool.idle_count(), 1);

        pool.checkin(h1);
        pool.checkin(h3);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 3);
    }

    // REQ-14.3: Automatic cleanup of idle resources
    #[test]
    fn test_cleanup_idle() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 5,
            idle_timeout_secs: 0, // Immediate timeout for testing
            resource_type: "conn".into(),
        });

        let h1 = pool.checkout("w1").unwrap();
        pool.checkin(h1);
        assert_eq!(pool.idle_count(), 1);

        // Sleep briefly to ensure idle timeout
        std::thread::sleep(Duration::from_millis(10));

        let cleaned = pool.cleanup_idle();
        assert_eq!(cleaned, 1);
        assert_eq!(pool.idle_count(), 0);
    }

    // REQ-14.3: Quota has_quota check
    #[test]
    fn test_has_quota() {
        let mut tracker = QuotaTracker::new();
        tracker.register(QuotaConfig {
            max_requests: 5,
            period: Duration::from_secs(3600),
            provider: "deepseek".into(),
        });

        assert!(tracker.has_quota("deepseek"));
        tracker.record_usage("deepseek", 5).unwrap();
        assert!(!tracker.has_quota("deepseek"));
    }

    // REQ-14.3: Unknown provider returns error
    #[test]
    fn test_unknown_provider() {
        let mut tracker = QuotaTracker::new();
        let result = tracker.record_usage("unknown", 1);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ResourceError::NotFound(_)));
    }

    // REQ-14.3: Resource handle contains owner info
    #[test]
    fn test_resource_handle_owner() {
        let mut pool = ResourcePool::new(ResourceConfig::default());
        let handle = pool.checkout("my-worker").unwrap();
        assert_eq!(handle.owner, "my-worker");
    }

    // REQ-14.3: Total count includes active and idle
    #[test]
    fn test_total_count() {
        let mut pool = ResourcePool::new(ResourceConfig {
            max_resources: 10,
            idle_timeout_secs: 300,
            resource_type: "db".into(),
        });

        let h1 = pool.checkout("w1").unwrap();
        let _h2 = pool.checkout("w2").unwrap();
        pool.checkin(h1);

        assert_eq!(pool.total_count(), 2); // 1 active + 1 idle
    }
}
