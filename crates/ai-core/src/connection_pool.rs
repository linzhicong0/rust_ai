// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! HTTP connection pooling for LLM providers (REQ-12.5).
//!
//! This module manages per-provider `reqwest::Client` instances with configurable
//! pool sizing, keep-alive behavior, and lightweight pool statistics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, CONNECTION};
use thiserror::Error;
use tokio::sync::RwLock;

/// Configuration for a provider-specific HTTP connection pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionPoolConfig {
    /// Maximum idle connections to retain per host.
    pub max_idle_per_host: usize,
    /// Maximum idle time before pooled connections are dropped.
    pub idle_timeout_secs: u64,
    /// Timeout for establishing new TCP/TLS connections.
    pub connect_timeout_secs: u64,
    /// Per-request timeout for provider HTTP calls.
    pub request_timeout_secs: u64,
    /// Maximum retry attempts when reconnecting after transport failures.
    pub max_retries: u32,
    /// Whether to enable keep-alive and connection reuse.
    pub keep_alive: bool,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: 32,
            idle_timeout_secs: 90,
            connect_timeout_secs: 10,
            request_timeout_secs: 60,
            max_retries: 3,
            keep_alive: true,
        }
    }
}

/// Observable connection pool statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Approximate number of active connections.
    pub active_connections: usize,
    /// Approximate number of idle pooled connections.
    pub idle_connections: usize,
    /// Total requests sent through the pool.
    pub total_requests: u64,
    /// Total failed requests observed by callers.
    pub failed_requests: u64,
    /// Total reconnection events observed by callers.
    pub reconnections: u64,
}

/// Errors for connection pool creation and lookup.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ConnectionPoolError {
    /// Provider pool does not exist.
    #[error("provider pool not found: {0}")]
    ProviderNotFound(String),
    /// Failed to create or refresh a provider connection.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    /// Pool resources are unavailable.
    #[error("connection pool exhausted: {0}")]
    PoolExhausted(String),
    /// Invalid connection pool configuration.
    #[error("invalid connection pool configuration: {0}")]
    ConfigError(String),
}

/// Per-provider HTTP connection pool.
#[derive(Debug, Clone)]
pub struct ProviderConnectionPool {
    client: reqwest::Client,
    /// Pool configuration.
    pub config: ConnectionPoolConfig,
    stats: Arc<Mutex<PoolStats>>,
}

impl ProviderConnectionPool {
    /// Create a new provider connection pool backed by a configured reqwest client.
    pub fn new(config: ConnectionPoolConfig) -> Result<Self, ConnectionPoolError> {
        Self::validate_config(&config)?;

        let idle_timeout = Duration::from_secs(config.idle_timeout_secs);
        let connect_timeout = Duration::from_secs(config.connect_timeout_secs);
        let request_timeout = Duration::from_secs(config.request_timeout_secs);

        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(if config.keep_alive {
                config.max_idle_per_host
            } else {
                0
            })
            .pool_idle_timeout(Some(idle_timeout))
            .connect_timeout(connect_timeout)
            .timeout(request_timeout)
            .tcp_keepalive(if config.keep_alive {
                Some(idle_timeout)
            } else {
                None
            });

        if !config.keep_alive {
            let mut headers = HeaderMap::new();
            headers.insert(CONNECTION, HeaderValue::from_static("close"));
            builder = builder.default_headers(headers);
        }

        let client = builder
            .build()
            .map_err(|err| ConnectionPoolError::ConnectionFailed(err.to_string()))?;

        Ok(Self {
            client,
            config,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        })
    }

    fn validate_config(config: &ConnectionPoolConfig) -> Result<(), ConnectionPoolError> {
        if config.keep_alive && config.max_idle_per_host == 0 {
            return Err(ConnectionPoolError::ConfigError(
                "max_idle_per_host must be greater than 0 when keep_alive is enabled".to_string(),
            ));
        }
        if config.idle_timeout_secs == 0 {
            return Err(ConnectionPoolError::ConfigError(
                "idle_timeout_secs must be greater than 0".to_string(),
            ));
        }
        if config.connect_timeout_secs == 0 {
            return Err(ConnectionPoolError::ConfigError(
                "connect_timeout_secs must be greater than 0".to_string(),
            ));
        }
        if config.request_timeout_secs == 0 {
            return Err(ConnectionPoolError::ConfigError(
                "request_timeout_secs must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the configured reqwest client.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Return a snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats
            .lock()
            .expect("connection pool stats mutex poisoned")
            .clone()
    }

    /// Record a successful or attempted request.
    pub fn record_request(&self) {
        let mut stats = self
            .stats
            .lock()
            .expect("connection pool stats mutex poisoned");
        stats.total_requests += 1;
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        let mut stats = self
            .stats
            .lock()
            .expect("connection pool stats mutex poisoned");
        stats.failed_requests += 1;
    }

    /// Record a reconnection event.
    pub fn record_reconnection(&self) {
        let mut stats = self
            .stats
            .lock()
            .expect("connection pool stats mutex poisoned");
        stats.reconnections += 1;
    }
}

/// Manager for provider-specific connection pools.
#[derive(Debug, Clone, Default)]
pub struct ConnectionPoolManager {
    pools: Arc<RwLock<HashMap<String, ProviderConnectionPool>>>,
}

impl ConnectionPoolManager {
    /// Create an empty connection pool manager.
    pub fn new() -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register or replace a provider-specific connection pool.
    pub fn register_provider(
        &self,
        name: &str,
        config: ConnectionPoolConfig,
    ) -> Result<(), ConnectionPoolError> {
        if name.trim().is_empty() {
            return Err(ConnectionPoolError::ConfigError(
                "provider name cannot be empty".to_string(),
            ));
        }

        let pool = ProviderConnectionPool::new(config)?;
        self.pools.blocking_write().insert(name.to_string(), pool);
        Ok(())
    }

    /// Get a clone of the provider client.
    pub fn get_client(&self, provider_name: &str) -> Result<reqwest::Client, ConnectionPoolError> {
        self.pools
            .blocking_read()
            .get(provider_name)
            .map(|pool| pool.client().clone())
            .ok_or_else(|| ConnectionPoolError::ProviderNotFound(provider_name.to_string()))
    }

    /// Get a snapshot of a provider pool's statistics.
    pub fn pool_stats(&self, provider_name: &str) -> Result<PoolStats, ConnectionPoolError> {
        self.pools
            .blocking_read()
            .get(provider_name)
            .map(ProviderConnectionPool::stats)
            .ok_or_else(|| ConnectionPoolError::ProviderNotFound(provider_name.to_string()))
    }

    /// Get statistics for all registered providers.
    pub fn all_stats(&self) -> HashMap<String, PoolStats> {
        self.pools
            .blocking_read()
            .iter()
            .map(|(name, pool)| (name.clone(), pool.stats()))
            .collect()
    }

    /// Remove a provider pool. Returns true if a pool was removed.
    pub fn remove_provider(&self, name: &str) -> bool {
        self.pools.blocking_write().remove(name).is_some()
    }

    /// Return the number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.pools.blocking_read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_pool_default_config_values_are_reasonable() {
        let config = ConnectionPoolConfig::default();

        assert_eq!(config.max_idle_per_host, 32);
        assert_eq!(config.idle_timeout_secs, 90);
        assert_eq!(config.connect_timeout_secs, 10);
        assert_eq!(config.request_timeout_secs, 60);
        assert_eq!(config.max_retries, 3);
        assert!(config.keep_alive);
    }

    #[test]
    fn connection_pool_provider_pool_builds_client() {
        let pool = ProviderConnectionPool::new(ConnectionPoolConfig::default()).unwrap();

        let _client = pool.client().clone();
        assert_eq!(pool.stats(), PoolStats::default());
    }

    #[test]
    fn connection_pool_rejects_invalid_config() {
        let err = ProviderConnectionPool::new(ConnectionPoolConfig {
            max_idle_per_host: 0,
            ..ConnectionPoolConfig::default()
        })
        .unwrap_err();

        assert!(matches!(err, ConnectionPoolError::ConfigError(_)));
    }

    #[test]
    fn connection_pool_manager_registers_providers_and_gets_clients() {
        let manager = ConnectionPoolManager::new();

        manager
            .register_provider("openai", ConnectionPoolConfig::default())
            .unwrap();
        manager
            .register_provider(
                "anthropic",
                ConnectionPoolConfig {
                    max_idle_per_host: 16,
                    ..ConnectionPoolConfig::default()
                },
            )
            .unwrap();

        let _openai_client = manager.get_client("openai").unwrap();
        let _anthropic_client = manager.get_client("anthropic").unwrap();

        assert_eq!(manager.provider_count(), 2);
    }

    #[test]
    fn connection_pool_stats_tracking_updates_counters() {
        let pool = ProviderConnectionPool::new(ConnectionPoolConfig::default()).unwrap();

        pool.record_request();
        pool.record_request();
        pool.record_failure();
        pool.record_reconnection();
        pool.record_reconnection();

        let stats = pool.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.reconnections, 2);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.idle_connections, 0);
    }

    #[test]
    fn connection_pool_manager_removes_providers() {
        let manager = ConnectionPoolManager::new();
        manager
            .register_provider("openai", ConnectionPoolConfig::default())
            .unwrap();

        assert!(manager.remove_provider("openai"));
        assert!(!manager.remove_provider("openai"));
        assert_eq!(manager.provider_count(), 0);
    }

    #[test]
    fn connection_pool_unknown_provider_returns_error() {
        let manager = ConnectionPoolManager::new();

        let err = manager.get_client("unknown").unwrap_err();
        assert_eq!(
            err,
            ConnectionPoolError::ProviderNotFound("unknown".to_string())
        );
    }

    #[test]
    fn connection_pool_multiple_providers_can_coexist() {
        let manager = ConnectionPoolManager::new();
        manager
            .register_provider(
                "openai",
                ConnectionPoolConfig {
                    max_idle_per_host: 8,
                    ..ConnectionPoolConfig::default()
                },
            )
            .unwrap();
        manager
            .register_provider(
                "anthropic",
                ConnectionPoolConfig {
                    max_idle_per_host: 12,
                    request_timeout_secs: 120,
                    ..ConnectionPoolConfig::default()
                },
            )
            .unwrap();

        {
            let pools = manager.pools.blocking_read();
            let openai = pools.get("openai").unwrap();
            let anthropic = pools.get("anthropic").unwrap();

            openai.record_request();
            anthropic.record_request();
            anthropic.record_failure();
        }

        let all_stats = manager.all_stats();
        assert_eq!(all_stats.len(), 2);
        assert_eq!(all_stats["openai"].total_requests, 1);
        assert_eq!(all_stats["anthropic"].total_requests, 1);
        assert_eq!(all_stats["anthropic"].failed_requests, 1);
    }

    #[test]
    fn connection_pool_manager_returns_provider_stats() {
        let manager = ConnectionPoolManager::new();
        manager
            .register_provider("openai", ConnectionPoolConfig::default())
            .unwrap();

        {
            let pools = manager.pools.blocking_read();
            pools.get("openai").unwrap().record_request();
            pools.get("openai").unwrap().record_reconnection();
        }

        let stats = manager.pool_stats("openai").unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.reconnections, 1);
    }
}
