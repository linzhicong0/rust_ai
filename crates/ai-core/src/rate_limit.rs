// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Token bucket rate limiting for AI framework (REQ-12.4).
//!
//! Provides configurable rate limiting per provider, per model, and per agent
//! with RPM (requests per minute) and TPM (tokens per minute) limits and queuing.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

/// Configuration for rate limits.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per minute (RPM).
    pub rpm: Option<u64>,
    /// Maximum tokens per minute (TPM).
    pub tpm: Option<u64>,
}

impl RateLimitConfig {
    /// Create a new rate limit configuration with RPM only.
    pub fn rpm(rpm: u64) -> Self {
        Self {
            rpm: Some(rpm),
            tpm: None,
        }
    }

    /// Create a new rate limit configuration with TPM only.
    pub fn tpm(tpm: u64) -> Self {
        Self {
            rpm: None,
            tpm: Some(tpm),
        }
    }

    /// Create a new rate limit configuration with both RPM and TPM.
    pub fn new(rpm: u64, tpm: u64) -> Self {
        Self {
            rpm: Some(rpm),
            tpm: Some(tpm),
        }
    }
}

/// Token bucket for rate limiting.
#[derive(Debug)]
struct TokenBucket {
    /// Maximum tokens in the bucket (capacity).
    capacity: u64,
    /// Current available tokens.
    tokens: f64,
    /// Last time tokens were refilled.
    last_refill: Instant,
    /// Refill rate: tokens per second.
    refill_rate: f64,
}

impl TokenBucket {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            last_refill: Instant::now(),
            refill_rate: capacity as f64 / 60.0, // refill over 1 minute
        }
    }

    /// Try to consume `count` tokens. Returns true if successful.
    fn try_consume(&mut self, count: u64) -> bool {
        self.refill();
        if self.tokens >= count as f64 {
            self.tokens -= count as f64;
            true
        } else {
            false
        }
    }

    /// Time until `count` tokens become available.
    fn time_until_available(&mut self, count: u64) -> Duration {
        self.refill();
        if self.tokens >= count as f64 {
            return Duration::ZERO;
        }
        let needed = count as f64 - self.tokens;
        let seconds = needed / self.refill_rate;
        Duration::from_secs_f64(seconds)
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f64);
        self.last_refill = now;
    }
}

/// Rate limiter entry for a specific scope (provider/model/agent).
#[derive(Debug)]
struct RateLimiterEntry {
    /// RPM bucket (each request costs 1 token).
    rpm_bucket: Option<TokenBucket>,
    /// TPM bucket (each request costs N tokens based on usage).
    tpm_bucket: Option<TokenBucket>,
}

impl RateLimiterEntry {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            rpm_bucket: config.rpm.map(TokenBucket::new),
            tpm_bucket: config.tpm.map(TokenBucket::new),
        }
    }
}

/// Error returned when rate limit is exceeded.
#[derive(Debug, Clone)]
pub struct RateLimitExceeded {
    /// How long to wait before retrying.
    pub retry_after: Duration,
    /// Description of which limit was exceeded.
    pub message: String,
}

impl std::fmt::Display for RateLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rate limit exceeded: {}. Retry after {:?}",
            self.message, self.retry_after
        )
    }
}

/// Token bucket rate limiter supporting per-provider, per-model, and per-agent limits.
#[derive(Debug, Clone)]
pub struct TokenBucketRateLimiter {
    entries: Arc<Mutex<HashMap<String, RateLimiterEntry>>>,
    configs: Arc<Mutex<HashMap<String, RateLimitConfig>>>,
}

impl TokenBucketRateLimiter {
    /// Create a new token bucket rate limiter.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            configs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Configure rate limits for a specific scope.
    ///
    /// The scope can be a provider name, model name, agent name, or any string key.
    pub async fn configure(&self, scope: impl Into<String>, config: RateLimitConfig) {
        let scope = scope.into();
        let mut configs = self.configs.lock().await;
        configs.insert(scope.clone(), config.clone());
        let mut entries = self.entries.lock().await;
        entries.insert(scope, RateLimiterEntry::new(&config));
    }

    /// Try to acquire permission to make a request.
    ///
    /// Returns `Ok(())` if the request is allowed, or `Err(RateLimitExceeded)` if
    /// the rate limit is exceeded.
    ///
    /// # Arguments
    /// * `scope` - The rate limit scope (provider/model/agent name)
    /// * `estimated_tokens` - Estimated token count for TPM limiting
    pub async fn try_acquire(
        &self,
        scope: &str,
        estimated_tokens: u64,
    ) -> Result<(), RateLimitExceeded> {
        let mut entries = self.entries.lock().await;
        let entry = match entries.get_mut(scope) {
            Some(e) => e,
            None => return Ok(()), // No limit configured for this scope
        };

        // Check RPM
        if let Some(rpm_bucket) = &mut entry.rpm_bucket {
            if !rpm_bucket.try_consume(1) {
                let wait = rpm_bucket.time_until_available(1);
                return Err(RateLimitExceeded {
                    retry_after: wait,
                    message: format!("RPM limit exceeded for scope '{}'", scope),
                });
            }
        }

        // Check TPM
        if let Some(tpm_bucket) = &mut entry.tpm_bucket {
            if !tpm_bucket.try_consume(estimated_tokens) {
                let wait = tpm_bucket.time_until_available(estimated_tokens);
                return Err(RateLimitExceeded {
                    retry_after: wait,
                    message: format!("TPM limit exceeded for scope '{}'", scope),
                });
            }
        }

        Ok(())
    }

    /// Acquire permission to make a request, waiting (queuing) if necessary.
    ///
    /// This will block until the rate limit allows the request to proceed.
    pub async fn acquire(&self, scope: &str, estimated_tokens: u64) {
        loop {
            match self.try_acquire(scope, estimated_tokens).await {
                Ok(()) => return,
                Err(exceeded) => {
                    tokio::time::sleep(exceeded.retry_after).await;
                }
            }
        }
    }
}

impl Default for TokenBucketRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-12.4: Rate Limiting Tests

    #[tokio::test]
    async fn test_rpm_limit_allows_exactly_n_requests() {
        let limiter = TokenBucketRateLimiter::new();
        limiter
            .configure("test_provider", RateLimitConfig::rpm(10))
            .await;

        // Should allow exactly 10 requests
        for i in 0..10 {
            let result = limiter.try_acquire("test_provider", 0).await;
            assert!(
                result.is_ok(),
                "Request {} should be allowed, got: {:?}",
                i,
                result
            );
        }

        // 11th request should be rejected
        let result = limiter.try_acquire("test_provider", 0).await;
        assert!(result.is_err(), "11th request should be rate limited");
    }

    #[tokio::test]
    async fn test_queued_request_executes_after_window_resets() {
        let limiter = TokenBucketRateLimiter::new();
        // Use a higher RPM so token refill is fast enough for testing
        // RPM=60 means 1 token/second refill rate
        limiter
            .configure("test_provider", RateLimitConfig::rpm(60))
            .await;

        // Exhaust all 60 tokens
        for _ in 0..60 {
            assert!(limiter.try_acquire("test_provider", 0).await.is_ok());
        }

        // Next request should be rejected
        let err = limiter.try_acquire("test_provider", 0).await.unwrap_err();
        assert!(err.retry_after > Duration::ZERO);

        // Wait for at least 1 token to refill (1 token per second at RPM=60)
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // Now should be able to acquire again
        assert!(limiter.try_acquire("test_provider", 0).await.is_ok());
    }

    #[tokio::test]
    async fn test_tpm_limit_rejects_exceeding_token_budget() {
        let limiter = TokenBucketRateLimiter::new();
        limiter
            .configure("test_model", RateLimitConfig::tpm(100_000))
            .await;

        // Request using 50K tokens - should succeed
        assert!(limiter.try_acquire("test_model", 50_000).await.is_ok());

        // Request using another 50K tokens - should succeed (total 100K)
        assert!(limiter.try_acquire("test_model", 50_000).await.is_ok());

        // Request using 1 more token should fail (over 100K budget)
        let result = limiter.try_acquire("test_model", 1).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("TPM"));
    }

    #[tokio::test]
    async fn test_different_agents_have_independent_limits() {
        let limiter = TokenBucketRateLimiter::new();
        limiter.configure("agent_a", RateLimitConfig::rpm(2)).await;
        limiter.configure("agent_b", RateLimitConfig::rpm(2)).await;

        // Use up agent_a's limit
        assert!(limiter.try_acquire("agent_a", 0).await.is_ok());
        assert!(limiter.try_acquire("agent_a", 0).await.is_ok());
        assert!(limiter.try_acquire("agent_a", 0).await.is_err()); // Exhausted

        // agent_b should still work independently
        assert!(limiter.try_acquire("agent_b", 0).await.is_ok());
        assert!(limiter.try_acquire("agent_b", 0).await.is_ok());
        assert!(limiter.try_acquire("agent_b", 0).await.is_err()); // Now exhausted
    }

    #[tokio::test]
    async fn test_concurrent_requests_respect_rpm_and_tpm() {
        let limiter = TokenBucketRateLimiter::new();
        limiter
            .configure("provider", RateLimitConfig::new(5, 10_000))
            .await;

        // Launch 5 concurrent requests each using 1000 tokens
        let mut handles = vec![];
        for _ in 0..5 {
            let l = limiter.clone();
            handles.push(tokio::spawn(async move {
                l.try_acquire("provider", 1000).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(successes, 5, "All 5 should succeed within RPM limit");

        // 6th request should be rejected (RPM exhausted)
        let result = limiter.try_acquire("provider", 1000).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_no_limit_configured_allows_all() {
        let limiter = TokenBucketRateLimiter::new();
        // No configuration for "unconfigured_scope"
        for _ in 0..100 {
            assert!(limiter
                .try_acquire("unconfigured_scope", 1000)
                .await
                .is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limit_config_constructors() {
        let rpm_only = RateLimitConfig::rpm(60);
        assert_eq!(rpm_only.rpm, Some(60));
        assert_eq!(rpm_only.tpm, None);

        let tpm_only = RateLimitConfig::tpm(100_000);
        assert_eq!(tpm_only.rpm, None);
        assert_eq!(tpm_only.tpm, Some(100_000));

        let both = RateLimitConfig::new(60, 100_000);
        assert_eq!(both.rpm, Some(60));
        assert_eq!(both.tpm, Some(100_000));
    }
}
