use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    extract::{FromRef, FromRequestParts},
    http::{header::RETRY_AFTER, request::Parts, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

use crate::middleware::auth::{extract_api_key, ApiKey};

#[derive(Clone, Debug)]
pub struct RateLimitState {
    max_requests: u64,
    window: Duration,
    entries: Arc<Mutex<HashMap<String, RateLimitEntry>>>,
}

#[derive(Clone, Debug)]
struct RateLimitEntry {
    count: u64,
    window_started_at: Instant,
}

impl RateLimitState {
    pub fn new(max_requests: u64, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            entries: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn check_key(&self, key: &str) -> Result<(), RateLimitError> {
        let now = Instant::now();
        let mut entries = self.entries.lock().expect("rate limit mutex poisoned");
        let entry = entries.entry(key.to_owned()).or_insert(RateLimitEntry {
            count: 0,
            window_started_at: now,
        });

        if now.duration_since(entry.window_started_at) >= self.window {
            entry.count = 0;
            entry.window_started_at = now;
        }

        if entry.count >= self.max_requests {
            let retry_after = self
                .window
                .saturating_sub(now.duration_since(entry.window_started_at))
                .as_secs()
                .max(1);
            return Err(RateLimitError::TooManyRequests { retry_after });
        }

        entry.count += 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RateLimit;

#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("rate limit exceeded")]
    TooManyRequests { retry_after: u64 },
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> Response {
        match self {
            Self::TooManyRequests { retry_after } => {
                let mut response = (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(ErrorBody {
                        error: "rate limit exceeded",
                    }),
                )
                    .into_response();
                response.headers_mut().insert(
                    RETRY_AFTER,
                    HeaderValue::from_str(&retry_after.to_string())
                        .unwrap_or_else(|_| HeaderValue::from_static("1")),
                );
                response
            }
        }
    }
}

impl<S> FromRequestParts<S> for RateLimit
where
    RateLimitState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = RateLimitError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let rate_limit_state = RateLimitState::from_ref(state);
        let api_key = parts
            .extensions
            .get::<ApiKey>()
            .map(|api_key| api_key.0.clone())
            .or_else(|| extract_api_key(parts));

        if let Some(api_key) = api_key {
            rate_limit_state.check_key(&api_key)?;
        }

        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Body, extract::FromRequestParts, http::Request};

    use super::*;

    #[tokio::test]
    async fn allows_requests_within_limit() {
        let state = RateLimitState::new(2, Duration::from_secs(60));

        for _ in 0..2 {
            let request = Request::builder()
                .header("x-api-key", "key-1")
                .body(Body::empty())
                .unwrap();
            let (mut parts, _) = request.into_parts();
            let result = RateLimit::from_request_parts(&mut parts, &state).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn rejects_requests_over_limit() {
        let state = RateLimitState::new(1, Duration::from_secs(60));

        let first = Request::builder()
            .header("x-api-key", "key-1")
            .body(Body::empty())
            .unwrap();
        let (mut first_parts, _) = first.into_parts();
        assert!(RateLimit::from_request_parts(&mut first_parts, &state)
            .await
            .is_ok());

        let second = Request::builder()
            .header("x-api-key", "key-1")
            .body(Body::empty())
            .unwrap();
        let (mut second_parts, _) = second.into_parts();
        let error = RateLimit::from_request_parts(&mut second_parts, &state)
            .await
            .unwrap_err();

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(RETRY_AFTER));
    }

    #[tokio::test]
    async fn resets_after_window_expires() {
        let state = RateLimitState::new(1, Duration::from_millis(25));

        let first = Request::builder()
            .header("x-api-key", "key-1")
            .body(Body::empty())
            .unwrap();
        let (mut first_parts, _) = first.into_parts();
        assert!(RateLimit::from_request_parts(&mut first_parts, &state)
            .await
            .is_ok());

        tokio::time::sleep(Duration::from_millis(30)).await;

        let second = Request::builder()
            .header("x-api-key", "key-1")
            .body(Body::empty())
            .unwrap();
        let (mut second_parts, _) = second.into_parts();
        assert!(RateLimit::from_request_parts(&mut second_parts, &state)
            .await
            .is_ok());
    }
}
