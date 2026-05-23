use std::{collections::HashSet, sync::Arc};

use axum::{
    extract::{FromRef, FromRequestParts},
    http::{header, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct AuthState {
    valid_api_keys: Arc<HashSet<String>>,
}

impl AuthState {
    pub fn new<I, K>(keys: I) -> Self
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
    {
        let valid_api_keys = keys.into_iter().map(Into::into).collect();
        Self {
            valid_api_keys: Arc::new(valid_api_keys),
        }
    }

    pub fn is_valid_key(&self, key: &str) -> bool {
        self.valid_api_keys.contains(key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiKey(pub String);

impl ApiKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ApiKeyAuth;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("missing or invalid API key")]
    Unauthorized,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody {
                error: &self.to_string(),
            }),
        )
            .into_response()
    }
}

impl<S> FromRequestParts<S> for ApiKeyAuth
where
    AuthState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let auth_state = AuthState::from_ref(state);
        let api_key = extract_api_key(parts).ok_or(AuthError::Unauthorized)?;

        if !auth_state.is_valid_key(&api_key) {
            return Err(AuthError::Unauthorized);
        }

        parts.extensions.insert(ApiKey(api_key));
        Ok(Self)
    }
}

pub(crate) fn extract_api_key(parts: &Parts) -> Option<String> {
    parts
        .headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            parts
                .headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_bearer_token)
        })
}

fn parse_bearer_token(value: &str) -> Option<String> {
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }

    let token = token.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, extract::FromRequestParts, http::Request};

    use super::*;

    #[tokio::test]
    async fn accepts_x_api_key_header() {
        let state = AuthState::new(["secret-key"]);
        let request = Request::builder()
            .header("x-api-key", "secret-key")
            .body(Body::empty())
            .unwrap();
        let (mut parts, _) = request.into_parts();

        let result = ApiKeyAuth::from_request_parts(&mut parts, &state).await;

        assert!(result.is_ok());
        assert_eq!(
            parts.extensions.get::<ApiKey>().unwrap().as_str(),
            "secret-key"
        );
    }

    #[tokio::test]
    async fn accepts_bearer_token_header() {
        let state = AuthState::new(["bearer-key"]);
        let request = Request::builder()
            .header(header::AUTHORIZATION, "Bearer bearer-key")
            .body(Body::empty())
            .unwrap();
        let (mut parts, _) = request.into_parts();

        let result = ApiKeyAuth::from_request_parts(&mut parts, &state).await;

        assert!(result.is_ok());
        assert_eq!(
            parts.extensions.get::<ApiKey>().unwrap().as_str(),
            "bearer-key"
        );
    }

    #[tokio::test]
    async fn rejects_missing_api_key() {
        let state = AuthState::new(["secret-key"]);
        let request = Request::builder().body(Body::empty()).unwrap();
        let (mut parts, _) = request.into_parts();

        let error = ApiKeyAuth::from_request_parts(&mut parts, &state)
            .await
            .unwrap_err();

        assert_eq!(error.into_response().status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rejects_invalid_api_key() {
        let state = AuthState::new(["secret-key"]);
        let request = Request::builder()
            .header("x-api-key", "wrong-key")
            .body(Body::empty())
            .unwrap();
        let (mut parts, _) = request.into_parts();

        let error = ApiKeyAuth::from_request_parts(&mut parts, &state)
            .await
            .unwrap_err();

        assert_eq!(error.into_response().status(), StatusCode::UNAUTHORIZED);
    }
}
