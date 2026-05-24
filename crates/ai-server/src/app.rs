use std::{collections::HashSet, env, time::Duration};

use ai_core::config::ServerConfig;
use axum::{middleware, Router};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

use crate::{
    middleware::{
        auth::{ApiKeyAuth, AuthState},
        rate_limit::{RateLimit, RateLimitState},
    },
    routes::{agents, health, pipelines},
};

#[derive(Clone, Debug)]
pub struct AppState {
    pub config: ServerConfig,
    pub auth: AuthState,
    pub rate_limit: RateLimitState,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            auth: AuthState::new(load_api_keys()),
            rate_limit: RateLimitState::new(load_rate_limit(), Duration::from_secs(60)),
        }
    }
}

impl axum::extract::FromRef<AppState> for AuthState {
    fn from_ref(app_state: &AppState) -> Self {
        app_state.auth.clone()
    }
}

impl axum::extract::FromRef<AppState> for RateLimitState {
    fn from_ref(app_state: &AppState) -> Self {
        app_state.rate_limit.clone()
    }
}

pub fn create_app(config: ServerConfig) -> Router {
    let state = AppState::new(config);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderName::from_static("x-api-key"),
        ]);

    Router::new()
        .merge(health::router())
        .merge(agents::router())
        .merge(pipelines::router())
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(middleware::from_extractor_with_state::<RateLimit, _>(
            state.clone(),
        ))
        .layer(middleware::from_extractor_with_state::<ApiKeyAuth, _>(
            state.clone(),
        ))
        .with_state(state)
}

fn load_api_keys() -> HashSet<String> {
    env::var("AI_SERVER_API_KEYS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>()
        })
        .filter(|keys| !keys.is_empty())
        .unwrap_or_else(|| HashSet::from(["dev-api-key".to_string()]))
}

fn load_rate_limit() -> u64 {
    env::var("AI_SERVER_RATE_LIMIT_PER_MINUTE")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(60)
}
