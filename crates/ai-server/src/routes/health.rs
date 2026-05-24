use axum::{routing::get, Json, Router};
use serde::Serialize;

use crate::app::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health))
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}
