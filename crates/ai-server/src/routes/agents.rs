use std::collections::HashMap;

use axum::{extract::Path, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/agents/{id}/run", post(run_agent))
}

#[derive(Debug, Deserialize)]
pub struct AgentRunRequest {
    pub input: String,
    #[serde(default)]
    pub config: HashMap<String, Value>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct UsageResponse {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AgentRunResponse {
    pub id: String,
    pub output: String,
    pub usage: UsageResponse,
}

pub async fn run_agent(
    Path(id): Path<String>,
    Json(request): Json<AgentRunRequest>,
) -> Json<AgentRunResponse> {
    let prompt_tokens = request.input.split_whitespace().count() as u32;
    let completion_tokens = 0;
    let config_keys = request.config.len();

    Json(AgentRunResponse {
        id: id.clone(),
        output: format!(
            "Placeholder agent execution for '{id}'. Received input '{}' with {config_keys} config entr{}.",
            request.input,
            if config_keys == 1 { "y" } else { "ies" }
        ),
        usage: UsageResponse {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}
