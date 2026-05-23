use std::collections::HashMap;

use axum::{extract::Path, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/pipelines/{id}/execute", post(execute_pipeline))
}

#[derive(Debug, Deserialize)]
pub struct PipelineStepRequest {
    pub name: String,
    #[serde(default)]
    pub config: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct PipelineExecuteRequest {
    pub input: String,
    #[serde(default)]
    pub steps: Vec<PipelineStepRequest>,
}

#[derive(Debug, Serialize)]
pub struct PipelineExecuteResponse {
    pub id: String,
    pub result: String,
    pub steps_completed: usize,
}

pub async fn execute_pipeline(
    Path(id): Path<String>,
    Json(request): Json<PipelineExecuteRequest>,
) -> Json<PipelineExecuteResponse> {
    let steps_completed = request.steps.len();
    let step_names = request
        .steps
        .iter()
        .map(|step| step.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    Json(PipelineExecuteResponse {
        id: id.clone(),
        result: format!(
            "Placeholder pipeline execution for '{id}'. Received input '{}' and would execute {} step(s){}.",
            request.input,
            steps_completed,
            if step_names.is_empty() {
                String::new()
            } else {
                format!(": {step_names}")
            }
        ),
        steps_completed,
    })
}
