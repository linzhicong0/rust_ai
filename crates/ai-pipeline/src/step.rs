use serde_json::Value;
use std::collections::HashMap;

use ai_core::provider::Provider;

pub struct Step {
    pub name: String,
    pub kind: StepKind,
}

pub enum StepKind {
    Task {
        agent_name: String,
        input_key: String,
        output_key: String,
    },
    Parallel(Vec<Step>),
    Conditional {
        condition_key: String,
        condition_value: Value,
        then_step: Box<Step>,
        else_step: Option<Box<Step>>,
    },
    Loop {
        body: Box<Step>,
        condition_key: String,
        condition_value: Value,
        max_iterations: u32,
    },
}
