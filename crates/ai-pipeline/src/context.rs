use serde_json::Value;
use std::collections::HashMap;

pub struct PipelineContext {
    pub data: HashMap<String, Value>,
}

impl PipelineContext {
    pub fn new(input: Value) -> Self {
        let mut data = HashMap::new();
        data.insert("input".to_string(), input);
        Self { data }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.data.insert(key.into(), value);
    }
}
