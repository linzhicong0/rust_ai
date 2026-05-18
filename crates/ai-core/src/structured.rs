// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Structured output support for LLM responses.
//!
//! This module provides JSON schema validation and retry logic for ensuring
//! LLM responses conform to expected schemas.

use jsonschema::Validator;
use serde_json::Value;
use thiserror::Error;

/// Errors that can occur during structured output validation.
#[derive(Debug, Error)]
pub enum StructuredOutputError {
    /// Schema validation failed.
    #[error("Schema validation failed: {0}")]
    ValidationError(String),

    /// JSON parsing failed.
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),

    /// Max retries exceeded.
    #[error("Max retries ({0}) exceeded for structured output")]
    MaxRetriesExceeded(usize),
}

/// Configuration for structured output requests.
#[derive(Debug, Clone)]
pub struct StructuredOutputConfig {
    /// JSON schema for validation.
    pub schema: Value,

    /// Maximum number of retry attempts.
    pub max_retries: usize,

    /// Whether to include the schema in the system prompt.
    pub include_schema_in_prompt: bool,
}

impl StructuredOutputConfig {
    /// Create a new structured output configuration.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            max_retries: 3,
            include_schema_in_prompt: true,
        }
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Whether to include the schema in the system prompt.
    pub fn include_schema_in_prompt(mut self, include: bool) -> Self {
        self.include_schema_in_prompt = include;
        self
    }

    /// Build a system prompt that includes the JSON schema.
    pub fn build_system_prompt(&self, base_prompt: Option<&str>) -> String {
        let schema_str = serde_json::to_string_pretty(&self.schema)
            .unwrap_or_else(|_| "{}".to_string());

        let mut instructions = "You must respond with valid JSON that conforms to the following schema:\n\n".to_string();
        instructions.push_str(&schema_str);
        instructions.push_str("\n\nRespond ONLY with the JSON object, no additional text.");

        if let Some(base) = base_prompt {
            format!("{}\n\n{}", base, instructions)
        } else {
            instructions
        }
    }
}

/// Validator for structured output.
pub struct StructuredOutputValidator {
    schema: Validator,
    config: StructuredOutputConfig,
}

impl StructuredOutputValidator {
    /// Create a new validator from a JSON schema.
    pub fn new(config: StructuredOutputConfig) -> Result<Self, StructuredOutputError> {
        let validator = Validator::new(&config.schema)
            .map_err(|e| StructuredOutputError::ValidationError(e.to_string()))?;

        Ok(Self { schema: validator, config })
    }

    /// Validate a JSON response against the schema.
    pub fn validate(&self, response: &Value) -> Result<(), StructuredOutputError> {
        self.schema.validate(response)
            .map_err(|e| StructuredOutputError::ValidationError(e.to_string()))
    }

    /// Validate a string response by parsing it as JSON first.
    pub fn validate_str(&self, response: &str) -> Result<Value, StructuredOutputError> {
        let parsed: Value = serde_json::from_str(response)?;
        self.validate(&parsed)?;
        Ok(parsed)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &StructuredOutputConfig {
        &self.config
    }
}

/// Helper to extract JSON from a response that may include markdown code blocks.
pub fn extract_json(response: &str) -> Option<String> {
    let response = response.trim();

    // Try parsing as-is first
    if serde_json::from_str::<Value>(response).is_ok() {
        return Some(response.to_string());
    }

    // Try extracting from markdown code blocks
    if let Some(start) = response.find("```json") {
        let start = start + 7; // len("```json")
        if let Some(end) = response[start..].find("```") {
            let json_str = response[start..start + end].trim();
            if serde_json::from_str::<Value>(json_str).is_ok() {
                return Some(json_str.to_string());
            }
        }
    }

    if let Some(start) = response.find("```") {
        let start = start + 3; // len("```")
        if let Some(end) = response[start..].find("```") {
            let json_str = response[start..start + end].trim();
            if serde_json::from_str::<Value>(json_str).is_ok() {
                return Some(json_str.to_string());
            }
        }
    }

    // Try to find the first { and last } and extract that
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            if end > start {
                let json_str = &response[start..=end];
                if serde_json::from_str::<Value>(json_str).is_ok() {
                    return Some(json_str.to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_json_from_plain() {
        let response = r#"{"name": "test", "value": 42}"#;
        assert_eq!(extract_json(response), Some(response.to_string()));
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let response = r#"Here's the result:
```json
{"name": "test", "value": 42}
```"#;
        let expected = r#"{"name": "test", "value": 42}"#;
        assert_eq!(extract_json(response), Some(expected.to_string()));
    }

    #[test]
    fn test_extract_json_from_code_block() {
        let response = r#"Here's the result:
```
{"name": "test", "value": 42}
```"#;
        let expected = r#"{"name": "test", "value": 42}"#;
        assert_eq!(extract_json(response), Some(expected.to_string()));
    }

    #[test]
    fn test_extract_json_from_text() {
        let response = r#"Some text before {"name": "test"} and after"#;
        let expected = r#"{"name": "test"}"#;
        assert_eq!(extract_json(response), Some(expected.to_string()));
    }

    #[test]
    fn test_schema_validation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });

        let config = StructuredOutputConfig::new(schema);
        let validator = StructuredOutputValidator::new(config).unwrap();

        let valid = json!({"name": "Alice", "age": 30});
        assert!(validator.validate(&valid).is_ok());

        let invalid = json!({"name": "Bob"}); // missing age
        assert!(validator.validate(&invalid).is_err());
    }

    #[test]
    fn test_build_system_prompt() {
        let schema = json!({
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        });

        let config = StructuredOutputConfig::new(schema);
        let prompt = config.build_system_prompt(Some("You are a helpful assistant."));

        assert!(prompt.contains("You are a helpful assistant"));
        assert!(prompt.contains("JSON schema"));
        assert!(prompt.contains("respond with valid JSON"));
    }

    #[test]
    fn test_validate_str() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        });

        let config = StructuredOutputConfig::new(schema);
        let validator = StructuredOutputValidator::new(config).unwrap();

        let response = r#"{"value": 123}"#;
        let result = validator.validate_str(response).unwrap();
        assert_eq!(result["value"], 123);

        let invalid = r#"{"value": "not a number"}"#;
        assert!(validator.validate_str(invalid).is_err());
    }
}
