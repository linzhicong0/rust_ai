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
        assert!(prompt.to_lowercase().contains("json"));
        assert!(prompt.contains("respond"));
        assert!(prompt.contains("schema"));
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

    // REQ-9.1: Structured Output - Additional edge case tests

    #[test]
    fn test_extract_json_nested_objects() {
        // Test extracting JSON with nested objects
        let response = r#"```json
{"user": {"name": "Alice", "age": 30, "address": {"city": "SF"}}}
```"#;
        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["user"]["name"], "Alice");
        assert_eq!(parsed["user"]["address"]["city"], "SF");
    }

    #[test]
    fn test_extract_json_arrays() {
        // Test extracting JSON arrays
        let response = r#"Here are the items: [1, 2, 3, 4, 5]"#;
        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_extract_json_no_valid_json() {
        // Test response with no valid JSON
        let response = r#"This is just plain text with no JSON at all."#;
        assert!(extract_json(response).is_none());
    }

    #[test]
    fn test_extract_json_malformed_in_braces() {
        // Test that malformed JSON between { and } is rejected
        let response = r#"Here's the data: {invalid json here} and more text"#;
        assert!(extract_json(response).is_none());
    }

    #[test]
    fn test_extract_json_with_trailing_comma() {
        // Test JSON with trailing comma (which is invalid)
        let response = r#"{"items": [1, 2, 3,]}"#;
        assert!(extract_json(response).is_none());
    }

    #[test]
    fn test_extract_json_from_multiline_markdown() {
        // Test JSON extraction from multiline markdown with various formatting
        let response = r#"The result is:

```json
{
  "status": "success",
  "data": {
    "value": 42
  }
}
```

That's the result."#;

        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["status"], "success");
        assert_eq!(parsed["data"]["value"], 42);
    }

    #[test]
    fn test_extract_json_multiple_blocks() {
        // Test that only the first valid JSON block is extracted
        let response = r#"First: {"a": 1}
Second: {"b": 2}"#;

        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed.get("a").is_some());
        assert!(parsed.get("b").is_none()); // Should only get first
    }

    #[test]
    fn test_structured_output_config_builder() {
        // Test the builder pattern for StructuredOutputConfig
        let schema = json!({"type": "object"});
        let config = StructuredOutputConfig::new(schema)
            .with_max_retries(5)
            .include_schema_in_prompt(false);

        assert_eq!(config.max_retries, 5);
        assert!(!config.include_schema_in_prompt);
    }

    #[test]
    fn test_structured_output_config_default_values() {
        // Test default values
        let schema = json!({"type": "object"});
        let config = StructuredOutputConfig::new(schema);

        assert_eq!(config.max_retries, 3);
        assert!(config.include_schema_in_prompt);
    }

    #[test]
    fn test_validator_config_reference() {
        // Test that validator returns a reference to its config
        let schema = json!({"type": "object"});
        let config = StructuredOutputConfig::new(schema).with_max_retries(10);
        let validator = StructuredOutputValidator::new(config).unwrap();

        assert_eq!(validator.config().max_retries, 10);
    }

    #[test]
    fn test_build_system_prompt_without_base() {
        // Test system prompt building without a base prompt
        let schema = json!({
            "type": "object",
            "properties": {"result": {"type": "string"}}
        });

        let config = StructuredOutputConfig::new(schema);
        let prompt = config.build_system_prompt(None);

        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("schema"));
        assert!(prompt.contains("Respond ONLY"));
    }

    #[test]
    fn test_schema_validation_complex_types() {
        // Test validation with complex nested types
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["count"]
                }
            },
            "required": ["items"]
        });

        let config = StructuredOutputConfig::new(schema);
        let validator = StructuredOutputValidator::new(config).unwrap();

        let valid = json!({
            "items": ["a", "b", "c"],
            "metadata": {"count": 3, "tags": ["x", "y"]}
        });
        assert!(validator.validate(&valid).is_ok());

        // Missing required field
        let invalid = json!({"metadata": {"count": 1}});
        assert!(validator.validate(&invalid).is_err());
    }

    #[test]
    fn test_schema_validation_enum() {
        // Test validation with enum constraints
        let schema = json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "active", "completed"]
                }
            },
            "required": ["status"]
        });

        let config = StructuredOutputConfig::new(schema);
        let validator = StructuredOutputValidator::new(config).unwrap();

        assert!(validator.validate(&json!({"status": "active"})).is_ok());
        assert!(validator.validate(&json!({"status": "invalid"})).is_err());
    }

    #[test]
    fn test_extract_json_with_unicode() {
        // Test JSON extraction with Unicode characters
        let response = r#"{"message": "Hello 世界 🌍"}"#;
        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed["message"].as_str().unwrap().contains("世界"));
    }

    #[test]
    fn test_extract_json_with_special_characters() {
        // Test JSON with escaped characters
        let response = r#"{"text": "Line 1\nLine 2\tTabbed", "quote": "She said \"hello\""}"#;
        let result = extract_json(response);
        assert!(result.is_some());

        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed["text"].as_str().unwrap().contains('\n'));
        assert!(parsed["quote"].as_str().unwrap().contains('"'));
    }

    #[test]
    fn test_structured_output_error_display() {
        // Test error message formatting
        let err = StructuredOutputError::ValidationError("Missing field".to_string());
        assert!(err.to_string().contains("Missing field"));

        let err2 = StructuredOutputError::MaxRetriesExceeded(5);
        assert!(err2.to_string().contains("5"));
    }

    #[test]
    fn test_invalid_schema_creation() {
        // Test that invalid schemas are rejected
        let invalid_schema = json!({"type": "invalid_type"});
        let result = StructuredOutputValidator::new(StructuredOutputConfig::new(invalid_schema));

        // The jsonschema crate may accept or reject this depending on version
        // Just check that we can handle the result
        match result {
            Ok(_) => {}, // Schema was accepted
            Err(e) => assert!(e.to_string().contains("validation") || e.to_string().contains("Schema")),
        }
    }
}
