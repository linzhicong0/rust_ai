//! Input and output validation guardrails.
//!
//! Guardrails validate and optionally modify LLM inputs and outputs to
//! ensure safety, policy compliance, and output quality.
//!
//! ## Example
//!
//! ```rust,no_run
//! # use ai_core::{Guardrail, GuardrailError};
//! # use ai_core::guardrail::GuardrailAction;
//! struct PiiFilter;
//!
//! # #[async_trait::async_trait]
//! impl Guardrail for PiiFilter {
//! #     async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
//!         if input.contains("SSN:") {
//!             return Ok(GuardrailAction::Block("PII detected".to_string()));
//!         }
//!         Ok(GuardrailAction::Allow)
//! #     }
//!
//! #     async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
//!         Ok(GuardrailAction::Allow)
//! #     }
//!
//! #     fn name(&self) -> &str {
//!         "pii_filter"
//! #     }
//! }
//! ```

use async_trait::async_trait;

use crate::error::GuardrailError;

/// Action to take after guardrail evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardrailAction {
    /// Allow the input or output to proceed unchanged.
    Allow,

    /// Block the input or output with the given reason.
    Block(String),

    /// Allow after modifying the content (sanitization, redaction, etc.).
    Modify(String),
}

/// A guardrail for validating inputs and outputs.
///
/// Guardrails can check for:
/// - PII (personally identifiable information)
/// - Toxic or harmful content
/// - Policy violations
/// - Format requirements
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Check an input (user message, prompt, etc.) before sending to the LLM.
    ///
    /// # Arguments
    ///
    /// * `input` — The input to validate
    ///
    /// # Returns
    ///
    /// An action indicating whether to allow, block, or modify the input.
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError>;

    /// Check an output (LLM response) before returning to the user.
    ///
    /// # Arguments
    ///
    /// * `output` — The output to validate
    ///
    /// # Returns
    ///
    /// An action indicating whether to allow, block, or modify the output.
    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError>;

    /// Returns the name of this guardrail.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock guardrail for testing
    struct MockGuardrail {
        block_words: Vec<String>,
    }

    #[async_trait::async_trait]
    impl Guardrail for MockGuardrail {
        async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
            for word in &self.block_words {
                if input.contains(word) {
                    return Ok(GuardrailAction::Block(format!("Blocked word: {}", word)));
                }
            }
            Ok(GuardrailAction::Allow)
        }

        async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
            for word in &self.block_words {
                if output.contains(word) {
                    return Ok(GuardrailAction::Modify(
                        output.replace(word, "[REDACTED]")
                    ));
                }
            }
            Ok(GuardrailAction::Allow)
        }

        fn name(&self) -> &str {
            "mock_guardrail"
        }
    }

    #[tokio::test]
    async fn test_guardrail_allow() {
        let guardrail = MockGuardrail {
            block_words: vec!["secret".to_string()],
        };

        let result = guardrail.check_input("Hello, world!").await.unwrap();
        assert_eq!(result, GuardrailAction::Allow);
    }

    #[tokio::test]
    async fn test_guardrail_block_input() {
        let guardrail = MockGuardrail {
            block_words: vec!["secret".to_string()],
        };

        let result = guardrail.check_input("This is a secret message").await.unwrap();
        match result {
            GuardrailAction::Block(reason) => {
                assert!(reason.contains("secret"));
            }
            _ => panic!("Expected Block action"),
        }
    }

    #[tokio::test]
    async fn test_guardrail_modify_output() {
        let guardrail = MockGuardrail {
            block_words: vec!["password".to_string()],
        };

        let result = guardrail
            .check_output("Your password is 12345")
            .await
            .unwrap();

        match result {
            GuardrailAction::Modify(modified) => {
                assert!(modified.contains("[REDACTED]"));
                assert!(!modified.contains("password"));
            }
            _ => panic!("Expected Modify action"),
        }
    }

    #[tokio::test]
    async fn test_guardrail_name() {
        let guardrail = MockGuardrail {
            block_words: vec![],
        };
        assert_eq!(guardrail.name(), "mock_guardrail");
    }

    #[test]
    fn test_guardrail_action_equality() {
        assert_eq!(GuardrailAction::Allow, GuardrailAction::Allow);
        assert_ne!(GuardrailAction::Allow, GuardrailAction::Block("reason".to_string()));

        let block1 = GuardrailAction::Block("test".to_string());
        let block2 = GuardrailAction::Block("test".to_string());
        assert_eq!(block1, block2);
    }
}
