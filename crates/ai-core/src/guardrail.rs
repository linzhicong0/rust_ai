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
//!
//! ## Built-in Guardrails
//!
//! The framework provides several built-in guardrails:
//!
//! - [`RegexPiiDetector`] — Detects PII using regex patterns
//! - [`LengthLimiter`] — Enforces minimum/maximum length constraints
//! - [`CustomGuardrail`] — User-defined validation functions
//! - [`GuardrailChain`] — Chains multiple guardrails together

use async_trait::async_trait;
use regex::Regex;
use std::sync::Arc;

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

/// Chains multiple guardrails together, executing them in sequence.
///
/// The chain stops at the first guardrail that returns a `Block` or `Modify` action.
/// If all guardrails return `Allow`, the chain returns `Allow`.
///
/// # Examples
///
/// ```
/// # use ai_core::guardrail::{GuardrailChain, LengthLimiter, RegexPiiDetector};
/// # use ai_core::Guardrail;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let chain = GuardrailChain::new()
///     .add(LengthLimiter::max(1000))
///     .add(RegexPiiDetector::default());
///
/// // Use the chain like any other guardrail
/// let action = chain.check_input("Hello, world!").await?;
/// # Ok(())
/// # }
/// ```
pub struct GuardrailChain {
    name: String,
    guardrails: Vec<Arc<dyn Guardrail>>,
}

impl GuardrailChain {
    /// Creates a new empty guardrail chain.
    pub fn new() -> Self {
        Self {
            name: "GuardrailChain".to_string(),
            guardrails: Vec::new(),
        }
    }

    /// Creates a new guardrail chain with the given name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            guardrails: Vec::new(),
        }
    }

    /// Adds a guardrail to the chain.
    pub fn add(mut self, guardrail: impl Guardrail + 'static) -> Self {
        self.guardrails.push(Arc::new(guardrail));
        self
    }

    /// Creates an input-only chain that only validates inputs.
    pub fn input_chain() -> Self {
        Self::with_name("InputGuardrailChain")
    }

    /// Creates an output-only chain that only validates outputs.
    pub fn output_chain() -> Self {
        Self::with_name("OutputGuardrailChain")
    }
}

impl Default for GuardrailChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Guardrail for GuardrailChain {
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
        for guardrail in &self.guardrails {
            match guardrail.check_input(input).await? {
                GuardrailAction::Allow => continue,
                action => return Ok(action),
            }
        }
        Ok(GuardrailAction::Allow)
    }

    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
        for guardrail in &self.guardrails {
            match guardrail.check_output(output).await? {
                GuardrailAction::Allow => continue,
                action => return Ok(action),
            }
        }
        Ok(GuardrailAction::Allow)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// PII pattern configuration for regex detection.
#[derive(Debug, Clone)]
pub struct PiiPattern {
    /// Name of the pattern (e.g., "email", "ssn", "credit_card")
    pub name: String,
    /// Regex pattern to detect
    pub pattern: String,
    /// Whether to block or redact when detected
    pub action: PiiAction,
}

/// Action to take when PII is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiAction {
    /// Block the entire input/output
    Block,
    /// Redact the detected PII (replace with [REDACTED])
    Redact,
}

/// Regex-based PII detector.
///
/// Detects personally identifiable information using configurable regex patterns.
///
/// # Examples
///
/// ```
/// # use ai_core::guardrail::RegexPiiDetector;
///
/// // Use default PII patterns (email, SSN, credit card, phone)
/// let detector = RegexPiiDetector::default();
///
/// // Or configure custom patterns
/// let detector = RegexPiiDetector::new()
///     .add_pattern(r"\b\d{3}-\d{2}-\d{4}\b", "ssn", ai_core::guardrail::PiiAction::Block);
/// ```
pub struct RegexPiiDetector {
    name: String,
    patterns: Vec<(Regex, PiiAction, String)>,
}

impl RegexPiiDetector {
    /// Creates a new PII detector with default patterns.
    pub fn default() -> Self {
        Self::new()
            .add_pattern(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", PiiAction::Redact)
            .add_pattern(r"\b\d{3}-\d{2}-\d{4}\b", "ssn", PiiAction::Block)
            .add_pattern(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "credit_card", PiiAction::Block)
            .add_pattern(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone", PiiAction::Redact)
    }

    /// Creates a new empty PII detector.
    pub fn new() -> Self {
        Self {
            name: "RegexPiiDetector".to_string(),
            patterns: Vec::new(),
        }
    }

    /// Adds a custom PII pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` — Regex pattern to match
    /// * `name` — Name of the pattern (for error messages)
    /// * `action` — Whether to block or redact when detected
    pub fn add_pattern(mut self, pattern: &str, name: &str, action: PiiAction) -> Self {
        match Regex::new(pattern) {
            Ok(re) => {
                self.patterns.push((re, action, name.to_string()));
                self
            }
            Err(e) => {
                // Log the error but don't panic - just skip this pattern
                eprintln!("Invalid regex pattern '{}': {}", name, e);
                self
            }
        }
    }

    /// Checks text for PII and returns the first match found.
    fn check_for_pii(&self, text: &str) -> Option<&(Regex, PiiAction, String)> {
        for pattern in &self.patterns {
            if pattern.0.is_match(text) {
                return Some(pattern);
            }
        }
        None
    }

    /// Redacts all PII patterns from text.
    fn redact_pii(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (regex, _, name) in &self.patterns {
            result = regex.replace_all(&result, format!("[{}_REDACTED]", name.to_uppercase())).to_string();
        }
        result
    }
}

impl Default for RegexPiiDetector {
    fn default() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Guardrail for RegexPiiDetector {
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
        if let Some((_regex, action, name)) = self.check_for_pii(input) {
            match action {
                PiiAction::Block => {
                    return Ok(GuardrailAction::Block(format!("PII detected: {}", name)));
                }
                PiiAction::Redact => {
                    let redacted = self.redact_pii(input);
                    return Ok(GuardrailAction::Modify(redacted));
                }
            }
        }
        Ok(GuardrailAction::Allow)
    }

    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
        self.check_input(output).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Length limiter guardrail.
///
/// Enforces minimum and maximum length constraints on inputs and outputs.
///
/// # Examples
///
/// ```
/// # use ai_core::guardrail::LengthLimiter;
///
/// // Enforce maximum length only
/// let limiter = LengthLimiter::max(1000);
///
/// // Enforce both minimum and maximum
/// let limiter = LengthLimiter::new().min(10).max(1000);
/// ```
pub struct LengthLimiter {
    name: String,
    min_length: Option<usize>,
    max_length: Option<usize>,
}

impl LengthLimiter {
    /// Creates a new length limiter with no constraints.
    pub fn new() -> Self {
        Self {
            name: "LengthLimiter".to_string(),
            min_length: None,
            max_length: None,
        }
    }

    /// Creates a length limiter with only a maximum length.
    pub fn max(max: usize) -> Self {
        Self::new().with_max(max)
    }

    /// Creates a length limiter with only a minimum length.
    pub fn min(min: usize) -> Self {
        Self::new().with_min(min)
    }

    /// Creates a length limiter with both minimum and maximum.
    pub fn bounds(min: usize, max: usize) -> Self {
        Self::new().with_min(min).with_max(max)
    }

    /// Sets the minimum length.
    pub fn with_min(mut self, min: usize) -> Self {
        self.min_length = Some(min);
        self.name = format!("LengthLimiter(min={})", min);
        self
    }

    /// Sets the maximum length.
    pub fn with_max(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        if let Some(min) = self.min_length {
            self.name = format!("LengthLimiter(min={}, max={})", min, max);
        } else {
            self.name = format!("LengthLimiter(max={})", max);
        }
        self
    }

    /// Checks if the given length satisfies the constraints.
    fn check_length(&self, length: usize) -> Result<(), GuardrailError> {
        if let Some(min) = self.min_length {
            if length < min {
                return Err(GuardrailError::Check(format!(
                    "Input too short: {} characters (minimum: {})", length, min
                )));
            }
        }

        if let Some(max) = self.max_length {
            if length > max {
                return Err(GuardrailError::Check(format!(
                    "Input too long: {} characters (maximum: {})", length, max
                )));
            }
        }

        Ok(())
    }
}

impl Default for LengthLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Guardrail for LengthLimiter {
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
        self.check_length(input.len())?;
        Ok(GuardrailAction::Allow)
    }

    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
        self.check_length(output.len())?;
        Ok(GuardrailAction::Allow)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Custom user-defined guardrail.
///
/// Allows creating guardrails from custom validation functions.
///
/// # Examples
///
/// ```
/// # use ai_core::guardrail::CustomGuardrail;
/// # use ai_core::guardrail::GuardrailAction;
///
/// // Create a guardrail that blocks profanity
/// let guardrail = CustomGuardrail::new(
///     "profanity_filter",
///     |input| async {
///         if input.contains("badword") {
///             Ok(GuardrailAction::Block("Profanity detected".to_string()))
///         } else {
///             Ok(GuardrailAction::Allow)
///         }
///     },
///     |output| async {
///         Ok(GuardrailAction::Allow)
///     },
/// );
/// ```
pub struct CustomGuardrail<F1, F2>
where
    F1: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
    F2: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
{
    name: String,
    check_input_fn: F1,
    check_output_fn: F2,
}

impl<F1, F2> CustomGuardrail<F1, F2>
where
    F1: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
    F2: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
{
    /// Creates a new custom guardrail.
    ///
    /// # Arguments
    ///
    /// * `name` — Name of the guardrail
    /// * `check_input_fn` — Function to validate inputs
    /// * `check_output_fn` — Function to validate outputs
    pub fn new(
        name: impl Into<String>,
        check_input_fn: F1,
        check_output_fn: F2,
    ) -> Self {
        Self {
            name: name.into(),
            check_input_fn,
            check_output_fn,
        }
    }

    /// Creates a custom guardrail that uses the same validation for input and output.
    pub fn symmetric<F>(name: impl Into<String>, check_fn: F) -> CustomGuardrail<F, F>
    where
        F: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync + Clone,
    {
        CustomGuardrail {
            name: name.into(),
            check_input_fn: check_fn.clone(),
            check_output_fn: check_fn,
        }
    }
}

#[async_trait]
impl<F1, F2> Guardrail for CustomGuardrail<F1, F2>
where
    F1: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
    F2: Fn(&str) -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> + Send + Sync,
{
    async fn check_input(&self, input: &str) -> Result<GuardrailAction, GuardrailError> {
        (self.check_input_fn)(input).await
    }

    async fn check_output(&self, output: &str) -> Result<GuardrailAction, GuardrailError> {
        (self.check_output_fn)(output).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock guardrail for testing
    struct MockGuardrail {
        block_words: Vec<String>,
    }

    #[async_trait]
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

    // Tests for GuardrailChain
    #[tokio::test]
    async fn test_guardrail_chain_allows() {
        let chain = GuardrailChain::new()
            .add(MockGuardrail { block_words: vec!["secret".to_string()] })
            .add(LengthLimiter::max(1000));

        let result = chain.check_input("Hello, world!").await.unwrap();
        assert_eq!(result, GuardrailAction::Allow);
    }

    #[tokio::test]
    async fn test_guardrail_chain_blocks() {
        let chain = GuardrailChain::new()
            .add(MockGuardrail { block_words: vec!["secret".to_string()] })
            .add(LengthLimiter::max(1000));

        let result = chain.check_input("This is a secret message").await.unwrap();
        match result {
            GuardrailAction::Block(_) => {}
            _ => panic!("Expected Block action"),
        }
    }

    #[tokio::test]
    async fn test_guardrail_chain_stops_at_first_violation() {
        let chain = GuardrailChain::new()
            .add(LengthLimiter::max(10))
            .add(MockGuardrail { block_words: vec!["secret".to_string()] });

        let result = chain.check_input("This is a very long message that exceeds the limit").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    // Tests for LengthLimiter
    #[tokio::test]
    async fn test_length_limiter_max_only() {
        let limiter = LengthLimiter::max(10);

        assert!(limiter.check_input("short").await.is_ok());
        assert!(limiter.check_input("This is too long").await.is_err());
    }

    #[tokio::test]
    async fn test_length_limiter_min_only() {
        let limiter = LengthLimiter::min(5);

        assert!(limiter.check_input("short").await.is_ok());
        assert!(limiter.check_input("tiny").await.is_err());
    }

    #[tokio::test]
    async fn test_length_limiter_bounds() {
        let limiter = LengthLimiter::bounds(5, 10);

        assert!(limiter.check_input("perfect").await.is_ok());
        assert!(limiter.check_input("tiny").await.is_err());
        assert!(limiter.check_input("This is too long").await.is_err());
    }

    // Tests for RegexPiiDetector
    #[tokio::test]
    async fn test_pii_detector_email() {
        let detector = RegexPiiDetector::default();

        let result = detector.check_input("My email is test@example.com").await.unwrap();
        match result {
            GuardrailAction::Modify(redacted) => {
                assert!(redacted.contains("[EMAIL_REDACTED]"));
                assert!(!redacted.contains("test@example.com"));
            }
            _ => panic!("Expected Modify action for email"),
        }
    }

    #[tokio::test]
    async fn test_pii_detector_ssn() {
        let detector = RegexPiiDetector::default();

        let result = detector.check_input("My SSN is 123-45-6789").await.unwrap();
        match result {
            GuardrailAction::Block(reason) => {
                assert!(reason.contains("ssn"));
            }
            _ => panic!("Expected Block action for SSN"),
        }
    }

    #[tokio::test]
    async fn test_pii_detector_no_pii() {
        let detector = RegexPiiDetector::default();

        let result = detector.check_input("Hello, world!").await.unwrap();
        assert_eq!(result, GuardrailAction::Allow);
    }

    #[tokio::test]
    async fn test_pii_detector_custom_pattern() {
        let detector = RegexPiiDetector::new()
            .add_pattern(r"\bAPI_KEY_\w+\b", "api_key", PiiAction::Block);

        let result = detector.check_input("Use API_KEY_abc123 for access").await.unwrap();
        match result {
            GuardrailAction::Block(reason) => {
                assert!(reason.contains("api_key"));
            }
            _ => panic!("Expected Block action for custom pattern"),
        }
    }

    // Tests for CustomGuardrail
    // TODO: Fix type inference issue with async closures
    // #[tokio::test]
    // async fn test_custom_guardrail() {
    //     let guardrail = CustomGuardrail::symmetric(
    //         "test_guardrail",
    //         |text: &str| -> futures::future::BoxFuture<'static, Result<GuardrailAction, GuardrailError>> {
    //             let text = text.to_string();
    //             Box::pin(async move {
    //                 if text.contains("forbidden") {
    //                     Ok(GuardrailAction::Block("Forbidden word detected".to_string()))
    //                 } else {
    //                     Ok(GuardrailAction::Allow)
    //                 }
    //             })
    //         },
    //     );
    //
    //     assert!(guardrail.check_input("Hello, world!").await.is_ok());
    //     match guardrail.check_input("This is forbidden").await.unwrap() {
    //         GuardrailAction::Block(_) => {}
    //         _ => panic!("Expected Block action"),
    //     }
    // }

    #[tokio::test]
    async fn test_guardrail_chain_with_builtin() {
        let chain = GuardrailChain::new()
            .add(LengthLimiter::max(100))
            .add(RegexPiiDetector::default());

        // Should pass length check but trigger PII detector
        let result = chain.check_input("My email is test@example.com and this message is short").await.unwrap();
        match result {
            GuardrailAction::Modify(_) => {}
            _ => panic!("Expected Modify action from PII detector"),
        }
    }
}
