//! Pipeline context for flowing data between steps.
//!
//! The [`PipelineContext`] holds the data that flows through a pipeline execution.
//! Each step can read from and write to the context.

use serde_json::Value;
use std::collections::HashMap;

/// Context that carries data through pipeline execution.
///
/// Each step in a pipeline can read from and write to the context,
/// allowing data to flow between steps.
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// The data stored in this context.
    pub data: HashMap<String, Value>,
}

impl PipelineContext {
    /// Create a new context with an initial input value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let ctx = PipelineContext::new(json!("initial data"));
    /// ```
    pub fn new(input: Value) -> Self {
        let mut data = HashMap::new();
        data.insert("input".to_string(), input);
        Self { data }
    }

    /// Create a new empty context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    ///
    /// let ctx = PipelineContext::empty();
    /// ```
    pub fn empty() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Get a value from the context by key.
    ///
    /// Returns `None` if the key doesn't exist.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("name", json!("Alice"));
    /// assert_eq!(ctx.get("name"), Some(&json!("Alice")));
    /// ```
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Get a value from the context, returning an error if not found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use ai_core::error::PipelineError;
    /// use serde_json::json;
    ///
    /// # fn main() -> Result<(), PipelineError> {
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("name", json!("Alice"));
    /// assert_eq!(ctx.require("name")?, &json!("Alice"));
    /// assert!(ctx.require("missing").is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn require(&self, key: &str) -> Result<&Value, crate::PipelineError> {
        self.get(key)
            .ok_or_else(|| crate::PipelineError::Context(format!("Missing required key: {}", key)))
    }

    /// Set a value in the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("greeting", json!("Hello, world!"));
    /// ```
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.data.insert(key.into(), value);
    }

    /// Check if a key exists in the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    ///
    /// let ctx = PipelineContext::empty();
    /// assert!(!ctx.has("name"));
    /// ```
    pub fn has(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all keys in the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("a", json!(1));
    /// ctx.set("b", json!(2));
    /// let keys: Vec<_> = ctx.keys().collect();
    /// assert_eq!(keys.len(), 2);
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Get the number of items in the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// assert_eq!(ctx.len(), 0);
    /// ctx.set("a", json!(1));
    /// assert_eq!(ctx.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the context is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    ///
    /// let ctx = PipelineContext::empty();
    /// assert!(ctx.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Merge another context into this one.
    ///
    /// Existing keys will be overwritten by values from `other`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx1 = PipelineContext::empty();
    /// ctx1.set("a", json!(1));
    ///
    /// let mut ctx2 = PipelineContext::empty();
    /// ctx2.set("b", json!(2));
    /// ctx2.set("a", json!(999));
    ///
    /// ctx1.merge(&ctx2);
    /// assert_eq!(ctx1.get("a"), Some(&json!(999)));
    /// assert_eq!(ctx1.get("b"), Some(&json!(2)));
    /// ```
    pub fn merge(&mut self, other: &PipelineContext) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
    }

    /// Remove a key from the context, returning its value if it existed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("temp", json!("value"));
    /// let value = ctx.remove("temp");
    /// assert_eq!(value, Some(json!("value")));
    /// assert!(!ctx.has("temp"));
    /// ```
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.data.remove(key)
    }

    /// Clear all data from the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::PipelineContext;
    /// use serde_json::json;
    ///
    /// let mut ctx = PipelineContext::empty();
    /// ctx.set("a", json!(1));
    /// ctx.clear();
    /// assert!(ctx.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_new() {
        let ctx = PipelineContext::new(json!("input"));
        assert_eq!(ctx.get("input"), Some(&json!("input")));
    }

    #[test]
    fn test_get_set() {
        let mut ctx = PipelineContext::empty();
        ctx.set("key", json!("value"));
        assert_eq!(ctx.get("key"), Some(&json!("value")));
    }

    #[test]
    fn test_require() {
        let mut ctx = PipelineContext::empty();
        ctx.set("key", json!("value"));
        assert_eq!(ctx.require("key").unwrap(), &json!("value"));
        assert!(ctx.require("missing").is_err());
    }

    #[test]
    fn test_has() {
        let mut ctx = PipelineContext::empty();
        assert!(!ctx.has("key"));
        ctx.set("key", json!("value"));
        assert!(ctx.has("key"));
    }

    #[test]
    fn test_merge() {
        let mut ctx1 = PipelineContext::empty();
        ctx1.set("a", json!(1));
        ctx1.set("b", json!(2));

        let mut ctx2 = PipelineContext::empty();
        ctx2.set("b", json!(20));
        ctx2.set("c", json!(3));

        ctx1.merge(&ctx2);
        assert_eq!(ctx1.get("a"), Some(&json!(1)));
        assert_eq!(ctx1.get("b"), Some(&json!(20)));
        assert_eq!(ctx1.get("c"), Some(&json!(3)));
    }

    #[test]
    fn test_remove() {
        let mut ctx = PipelineContext::empty();
        ctx.set("key", json!("value"));
        assert_eq!(ctx.remove("key"), Some(json!("value")));
        assert!(!ctx.has("key"));
        assert_eq!(ctx.remove("key"), None);
    }

    #[test]
    fn test_clear() {
        let mut ctx = PipelineContext::empty();
        ctx.set("a", json!(1));
        ctx.set("b", json!(2));
        ctx.clear();
        assert!(ctx.is_empty());
    }
}
