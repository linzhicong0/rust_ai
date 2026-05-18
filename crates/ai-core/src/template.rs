// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Prompt template engine.
//!
//! The [`TemplateEngine`] provides template rendering for prompts using
//! the Tera templating language. Supports variable interpolation,
//! conditionals, and template inheritance.
//!
//! ## Features
//!
//! - Variable interpolation: `{{ variable }}`
//! - Conditionals: `{% if condition %}...{% endif %}`
//! - Loops: `{% for item in items %}...{% endfor %}`
//! - Template inheritance: `{% extends "base" %}` and `{% block name %}`
//! - Partials/include: `{% include "partial" %}`
//! - Custom filters and functions
//!
//! ## Example
//!
//! ```rust
//! use ai_core::template::TemplateEngine;
//! use serde_json::json;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = TemplateEngine::new()?;
//!
//! // Add a template
//! engine.add_template("greeting", "Hello {{ name }}!")?;
//!
//! // Render with context
//! let mut context = std::collections::HashMap::new();
//! context.insert("name".to_string(), json!("world"));
//!
//! let rendered = engine.render("greeting", &context)?;
//! assert_eq!(rendered, "Hello world!");
//!
//! // Load from file
//! engine.load_from_file("prompt", "templates/prompt.tera")?;
//!
//! // Use built-in agent templates
//! let rendered = engine.render_builtin("agent_default", &context)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;
use tera::{Tera, Context as TeraContext};
use tracing::{debug, info};


/// Prompt template engine using Tera.
///
/// Provides flexible template rendering for prompts with support for:
/// - Variable interpolation: `{{ variable }}`
/// - Conditionals: `{% if condition %}...{% endif %}`
/// - Loops: `{% for item in items %}...{% endfor %}`
/// - Template inheritance: `{% extends "base" %}`
pub struct TemplateEngine {
    tera: Tera,
    templates_dir: Option<std::path::PathBuf>,
}

impl TemplateEngine {
    /// Create a new template engine.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_core::template::TemplateEngine;
    /// let engine = TemplateEngine::new().unwrap();
    /// ```
    pub fn new() -> Result<Self, tera::Error> {
        Ok(Self {
            tera: Tera::default(),
            templates_dir: None,
        })
    }

    /// Create a new template engine with templates loaded from a directory.
    ///
    /// # Arguments
    ///
    /// * `dir` — Path to directory containing `.tera` template files
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ai_core::template::TemplateEngine;
    /// let engine = TemplateEngine::with_templates_dir("./templates").unwrap();
    /// ```
    pub fn with_templates_dir(dir: impl AsRef<Path>) -> Result<Self, tera::Error> {
        let dir = dir.as_ref();
        let dir_str = dir.to_str()
            .ok_or_else(|| tera::Error::msg("Invalid template directory path"))?;
        Ok(Self {
            tera: Tera::new(dir_str)?,
            templates_dir: Some(dir.to_path_buf()),
        })
    }

    /// Add a template to the engine.
    ///
    /// # Arguments
    ///
    /// * `name` — Unique identifier for this template
    /// * `template` — Template source using Tera syntax
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_core::template::TemplateEngine;
    /// # let mut engine = TemplateEngine::new().unwrap();
    /// engine.add_template("greeting", "Hello {{ name }}!").unwrap();
    /// ```
    pub fn add_template(&mut self, name: &str, template: &str) -> Result<(), tera::Error> {
        self.tera.add_raw_template(name, template)
    }

    /// Add multiple templates from a map.
    ///
    /// Useful for loading multiple templates at once.
    ///
    /// # Arguments
    ///
    /// * `templates` — Map of template names to template sources
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_core::template::TemplateEngine;
    /// # use std::collections::HashMap;
    /// # let mut engine = TemplateEngine::new().unwrap();
    /// let mut templates = HashMap::new();
    /// templates.insert("greeting".to_string(), "Hello {{ name }}!".to_string());
    /// templates.insert("farewell".to_string(), "Goodbye {{ name }}!".to_string());
    /// engine.add_templates_map(templates).unwrap();
    /// ```
    pub fn add_templates_map(&mut self, templates: HashMap<String, String>) -> Result<(), tera::Error> {
        for (name, content) in templates {
            self.tera.add_raw_template(&name, &content)?;
        }
        Ok(())
    }

    /// Load a template from a file.
    ///
    /// # Arguments
    ///
    /// * `name` — Name to give the template
    /// * `path` — Path to the template file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ai_core::template::TemplateEngine;
    /// # let mut engine = TemplateEngine::new().unwrap();
    /// engine.load_from_file("my_prompt", "templates/my_prompt.tera").unwrap();
    /// ```
    pub fn load_from_file(&mut self, name: &str, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        self.tera
            .add_raw_template(name, &content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Reload all templates from the templates directory.
    ///
    /// Useful during development for hot-reloading prompt changes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ai_core::template::TemplateEngine;
    /// # let mut engine = TemplateEngine::with_templates_dir("./templates").unwrap();
    /// engine.reload().unwrap();
    /// ```
    pub fn reload(&mut self) -> Result<(), tera::Error> {
        if let Some(dir) = &self.templates_dir {
            let dir_str = dir.to_str()
                .ok_or_else(|| tera::Error::msg("Invalid template directory path"))?;
            self.tera = Tera::new(dir_str)?;
            info!("Reloaded templates from {:?}", dir);
        }
        Ok(())
    }

    /// Render a template with the given context.
    ///
    /// # Arguments
    ///
    /// * `name` — Name of the template to render
    /// * `context` — Variables to interpolate into the template
    ///
    /// # Returns
    ///
    /// The rendered template as a string.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_core::template::TemplateEngine;
    /// # use serde_json::json;
    /// # let mut engine = TemplateEngine::new().unwrap();
    /// # engine.add_template("test", "Hello {{ name }}!").unwrap();
    /// let mut context = std::collections::HashMap::new();
    /// context.insert("name".to_string(), json!("world"));
    ///
    /// let rendered = engine.render("test", &context).unwrap();
    /// assert_eq!(rendered, "Hello world!");
    /// ```
    pub fn render(
        &self,
        name: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> Result<String, tera::Error> {
        debug!("Rendering template: {}", name);
        let ctx = TeraContext::from_serialize(context)?;
        self.tera.render(name, &ctx)
    }

    /// Render a built-in template.
    ///
    /// Built-in templates include:
    /// - `agent_default`: Generic agent with role and backstory
    /// - `agent_coder`: Coding assistant template
    /// - `agent_researcher`: Research assistant template
    /// - `agent_writer`: Writing assistant template
    ///
    /// # Arguments
    ///
    /// * `name` — Name of the built-in template
    /// * `context` — Variables to interpolate
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ai_core::template::TemplateEngine;
    /// # use serde_json::json;
    /// # let mut engine = TemplateEngine::new().unwrap();
    /// let mut context = std::collections::HashMap::new();
    /// context.insert("role".to_string(), json!("You are a helpful assistant."));
    ///
    /// let rendered = engine.render_builtin("agent_default", &context).unwrap();
    /// ```
    pub fn render_builtin(
        &mut self,
        name: &str,
        context: &HashMap<String, serde_json::Value>,
    ) -> Result<String, tera::Error> {
        // Add built-in template if not already present
        if !self.has_template(name) {
            let template = get_builtin_template(name)?;
            self.add_template(name, template)?;
        }
        self.render(name, context)
    }

    /// Check if a template exists.
    pub fn has_template(&self, name: &str) -> bool {
        self.tera.get_template_names().any(|n| n == name)
    }

    /// Get all registered template names.
    pub fn template_names(&self) -> Vec<String> {
        self.tera.get_template_names().map(|s| s.to_string()).collect()
    }

    /// Validate a template without adding it.
    ///
    /// Useful for checking template syntax before adding.
    ///
    /// # Arguments
    ///
    /// * `template` — Template source to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with details if invalid.
    pub fn validate(&self, template: &str) -> Result<(), tera::Error> {
        let mut tera = Tera::default();
        tera.add_raw_template("_validation_test_", template)
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self { tera: Tera::default(), templates_dir: None }
    }
}

// ===== Built-in Templates =====

const BUILTIN_AGENT_DEFAULT: &str = r#"
You are {{ role }}.

{% if backstory %}
Backstory: {{ backstory }}
{% endif %}

{% if goal %}
Your goal: {{ goal }}
{% endif %}

Instructions:
- Be helpful and accurate
- If you don't know something, say so
- Use available tools when appropriate
"#;

const BUILTIN_AGENT_CODER: &str = r#"
You are an expert programmer and coding assistant.

Role: {{ role|default(value="Senior Software Engineer") }}

{% if backstory %}
Background: {{ backstory }}
{% endif %}

Guidelines:
- Write clean, well-commented code
- Follow best practices for the language
- Explain your reasoning
- Consider edge cases and error handling
- Suggest tests when appropriate
"#;

const BUILTIN_AGENT_RESEARCHER: &str = r#"
You are a research assistant with access to information sources.

Role: {{ role|default(value="Research Assistant") }}

{% if backstory %}
Background: {{ backstory }}
{% endif %}

Guidelines:
- Find and cite accurate information
- Distinguish between facts and opinions
- Consider multiple perspectives
- Acknowledge uncertainty
- Provide sources when available
"#;

const BUILTIN_AGENT_WRITER: &str = r#"
You are a writing assistant.

Role: {{ role|default(value="Writing Coach") }}

{% if backstory %}
Background: {{ backstory }}
{% endif %}

Guidelines:
- Adapt tone and style to the context
- Be clear and concise
- Use active voice when appropriate
- Vary sentence structure
- Check grammar and spelling
"#;

/// Get a built-in template by name.
fn get_builtin_template(name: &str) -> Result<&'static str, tera::Error> {
    match name {
        "agent_default" => Ok(BUILTIN_AGENT_DEFAULT),
        "agent_coder" => Ok(BUILTIN_AGENT_CODER),
        "agent_researcher" => Ok(BUILTIN_AGENT_RESEARCHER),
        "agent_writer" => Ok(BUILTIN_AGENT_WRITER),
        _ => Err(tera::Error::msg(format!("Unknown built-in template: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_template_rendering() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "Hello {{ name }}!").unwrap();

        let mut context = HashMap::new();
        context.insert("name".to_string(), json!("world"));

        let rendered = engine.render("test", &context).unwrap();
        assert_eq!(rendered, "Hello world!");
    }

    #[test]
    fn test_conditional() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("cond", "{% if show_greeting %}Hello!{% endif %}World")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("show_greeting".to_string(), json!(true));

        let rendered = engine.render("cond", &context).unwrap();
        assert_eq!(rendered, "Hello!World");
    }

    #[test]
    fn test_loop() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("loop", "Items: {% for item in items %}{{ item }} {% endfor %}")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("items".to_string(), json!(["a", "b", "c"]));

        let rendered = engine.render("loop", &context).unwrap();
        assert!(rendered.contains("a"));
        assert!(rendered.contains("b"));
        assert!(rendered.contains("c"));
    }

    #[test]
    fn test_validate() {
        let engine = TemplateEngine::new().unwrap();

        // Valid template
        assert!(engine.validate("Hello {{ name }}!").is_ok());

        // Invalid template
        assert!(engine.validate("Hello {{ name }!").is_err());
    }

    #[test]
    fn test_builtin_template() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Test Agent"));
        context.insert("backstory".to_string(), json!("A test agent."));
        context.insert("goal".to_string(), json!("To test templates."));

        let rendered = engine.render_builtin("agent_default", &context).unwrap();
        assert!(rendered.contains("Test Agent"));
        assert!(rendered.contains("A test agent."));
        assert!(rendered.contains("To test templates."));
    }

    #[test]
    fn test_has_template() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "Hello {{ name }}!").unwrap();

        assert!(engine.has_template("test"));
        assert!(!engine.has_template("nonexistent"));
    }

    #[test]
    fn test_template_names() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test1", "Hello {{ name }}!").unwrap();
        engine.add_template("test2", "Goodbye {{ name }}!").unwrap();

        let names = engine.template_names();
        assert!(names.contains(&"test1".to_string()));
        assert!(names.contains(&"test2".to_string()));
    }
}
