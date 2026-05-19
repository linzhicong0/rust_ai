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
//! # use ai_core::template::TemplateEngine;
//! # use serde_json::json;
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
//! # // engine.load_from_file("prompt", "templates/prompt.tera")?;
//!
//! // Use built-in agent templates
//! # context.insert("role".to_string(), json!("Agent"));
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

    // REQ-4.1: Template System Tests

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
    fn test_conditional_false() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("cond", "{% if show_greeting %}Hello!{% endif %}World")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("show_greeting".to_string(), json!(false));

        let rendered = engine.render("cond", &context).unwrap();
        assert_eq!(rendered, "World");
    }

    #[test]
    fn test_conditional_if_else() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("cond", "{% if is_admin %}Admin{% else %}User{% endif %}")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("is_admin".to_string(), json!(false));

        let rendered = engine.render("cond", &context).unwrap();
        assert_eq!(rendered, "User");
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
    fn test_loop_empty() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("loop", "{% for item in items %}{{ item }}{% else %}empty{% endfor %}")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("items".to_string(), json!([]));

        let rendered = engine.render("loop", &context).unwrap();
        assert_eq!(rendered, "empty");
    }

    #[test]
    fn test_loop_with_index() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("loop", "{% for item in items %}{{ loop.index }}: {{ item }} {% endfor %}")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("items".to_string(), json!(["x", "y"]));

        let rendered = engine.render("loop", &context).unwrap();
        assert!(rendered.contains("1: x"));
        assert!(rendered.contains("2: y"));
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
    fn test_validate_complex() {
        let engine = TemplateEngine::new().unwrap();

        // Valid complex template
        let valid = r#"
            {% if show %}
                Hello {{ name }}!
                {% for item in items %}
                    {{ item }}
                {% endfor %}
            {% endif %}
        "#;
        assert!(engine.validate(valid).is_ok());

        // Invalid: unclosed if
        let invalid = r#"
            {% if show %}
                Hello
            {# missing endif #}
        "#;
        assert!(engine.validate(invalid).is_err());
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

    #[test]
    fn test_template_names_empty() {
        let engine = TemplateEngine::new().unwrap();
        assert_eq!(engine.template_names(), Vec::<String>::new());
    }

    #[test]
    fn test_builtin_template_agent_default() {
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
    fn test_builtin_template_agent_default_without_optional() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Test Agent"));

        let rendered = engine.render_builtin("agent_default", &context).unwrap();
        assert!(rendered.contains("Test Agent"));
        assert!(!rendered.contains("Backstory:"));
        assert!(!rendered.contains("Your goal:"));
    }

    #[test]
    fn test_builtin_template_agent_coder() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Senior Developer"));
        context.insert("backstory".to_string(), json!("10 years of experience"));

        let rendered = engine.render_builtin("agent_coder", &context).unwrap();
        assert!(rendered.contains("Senior Developer"));
        assert!(rendered.contains("10 years of experience"));
        assert!(rendered.contains("clean, well-commented code"));
    }

    #[test]
    fn test_builtin_template_agent_coder_defaults() {
        let mut engine = TemplateEngine::new().unwrap();

        let context = HashMap::new();
        let rendered = engine.render_builtin("agent_coder", &context).unwrap();
        // Uses default value from template
        assert!(rendered.contains("Senior Software Engineer"));
    }

    #[test]
    fn test_builtin_template_agent_researcher() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Research Analyst"));

        let rendered = engine.render_builtin("agent_researcher", &context).unwrap();
        assert!(rendered.contains("Research Analyst"));
        assert!(rendered.contains("Find and cite accurate information"));
    }

    #[test]
    fn test_builtin_template_agent_writer() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Content Writer"));

        let rendered = engine.render_builtin("agent_writer", &context).unwrap();
        assert!(rendered.contains("Content Writer"));
        assert!(rendered.contains("Adapt tone and style"));
    }

    #[test]
    fn test_builtin_template_unknown() {
        let mut engine = TemplateEngine::new().unwrap();
        let context = HashMap::new();

        let result = engine.render_builtin("unknown_template", &context);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown built-in template"));
    }

    #[test]
    fn test_add_templates_map() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut templates = HashMap::new();
        templates.insert("greeting".to_string(), "Hello {{ name }}!".to_string());
        templates.insert("farewell".to_string(), "Goodbye {{ name }}!".to_string());

        engine.add_templates_map(templates).unwrap();

        let mut context = HashMap::new();
        context.insert("name".to_string(), json!("World"));

        assert_eq!(engine.render("greeting", &context).unwrap(), "Hello World!");
        assert_eq!(engine.render("farewell", &context).unwrap(), "Goodbye World!");
    }

    #[test]
    fn test_add_templates_map_empty() {
        let mut engine = TemplateEngine::new().unwrap();
        let templates = HashMap::new();

        let result = engine.add_templates_map(templates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_templates_map_invalid_template() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut templates = HashMap::new();
        templates.insert("valid".to_string(), "Hello {{ name }}!".to_string());
        templates.insert("invalid".to_string(), "Hello {{ name }!".to_string());

        let result = engine.add_templates_map(templates);
        assert!(result.is_err());
    }

    #[test]
    fn test_render_missing_variable() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "Hello {{ name }}!").unwrap();

        let context = HashMap::new();

        // Tera fails on missing variables
        let result = engine.render("test", &context);
        assert!(result.is_err());
        // Error message should mention the variable name or template
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("name") || err_msg.contains("test"));
    }

    #[test]
    fn test_render_nested_objects() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "User: {{ user.name }}").unwrap();

        let mut context = HashMap::new();
        context.insert("user".to_string(), json!({"name": "Alice"}));

        let rendered = engine.render("test", &context).unwrap();
        assert_eq!(rendered, "User: Alice");
    }

    #[test]
    fn test_render_filters() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "{{ name | upper }}").unwrap();

        let mut context = HashMap::new();
        context.insert("name".to_string(), json!("hello"));

        let rendered = engine.render("test", &context).unwrap();
        assert_eq!(rendered, "HELLO");
    }

    #[test]
    fn test_template_default_trait() {
        let engine = TemplateEngine::default();
        assert!(!engine.has_template("anything"));
    }

    #[test]
    fn test_variable_interpolation_multiple() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "{{ greeting }}, {{ name }}!").unwrap();

        let mut context = HashMap::new();
        context.insert("greeting".to_string(), json!("Hello"));
        context.insert("name".to_string(), json!("World"));

        let rendered = engine.render("test", &context).unwrap();
        assert_eq!(rendered, "Hello, World!");
    }

    #[test]
    fn test_whitespace_control() {
        let mut engine = TemplateEngine::new().unwrap();
        engine
            .add_template("test", "A {% if show %} B {% endif %} C")
            .unwrap();

        let mut context = HashMap::new();
        context.insert("show".to_string(), json!(true));

        let rendered = engine.render("test", &context).unwrap();
        // Note: Tera preserves whitespace by default
        assert!(rendered.contains("A"));
        assert!(rendered.contains("B"));
        assert!(rendered.contains("C"));
    }

    #[test]
    fn test_template_render_twice() {
        let mut engine = TemplateEngine::new().unwrap();
        engine.add_template("test", "Value: {{ x }}").unwrap();

        let mut context1 = HashMap::new();
        context1.insert("x".to_string(), json!(1));
        assert_eq!(engine.render("test", &context1).unwrap(), "Value: 1");

        let mut context2 = HashMap::new();
        context2.insert("x".to_string(), json!(2));
        assert_eq!(engine.render("test", &context2).unwrap(), "Value: 2");
    }

    #[test]
    fn test_render_builtin_caches_template() {
        let mut engine = TemplateEngine::new().unwrap();

        let mut context = HashMap::new();
        context.insert("role".to_string(), json!("Test"));

        // First call adds the template
        engine.render_builtin("agent_default", &context).unwrap();
        // Template should now exist
        assert!(engine.has_template("agent_default"));

        // Second call uses cached template
        engine.render_builtin("agent_default", &context).unwrap();
    }
}
