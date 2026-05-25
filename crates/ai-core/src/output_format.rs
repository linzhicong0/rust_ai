// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Output formatting support for agent and task responses.
//!
//! Provides configurable output formatting: plain text, Markdown, code blocks,
//! tables, and custom templates (REQ-9.4).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Supported output formats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Plain text output (no formatting applied).
    PlainText,
    /// Markdown formatted output.
    Markdown,
    /// Wrap output in a code block with optional language hint.
    CodeBlock(Option<String>),
    /// Render structured data as a markdown table.
    Table,
    /// Custom template-based formatting.
    Custom(String),
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::PlainText
    }
}

/// Configuration for output formatting on a task or agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFormatConfig {
    /// The output format to use.
    pub format: OutputFormat,
    /// Optional custom templates (name -> template string).
    pub templates: HashMap<String, String>,
}

impl Default for OutputFormatConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::PlainText,
            templates: HashMap::new(),
        }
    }
}

impl OutputFormatConfig {
    /// Create a new output format configuration.
    pub fn new(format: OutputFormat) -> Self {
        Self {
            format,
            templates: HashMap::new(),
        }
    }

    /// Add a custom template.
    pub fn with_template(mut self, name: impl Into<String>, template: impl Into<String>) -> Self {
        self.templates.insert(name.into(), template.into());
        self
    }
}

/// Format output according to the specified format.
///
/// Applies post-processing transformation to the raw output based on
/// the configured format.
pub fn format_output(content: &str, config: &OutputFormatConfig) -> String {
    match &config.format {
        OutputFormat::PlainText => content.to_string(),
        OutputFormat::Markdown => format_as_markdown(content),
        OutputFormat::CodeBlock(lang) => format_as_code_block(content, lang.as_deref()),
        OutputFormat::Table => format_as_table(content),
        OutputFormat::Custom(template_name) => {
            if let Some(template) = config.templates.get(template_name) {
                apply_template(content, template)
            } else {
                content.to_string()
            }
        }
    }
}

/// Format content as Markdown.
///
/// Ensures proper markdown formatting with headers and structure.
fn format_as_markdown(content: &str) -> String {
    // If it already looks like markdown (has headers, lists, etc.), return as-is
    if content.contains('#') || content.contains("- ") || content.contains("* ") {
        return content.to_string();
    }
    // Otherwise, treat each line as a paragraph
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Format content as a code block.
fn format_as_code_block(content: &str, language: Option<&str>) -> String {
    let lang = language.unwrap_or("");
    format!("```{}\n{}\n```", lang, content)
}

/// Format structured data (JSON) as a markdown table.
///
/// Expects content to be a JSON array of objects. Each object becomes a row,
/// and the keys become column headers.
fn format_as_table(content: &str) -> String {
    // Try to parse as JSON array of objects
    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(content) {
        if items.is_empty() {
            return "| (empty) |\n|---------|".to_string();
        }

        // Collect all keys from first object for headers
        let headers: Vec<String> = if let Some(Value::Object(first)) = items.first() {
            first.keys().cloned().collect()
        } else {
            return content.to_string();
        };

        if headers.is_empty() {
            return content.to_string();
        }

        // Build header row
        let header_row = format!("| {} |", headers.join(" | "));
        let separator = format!(
            "| {} |",
            headers
                .iter()
                .map(|_| "---")
                .collect::<Vec<_>>()
                .join(" | ")
        );

        // Build data rows
        let mut rows = vec![header_row, separator];
        for item in &items {
            if let Value::Object(obj) = item {
                let cells: Vec<String> = headers
                    .iter()
                    .map(|h| {
                        obj.get(h)
                            .map(|v| match v {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                rows.push(format!("| {} |", cells.join(" | ")));
            }
        }

        rows.join("\n")
    } else {
        // Not valid JSON array, return as-is
        content.to_string()
    }
}

/// Apply a custom template to content.
///
/// Templates use `{{content}}` as a placeholder for the output content.
/// Additional placeholders like `{{title}}`, `{{date}}` can be used.
fn apply_template(content: &str, template: &str) -> String {
    template.replace("{{content}}", content)
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-9.4: Output Formatting Tests

    #[test]
    fn test_format_markdown_produces_properly_formatted_output() {
        let config = OutputFormatConfig::new(OutputFormat::Markdown);
        let input = "Line one\nLine two\nLine three";
        let result = format_output(input, &config);
        // Should format as separate paragraphs
        assert!(result.contains("Line one"));
        assert!(result.contains("Line two"));
        assert!(result.contains("\n\n")); // paragraphs separated
    }

    #[test]
    fn test_format_code_block_wraps_output() {
        let config = OutputFormatConfig::new(OutputFormat::CodeBlock(Some("rust".to_string())));
        let input = "fn main() {\n    println!(\"Hello\");\n}";
        let result = format_output(input, &config);
        assert!(result.starts_with("```rust\n"));
        assert!(result.ends_with("\n```"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_format_table_produces_markdown_table() {
        let config = OutputFormatConfig::new(OutputFormat::Table);
        let input = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
        let result = format_output(input, &config);
        assert!(result.contains("|"));
        assert!(result.contains("---"));
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
        assert!(result.contains("30"));
        assert!(result.contains("25"));
    }

    #[test]
    fn test_custom_template_applies_user_defined_pattern() {
        let config = OutputFormatConfig::new(OutputFormat::Custom("report".to_string()))
            .with_template("report", "# Report\n\n{{content}}\n\n---\nEnd of report");
        let input = "Some analysis results here";
        let result = format_output(input, &config);
        assert!(result.starts_with("# Report\n\n"));
        assert!(result.contains("Some analysis results here"));
        assert!(result.ends_with("---\nEnd of report"));
    }

    #[test]
    fn test_plain_text_returns_unchanged() {
        let config = OutputFormatConfig::new(OutputFormat::PlainText);
        let input = "Hello, world!";
        let result = format_output(input, &config);
        assert_eq!(result, input);
    }

    #[test]
    fn test_code_block_without_language() {
        let config = OutputFormatConfig::new(OutputFormat::CodeBlock(None));
        let input = "some code";
        let result = format_output(input, &config);
        assert!(result.starts_with("```\n"));
        assert!(result.ends_with("\n```"));
    }

    #[test]
    fn test_table_with_empty_array() {
        let config = OutputFormatConfig::new(OutputFormat::Table);
        let input = "[]";
        let result = format_output(input, &config);
        assert!(result.contains("empty"));
    }

    #[test]
    fn test_custom_template_missing_returns_content() {
        let config = OutputFormatConfig::new(OutputFormat::Custom("nonexistent".to_string()));
        let input = "fallback content";
        let result = format_output(input, &config);
        assert_eq!(result, input);
    }

    #[test]
    fn test_markdown_already_formatted() {
        let config = OutputFormatConfig::new(OutputFormat::Markdown);
        let input = "# Title\n\n- Item 1\n- Item 2";
        let result = format_output(input, &config);
        // Should return as-is since it already has markdown
        assert_eq!(result, input);
    }
}
