//! Context injection for RAG pipelines.
//!
//! This module provides automatic injection of retrieved context into prompts
//! with citation support, enabling source attribution in responses.
//!
//! ## Features
//!
//! - Template placeholder for retrieved context: `{{#context}}...{{/context}}`
//! - Source citations: [1], [2] linked to original documents
//! - Deduplication of similar retrieved chunks
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_rag::context_injection::{ContextInjector, ContextInjectorConfig, CitedContext};
//! use ai_rag::retrieval::RetrievalResult;
//! use std::collections::HashMap;
//!
//! let config = ContextInjectorConfig::default();
//! let injector = ContextInjector::new(config);
//!
//! let results = vec![
//!     RetrievalResult {
//!         id: "doc1".to_string(),
//!         content: "Rust is a systems programming language.".to_string(),
//!         score: 0.95,
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let prompt = "What is Rust? {{#context}}{{/context}}";
//! let injected = injector.inject(prompt, &results);
//! assert!(injected.text.contains("[1]"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::retrieval::RetrievalResult;

/// Configuration for context injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInjectorConfig {
    /// Maximum number of context chunks to inject.
    pub max_chunks: usize,

    /// Similarity threshold for deduplication (0.0 to 1.0).
    /// Chunks with text similarity above this threshold are considered duplicates.
    pub dedup_threshold: f64,

    /// Template for formatting each context chunk.
    /// Supports placeholders: `{content}`, `{citation}`, `{source}`, `{score}`
    pub chunk_template: String,

    /// Separator between context chunks.
    pub chunk_separator: String,

    /// Header text before the context block.
    pub context_header: String,

    /// Footer text after the context block.
    pub context_footer: String,

    /// Whether to include source citations.
    pub include_citations: bool,

    /// Whether to include relevance scores.
    pub include_scores: bool,

    /// Minimum score threshold for including a chunk.
    pub min_score: f32,
}

impl Default for ContextInjectorConfig {
    fn default() -> Self {
        Self {
            max_chunks: 5,
            dedup_threshold: 0.85,
            chunk_template: "[{citation}] {content}".to_string(),
            chunk_separator: "\n\n".to_string(),
            context_header: "--- Retrieved Context ---\n".to_string(),
            context_footer: "\n--- End Context ---".to_string(),
            include_citations: true,
            include_scores: false,
            min_score: 0.0,
        }
    }
}

/// A citation reference linking to a source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation number (1-based).
    pub number: usize,

    /// Source document ID.
    pub source_id: String,

    /// Source document content (the chunk).
    pub content: String,

    /// Relevance score.
    pub score: f32,

    /// Source metadata.
    pub metadata: HashMap<String, Value>,
}

/// Result of context injection containing the final text and citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitedContext {
    /// The prompt text with injected context.
    pub text: String,

    /// List of citations referenced in the context.
    pub citations: Vec<Citation>,

    /// Number of chunks that were deduplicated (removed).
    pub deduplicated_count: usize,
}

/// Injects retrieved context into prompts with citation support.
pub struct ContextInjector {
    config: ContextInjectorConfig,
}

impl ContextInjector {
    /// Create a new context injector with the given configuration.
    pub fn new(config: ContextInjectorConfig) -> Self {
        Self { config }
    }

    /// Inject retrieved context into a prompt template.
    ///
    /// The prompt should contain `{{#context}}{{/context}}` placeholder markers.
    /// If no markers are found, context is appended at the end.
    ///
    /// # Arguments
    ///
    /// * `prompt` — The prompt template with context placeholders
    /// * `results` — Retrieved results to inject as context
    ///
    /// # Returns
    ///
    /// A `CitedContext` containing the final prompt text and citation references.
    pub fn inject(&self, prompt: &str, results: &[RetrievalResult]) -> CitedContext {
        // Filter by minimum score
        let filtered: Vec<&RetrievalResult> = results
            .iter()
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        // Deduplicate similar chunks
        let (deduped, deduplicated_count) = self.deduplicate(&filtered);

        // Limit to max_chunks
        let chunks: Vec<&RetrievalResult> =
            deduped.into_iter().take(self.config.max_chunks).collect();

        // Build citations
        let citations: Vec<Citation> = chunks
            .iter()
            .enumerate()
            .map(|(i, result)| Citation {
                number: i + 1,
                source_id: result.id.clone(),
                content: result.content.clone(),
                score: result.score,
                metadata: result.metadata.clone(),
            })
            .collect();

        // Format context block
        let context_block = self.format_context_block(&citations);

        // Inject into prompt
        let text = if prompt.contains("{{#context}}") && prompt.contains("{{/context}}") {
            // Replace the placeholder markers and any content between them
            let start_marker = "{{#context}}";
            let end_marker = "{{/context}}";

            if let (Some(start_idx), Some(end_idx)) =
                (prompt.find(start_marker), prompt.find(end_marker))
            {
                let before = &prompt[..start_idx];
                let after = &prompt[end_idx + end_marker.len()..];
                format!("{}{}{}", before, context_block, after)
            } else {
                format!("{}\n\n{}", prompt, context_block)
            }
        } else {
            // No markers found, append at end
            format!("{}\n\n{}", prompt, context_block)
        };

        CitedContext {
            text,
            citations,
            deduplicated_count,
        }
    }

    /// Format the context block from citations.
    fn format_context_block(&self, citations: &[Citation]) -> String {
        if citations.is_empty() {
            return String::new();
        }

        let formatted_chunks: Vec<String> = citations
            .iter()
            .map(|citation| {
                let mut chunk_text = self.config.chunk_template.clone();
                chunk_text = chunk_text.replace("{citation}", &citation.number.to_string());
                chunk_text = chunk_text.replace("{content}", &citation.content);
                chunk_text = chunk_text.replace("{source}", &citation.source_id);
                chunk_text = chunk_text.replace("{score}", &format!("{:.2}", citation.score));
                chunk_text
            })
            .collect();

        format!(
            "{}{}{}",
            self.config.context_header,
            formatted_chunks.join(&self.config.chunk_separator),
            self.config.context_footer,
        )
    }

    /// Deduplicate results based on text similarity.
    ///
    /// Returns the deduplicated results and the count of removed duplicates.
    fn deduplicate<'a>(
        &self,
        results: &[&'a RetrievalResult],
    ) -> (Vec<&'a RetrievalResult>, usize) {
        let mut deduped: Vec<&'a RetrievalResult> = Vec::new();
        let mut removed = 0;

        for result in results {
            let is_duplicate = deduped.iter().any(|existing| {
                self.text_similarity(&existing.content, &result.content)
                    > self.config.dedup_threshold
            });

            if is_duplicate {
                removed += 1;
            } else {
                deduped.push(result);
            }
        }

        (deduped, removed)
    }

    /// Compute text similarity using Jaccard coefficient on word n-grams.
    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }
        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Generate a citation summary string for appending to responses.
///
/// This produces a formatted list like:
/// ```text
/// Sources:
/// [1] doc_id_1 - "First few words of content..."
/// [2] doc_id_2 - "First few words of content..."
/// ```
pub fn format_citation_summary(citations: &[Citation]) -> String {
    if citations.is_empty() {
        return String::new();
    }

    let mut summary = String::from("\nSources:\n");
    for citation in citations {
        let preview: String = citation.content.chars().take(80).collect::<String>();
        let ellipsis = if citation.content.len() > 80 {
            "..."
        } else {
            ""
        };
        summary.push_str(&format!(
            "[{}] {} - \"{}{}\"\n",
            citation.number, citation.source_id, preview, ellipsis
        ));
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, content: &str, score: f32) -> RetrievalResult {
        RetrievalResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_basic_injection_with_markers() {
        let config = ContextInjectorConfig::default();
        let injector = ContextInjector::new(config);

        let results = vec![
            make_result("doc1", "Rust is a systems programming language.", 0.95),
            make_result("doc2", "Rust focuses on safety and performance.", 0.87),
        ];

        let prompt = "What is Rust? {{#context}}{{/context}} Please answer.";
        let injected = injector.inject(prompt, &results);

        assert!(injected
            .text
            .contains("[1] Rust is a systems programming language."));
        assert!(injected
            .text
            .contains("[2] Rust focuses on safety and performance."));
        assert!(injected.text.contains("Please answer."));
        assert_eq!(injected.citations.len(), 2);
        assert_eq!(injected.citations[0].number, 1);
        assert_eq!(injected.citations[1].number, 2);
    }

    #[test]
    fn test_injection_without_markers() {
        let config = ContextInjectorConfig::default();
        let injector = ContextInjector::new(config);

        let results = vec![make_result("doc1", "Some context.", 0.9)];

        let prompt = "Tell me about Rust.";
        let injected = injector.inject(prompt, &results);

        assert!(injected.text.starts_with("Tell me about Rust."));
        assert!(injected.text.contains("[1] Some context."));
    }

    #[test]
    fn test_deduplication() {
        let config = ContextInjectorConfig {
            dedup_threshold: 0.7,
            ..Default::default()
        };
        let injector = ContextInjector::new(config);

        let results = vec![
            make_result(
                "doc1",
                "Rust is a systems programming language focused on safety.",
                0.95,
            ),
            make_result(
                "doc2",
                "Rust is a systems programming language focused on safety.",
                0.90,
            ),
            make_result("doc3", "Python is an interpreted language.", 0.85),
        ];

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &results);

        // doc2 should be deduplicated as it's identical to doc1
        assert_eq!(injected.citations.len(), 2);
        assert_eq!(injected.deduplicated_count, 1);
        assert_eq!(injected.citations[0].source_id, "doc1");
        assert_eq!(injected.citations[1].source_id, "doc3");
    }

    #[test]
    fn test_max_chunks_limit() {
        let config = ContextInjectorConfig {
            max_chunks: 2,
            ..Default::default()
        };
        let injector = ContextInjector::new(config);

        let results = vec![
            make_result("doc1", "First document.", 0.95),
            make_result("doc2", "Second document.", 0.90),
            make_result("doc3", "Third document.", 0.85),
        ];

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &results);

        assert_eq!(injected.citations.len(), 2);
    }

    #[test]
    fn test_min_score_filtering() {
        let config = ContextInjectorConfig {
            min_score: 0.5,
            ..Default::default()
        };
        let injector = ContextInjector::new(config);

        let results = vec![
            make_result("doc1", "High relevance.", 0.9),
            make_result("doc2", "Low relevance.", 0.3),
        ];

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &results);

        assert_eq!(injected.citations.len(), 1);
        assert_eq!(injected.citations[0].source_id, "doc1");
    }

    #[test]
    fn test_empty_results() {
        let config = ContextInjectorConfig::default();
        let injector = ContextInjector::new(config);

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &[]);

        assert_eq!(injected.citations.len(), 0);
        assert_eq!(injected.deduplicated_count, 0);
    }

    #[test]
    fn test_custom_template() {
        let config = ContextInjectorConfig {
            chunk_template: "Source [{citation}] (score: {score}): {content}".to_string(),
            include_scores: true,
            ..Default::default()
        };
        let injector = ContextInjector::new(config);

        let results = vec![make_result("doc1", "Test content.", 0.92)];

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &results);

        assert!(injected
            .text
            .contains("Source [1] (score: 0.92): Test content."));
    }

    #[test]
    fn test_citation_summary() {
        let citations = vec![
            Citation {
                number: 1,
                source_id: "doc_alpha".to_string(),
                content: "This is a long document that should be truncated in the summary preview."
                    .to_string(),
                score: 0.95,
                metadata: HashMap::new(),
            },
            Citation {
                number: 2,
                source_id: "doc_beta".to_string(),
                content: "Short.".to_string(),
                score: 0.80,
                metadata: HashMap::new(),
            },
        ];

        let summary = format_citation_summary(&citations);
        assert!(summary.contains("[1] doc_alpha"));
        assert!(summary.contains("[2] doc_beta"));
        assert!(summary.contains("Short."));
    }

    #[test]
    fn test_text_similarity() {
        let config = ContextInjectorConfig::default();
        let injector = ContextInjector::new(config);

        // Identical texts
        let sim = injector.text_similarity("hello world foo", "hello world foo");
        assert!((sim - 1.0).abs() < f64::EPSILON);

        // Completely different
        let sim = injector.text_similarity("hello world", "foo bar baz");
        assert!(sim < 0.01);

        // Partial overlap
        let sim = injector.text_similarity("the quick brown fox", "the quick red fox");
        assert!(sim > 0.5);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_metadata_preservation_in_citations() {
        let config = ContextInjectorConfig::default();
        let injector = ContextInjector::new(config);

        let mut metadata = HashMap::new();
        metadata.insert(
            "source_file".to_string(),
            Value::String("readme.md".to_string()),
        );
        metadata.insert(
            "page".to_string(),
            Value::Number(serde_json::Number::from(3)),
        );

        let results = vec![RetrievalResult {
            id: "doc1".to_string(),
            content: "Some content.".to_string(),
            score: 0.9,
            metadata,
        }];

        let prompt = "{{#context}}{{/context}}";
        let injected = injector.inject(prompt, &results);

        assert_eq!(
            injected.citations[0].metadata.get("source_file").unwrap(),
            &Value::String("readme.md".to_string())
        );
        assert_eq!(
            injected.citations[0].metadata.get("page").unwrap(),
            &Value::Number(serde_json::Number::from(3))
        );
    }
}
