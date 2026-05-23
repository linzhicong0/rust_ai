//! Document ingestion pipeline.
//!
//! This module parses source documents, chunks them for retrieval, generates
//! embeddings, and upserts the resulting vectors into a [`VectorStore`].

use std::{collections::HashMap, sync::Arc};

use ai_core::{Embedder, EmbedderError};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use thiserror::Error;
use tracing::instrument;

use crate::chunker::{
    Chunker, ChunkerConfig, FixedSizeChunker, ParagraphChunker, RecursiveChunker, SentenceChunker,
};
use crate::vector_store::{VectorDocument, VectorStore, VectorStoreError};

/// Supported source document formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    /// UTF-8 plain text.
    PlainText,
    /// Markdown content.
    Markdown,
    /// HTML content.
    Html,
    /// PDF content.
    Pdf,
    /// DOCX content.
    Docx,
}

impl DocumentFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::PlainText => "plain_text",
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Pdf => "pdf",
            Self::Docx => "docx",
        }
    }
}

/// A document submitted to the ingestion pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Unique source document identifier.
    pub id: String,
    /// Raw document payload.
    pub content: String,
    /// Declared document format.
    pub format: DocumentFormat,
    /// Arbitrary document metadata propagated to chunks.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Supported chunking strategies for ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkingStrategy {
    /// Fixed-size character windows.
    FixedSize,
    /// Sentence-aware chunking.
    Sentence,
    /// Paragraph-aware chunking.
    Paragraph,
    /// Recursive multi-level chunking.
    Recursive,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Recursive
    }
}

/// Configuration for document ingestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Number of chunks to embed per batch.
    pub batch_size: usize,
    /// Chunking strategy used for parsed text.
    #[serde(default)]
    pub chunking_strategy: ChunkingStrategy,
    /// Shared configuration passed to the selected chunker.
    #[serde(default)]
    pub chunker_config: ChunkerConfig,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            chunking_strategy: ChunkingStrategy::default(),
            chunker_config: ChunkerConfig::default(),
        }
    }
}

/// Result summary returned after a document has been ingested.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestionResult {
    /// Source document identifier.
    pub document_id: String,
    /// Number of chunks indexed for the document.
    pub chunks_indexed: usize,
    /// Vector-store identifiers generated for each chunk.
    pub chunk_ids: Vec<String>,
}

/// Errors produced by document ingestion.
#[derive(Debug, Error)]
pub enum IngestionError {
    /// Returned when a document payload is not valid UTF-8.
    #[error("document payload is not valid UTF-8: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    /// Returned when a format requires an optional parser implementation.
    #[error("{format} format not supported without feature flag")]
    UnsupportedFormat { format: &'static str },
    /// Returned when a parser fails for format-specific reasons.
    #[error("failed to parse {format} document: {message}")]
    Parse {
        /// Document format that failed to parse.
        format: &'static str,
        /// Human-readable parser error.
        message: String,
    },
    /// Returned when the embedder returns an unexpected number of vectors.
    #[error("embedder `{embedder}` returned {actual} embeddings for {expected} chunk(s)")]
    EmbeddingCountMismatch {
        /// Name of the embedder implementation.
        embedder: String,
        /// Expected number of embeddings.
        expected: usize,
        /// Actual number of embeddings returned.
        actual: usize,
    },
    /// Embedding generation failure.
    #[error(transparent)]
    Embedder(#[from] EmbedderError),
    /// Vector-store failure.
    #[error(transparent)]
    VectorStore(#[from] VectorStoreError),
}

/// Parser abstraction for raw document bytes.
pub trait DocumentParser: Send + Sync {
    /// Parse raw document bytes into plain text.
    fn parse(&self, raw: &[u8]) -> Result<String, IngestionError>;
}

/// Parser for UTF-8 plain text documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct PlainTextParser;

impl DocumentParser for PlainTextParser {
    fn parse(&self, raw: &[u8]) -> Result<String, IngestionError> {
        Ok(std::str::from_utf8(raw)?.to_owned())
    }
}

/// Parser for Markdown documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct MarkdownParser;

impl DocumentParser for MarkdownParser {
    fn parse(&self, raw: &[u8]) -> Result<String, IngestionError> {
        let markdown = std::str::from_utf8(raw)?;
        Ok(strip_markdown(markdown))
    }
}

/// Parser for HTML documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct HtmlParser;

impl DocumentParser for HtmlParser {
    fn parse(&self, raw: &[u8]) -> Result<String, IngestionError> {
        let html = std::str::from_utf8(raw)?;
        Ok(strip_html(html))
    }
}

/// Stub parser for PDF documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdfParser;

impl DocumentParser for PdfParser {
    fn parse(&self, _raw: &[u8]) -> Result<String, IngestionError> {
        Err(IngestionError::UnsupportedFormat { format: "pdf" })
    }
}

/// Stub parser for DOCX documents.
#[derive(Debug, Default, Clone, Copy)]
pub struct DocxParser;

impl DocumentParser for DocxParser {
    fn parse(&self, _raw: &[u8]) -> Result<String, IngestionError> {
        Err(IngestionError::UnsupportedFormat { format: "docx" })
    }
}

/// End-to-end ingestion pipeline.
#[derive(Clone)]
pub struct IngestionPipeline {
    config: IngestionConfig,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline.
    pub fn new<E, V>(config: IngestionConfig, embedder: Arc<E>, vector_store: Arc<V>) -> Self
    where
        E: Embedder + 'static,
        V: VectorStore + 'static,
    {
        Self {
            config,
            embedder,
            vector_store,
        }
    }

    fn parser_for(format: DocumentFormat) -> &'static dyn DocumentParser {
        static PLAIN_TEXT: PlainTextParser = PlainTextParser;
        static MARKDOWN: MarkdownParser = MarkdownParser;
        static HTML: HtmlParser = HtmlParser;
        static PDF: PdfParser = PdfParser;
        static DOCX: DocxParser = DocxParser;

        match format {
            DocumentFormat::PlainText => &PLAIN_TEXT,
            DocumentFormat::Markdown => &MARKDOWN,
            DocumentFormat::Html => &HTML,
            DocumentFormat::Pdf => &PDF,
            DocumentFormat::Docx => &DOCX,
        }
    }

    fn build_chunker(&self, metadata: HashMap<String, Value>) -> Box<dyn Chunker> {
        match self.config.chunking_strategy {
            ChunkingStrategy::FixedSize => Box::new(
                FixedSizeChunker::new(self.config.chunker_config.clone()).with_metadata(metadata),
            ),
            ChunkingStrategy::Sentence => Box::new(
                SentenceChunker::new(self.config.chunker_config.clone()).with_metadata(metadata),
            ),
            ChunkingStrategy::Paragraph => Box::new(
                ParagraphChunker::new(self.config.chunker_config.clone()).with_metadata(metadata),
            ),
            ChunkingStrategy::Recursive => Box::new(
                RecursiveChunker::new(self.config.chunker_config.clone()).with_metadata(metadata),
            ),
        }
    }

    /// Parse, chunk, embed, and index a document.
    #[instrument(skip(self, document), fields(document_id = %document.id, format = document.format.as_str()))]
    pub async fn ingest(&self, document: Document) -> Result<IngestionResult, IngestionError> {
        let parser = Self::parser_for(document.format);
        let parsed = parser.parse(document.content.as_bytes())?;

        let chunker = self.build_chunker(document.metadata.clone());
        let chunks = chunker.chunk(&parsed);
        if chunks.is_empty() {
            return Ok(IngestionResult {
                document_id: document.id,
                chunks_indexed: 0,
                chunk_ids: Vec::new(),
            });
        }

        let batch_size = self.config.batch_size.max(1);
        let mut chunk_ids = Vec::with_capacity(chunks.len());

        for (batch_index, batch) in chunks.chunks(batch_size).enumerate() {
            let texts = batch
                .iter()
                .map(|chunk| chunk.content.clone())
                .collect::<Vec<_>>();
            let embeddings = self.embedder.embed(texts).await?;

            if embeddings.len() != batch.len() {
                return Err(IngestionError::EmbeddingCountMismatch {
                    embedder: self.embedder.name().to_owned(),
                    expected: batch.len(),
                    actual: embeddings.len(),
                });
            }

            let mut documents = Vec::with_capacity(batch.len());
            for (offset, (chunk, embedding)) in batch.iter().zip(embeddings.into_iter()).enumerate()
            {
                let chunk_index = batch_index * batch_size + offset;
                let chunk_id = format!("{}#chunk-{}", document.id, chunk_index);
                let mut metadata = chunk.metadata.clone();
                metadata.insert(
                    "source_document_id".to_string(),
                    Value::String(document.id.clone()),
                );
                metadata.insert(
                    "document_format".to_string(),
                    Value::String(document.format.as_str().to_string()),
                );
                metadata.insert(
                    "chunk_index".to_string(),
                    Value::Number(Number::from(chunk_index as u64)),
                );
                metadata.insert(
                    "start_offset".to_string(),
                    Value::Number(Number::from(chunk.start_offset as u64)),
                );
                metadata.insert(
                    "end_offset".to_string(),
                    Value::Number(Number::from(chunk.end_offset as u64)),
                );
                metadata.insert(
                    "embedder".to_string(),
                    Value::String(self.embedder.name().to_string()),
                );

                chunk_ids.push(chunk_id.clone());
                documents.push(VectorDocument {
                    id: chunk_id,
                    content: chunk.content.clone(),
                    embedding,
                    metadata,
                });
            }

            self.vector_store.upsert(documents).await?;
        }

        Ok(IngestionResult {
            document_id: document.id,
            chunks_indexed: chunk_ids.len(),
            chunk_ids,
        })
    }
}

fn strip_markdown(markdown: &str) -> String {
    markdown
        .lines()
        .map(strip_markdown_line)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_markdown_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut normalized = trimmed
        .trim_start_matches(|c: char| matches!(c, '#' | '>' | '-' | '*' | '+' | ' ' | '\t'))
        .trim();

    if let Some((prefix, rest)) = normalized.split_once('.') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            normalized = rest.trim();
        }
    }

    let without_links = replace_markdown_links(normalized);
    collapse_whitespace(
        &without_links
            .chars()
            .filter(|ch| !matches!(ch, '*' | '_' | '`' | '~' | '[' | ']' | '(' | ')' | '!'))
            .collect::<String>(),
    )
}

fn replace_markdown_links(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut index = 0;

    while index < chars.len() {
        if chars[index] == '!' && index + 1 < chars.len() && chars[index + 1] == '[' {
            index += 1;
            continue;
        }

        if chars[index] == '[' {
            let mut close_bracket = index + 1;
            while close_bracket < chars.len() && chars[close_bracket] != ']' {
                close_bracket += 1;
            }

            if close_bracket < chars.len()
                && close_bracket + 1 < chars.len()
                && chars[close_bracket + 1] == '('
            {
                let mut close_paren = close_bracket + 2;
                while close_paren < chars.len() && chars[close_paren] != ')' {
                    close_paren += 1;
                }

                if close_paren < chars.len() {
                    output.extend(chars[index + 1..close_bracket].iter());
                    index = close_paren + 1;
                    continue;
                }
            }
        }

        output.push(chars[index]);
        index += 1;
    }

    output
}

fn strip_html(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut inside_tag = false;

    for ch in html.chars() {
        match ch {
            '<' => {
                inside_tag = true;
                if !output.ends_with(char::is_whitespace) {
                    output.push(' ');
                }
            }
            '>' => inside_tag = false,
            _ if !inside_tag => output.push(ch),
            _ => {}
        }
    }

    collapse_whitespace(&decode_html_entities(&output))
}

fn decode_html_entities(input: &str) -> String {
    input
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;

    use crate::vector_store::{InMemoryVectorStore, VectorQuery};

    #[derive(Debug, Default)]
    struct MockEmbedder {
        calls: Mutex<Vec<Vec<String>>>,
        mismatch: bool,
    }

    impl MockEmbedder {
        fn new() -> Self {
            Self::default()
        }

        fn with_mismatch() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                mismatch: true,
            }
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            self.calls.lock().unwrap().push(texts.clone());
            if self.mismatch {
                return Ok(vec![vec![0.0, 0.0, 0.0]]);
            }

            Ok(texts
                .into_iter()
                .map(|text| {
                    vec![
                        text.chars().count() as f32,
                        text.bytes().map(u32::from).sum::<u32>() as f32,
                        text.split_whitespace().count() as f32,
                    ]
                })
                .collect())
        }

        fn name(&self) -> &str {
            "mock-embedder"
        }
    }

    #[test]
    fn plain_text_parser_reads_utf8() {
        let parsed = PlainTextParser.parse(b"hello world").unwrap();
        assert_eq!(parsed, "hello world");
    }

    #[test]
    fn markdown_parser_strips_common_markup() {
        let raw = b"# Title\n\nSome **bold** text with [link](https://example.com) and `code`.";
        let parsed = MarkdownParser.parse(raw).unwrap();
        assert_eq!(parsed, "Title\nSome bold text with link and code.");
    }

    #[test]
    fn html_parser_strips_tags_and_decodes_entities() {
        let raw = b"<h1>Title</h1><p>Hello <strong>world</strong> &amp; friends.</p>";
        let parsed = HtmlParser.parse(raw).unwrap();
        assert_eq!(parsed, "Title Hello world & friends.");
    }

    #[test]
    fn pdf_and_docx_parsers_require_feature_flags() {
        let pdf_error = PdfParser.parse(b"%PDF-1.7").unwrap_err();
        assert!(matches!(
            pdf_error,
            IngestionError::UnsupportedFormat { format: "pdf" }
        ));

        let docx_error = DocxParser.parse(b"PK").unwrap_err();
        assert!(matches!(
            docx_error,
            IngestionError::UnsupportedFormat { format: "docx" }
        ));
    }

    #[tokio::test]
    async fn ingest_chunks_embeds_and_indexes_documents() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = Arc::new(InMemoryVectorStore::new());
        let pipeline = IngestionPipeline::new(
            IngestionConfig {
                batch_size: 2,
                chunking_strategy: ChunkingStrategy::FixedSize,
                chunker_config: ChunkerConfig {
                    chunk_size: 16,
                    overlap: 0,
                    min_chunk_size: 1,
                },
            },
            embedder.clone(),
            store.clone(),
        );

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!("unit-test"));
        let document = Document {
            id: "doc-1".to_string(),
            content: "abcdefghijklmnopqrstuvwx1234567890ZYXWVUTSRQPONM".to_string(),
            format: DocumentFormat::PlainText,
            metadata,
        };

        let result = pipeline.ingest(document).await.unwrap();
        assert_eq!(result.document_id, "doc-1");
        assert_eq!(result.chunks_indexed, 3);
        assert_eq!(
            result.chunk_ids,
            vec![
                "doc-1#chunk-0".to_string(),
                "doc-1#chunk-1".to_string(),
                "doc-1#chunk-2".to_string()
            ]
        );

        let calls = embedder.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[1].len(), 1);

        let indexed = store
            .query(VectorQuery {
                embedding: vec![0.0, 0.0, 0.0],
                top_k: 10,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(indexed.len(), 3);
        assert_eq!(indexed[0].id, "doc-1#chunk-0");
        assert_eq!(indexed[1].id, "doc-1#chunk-1");
        assert_eq!(indexed[2].id, "doc-1#chunk-2");
        assert_eq!(indexed[0].metadata.get("source"), Some(&json!("unit-test")));
        assert_eq!(
            indexed[0].metadata.get("source_document_id"),
            Some(&json!("doc-1"))
        );
        assert_eq!(
            indexed[0].metadata.get("document_format"),
            Some(&json!("plain_text"))
        );
        assert_eq!(
            indexed[0].metadata.get("embedder"),
            Some(&json!("mock-embedder"))
        );
        assert_eq!(indexed[0].metadata.get("chunk_index"), Some(&json!(0)));
    }

    #[tokio::test]
    async fn ingest_returns_zero_for_empty_documents() {
        let embedder = Arc::new(MockEmbedder::new());
        let store = Arc::new(InMemoryVectorStore::new());
        let pipeline =
            IngestionPipeline::new(IngestionConfig::default(), embedder.clone(), store.clone());

        let result = pipeline
            .ingest(Document {
                id: "empty".to_string(),
                content: String::new(),
                format: DocumentFormat::PlainText,
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        assert_eq!(result.chunks_indexed, 0);
        assert!(result.chunk_ids.is_empty());
        assert!(embedder.calls().is_empty());
        let indexed = store
            .query(VectorQuery {
                embedding: vec![0.0],
                top_k: 10,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();
        assert!(indexed.is_empty());
    }

    #[tokio::test]
    async fn ingest_surfaces_embedder_count_mismatch() {
        let embedder = Arc::new(MockEmbedder::with_mismatch());
        let store = Arc::new(InMemoryVectorStore::new());
        let pipeline = IngestionPipeline::new(
            IngestionConfig {
                batch_size: 4,
                chunking_strategy: ChunkingStrategy::FixedSize,
                chunker_config: ChunkerConfig {
                    chunk_size: 4,
                    overlap: 0,
                    min_chunk_size: 1,
                },
            },
            embedder,
            store,
        );

        let error = pipeline
            .ingest(Document {
                id: "doc-2".to_string(),
                content: "abcdefghij".to_string(),
                format: DocumentFormat::PlainText,
                metadata: HashMap::new(),
            })
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            IngestionError::EmbeddingCountMismatch {
                expected: 3,
                actual: 1,
                ..
            }
        ));
    }
}
