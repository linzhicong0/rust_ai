//! Chunking strategies for RAG ingestion.
//!
//! This module provides a common [`Chunker`] trait plus several chunking
//! strategies for splitting documents into retrieval-friendly chunks while
//! preserving useful metadata such as page numbers and section headers.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Metadata attached to a chunk.
pub type ChunkMetadata = HashMap<String, Value>;

/// A chunk of text produced by a [`Chunker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Chunk content.
    pub content: String,

    /// Arbitrary metadata preserved from the source document.
    pub metadata: ChunkMetadata,

    /// Byte offset of the chunk start within the source text.
    pub start_offset: usize,

    /// Byte offset of the chunk end within the source text.
    pub end_offset: usize,
}

impl Chunk {
    /// Returns the chunk length in Unicode scalar values.
    pub fn len(&self) -> usize {
        self.content.chars().count()
    }

    /// Returns `true` when the chunk content is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Configuration shared by all chunking strategies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Target maximum chunk size measured in Unicode scalar values.
    pub chunk_size: usize,

    /// Target overlap between consecutive chunks measured in Unicode scalar values.
    pub overlap: usize,

    /// Minimum useful chunk size. Small trailing chunks are merged when possible.
    pub min_chunk_size: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 50,
            min_chunk_size: 32,
        }
    }
}

impl ChunkerConfig {
    fn normalized(&self) -> Self {
        let chunk_size = self.chunk_size.max(1);
        let overlap = self.overlap.min(chunk_size.saturating_sub(1));
        let min_chunk_size = self.min_chunk_size.min(chunk_size);

        Self {
            chunk_size,
            overlap,
            min_chunk_size,
        }
    }
}

/// Common interface for chunking strategies.
pub trait Chunker: Send + Sync {
    /// Split the provided text into chunks.
    fn chunk(&self, text: &str) -> Vec<Chunk>;
}

/// Fixed-size chunker that slices text into overlapping windows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedSizeChunker {
    /// Chunking configuration.
    pub config: ChunkerConfig,

    /// Base metadata copied into every chunk.
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

impl FixedSizeChunker {
    /// Create a fixed-size chunker.
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            metadata: HashMap::new(),
        }
    }

    /// Attach base metadata copied into each chunk.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let config = self.config.normalized();
        let total_chars = text.chars().count();

        if total_chars == 0 {
            return Vec::new();
        }

        let step = (config.chunk_size - config.overlap).max(1);
        let mut start_char = 0;
        let mut spans = Vec::new();

        while start_char < total_chars {
            let end_char = (start_char + config.chunk_size).min(total_chars);
            let start_offset = char_to_byte_offset(text, start_char);
            let end_offset = char_to_byte_offset(text, end_char);

            if let Some(span) = trim_span(text, start_offset, end_offset) {
                spans.push(span);
            }

            if end_char == total_chars {
                break;
            }

            start_char += step;
        }

        spans_to_chunks(text, spans, &config, &self.metadata)
    }
}

/// Sentence-aware chunker that prefers sentence boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceChunker {
    /// Chunking configuration.
    pub config: ChunkerConfig,

    /// Base metadata copied into every chunk.
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

impl SentenceChunker {
    /// Create a sentence-aware chunker.
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            metadata: HashMap::new(),
        }
    }

    /// Attach base metadata copied into each chunk.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let config = self.config.normalized();
        let spans = split_sentence_spans(text, 0, text.len())
            .into_iter()
            .flat_map(|span| fit_span(text, span, &config))
            .collect::<Vec<_>>();

        let aggregated = aggregate_segments(text, &spans, &config);
        spans_to_chunks(text, aggregated, &config, &self.metadata)
    }
}

/// Paragraph-aware chunker that prefers blank-line boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParagraphChunker {
    /// Chunking configuration.
    pub config: ChunkerConfig,

    /// Base metadata copied into every chunk.
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

impl ParagraphChunker {
    /// Create a paragraph-aware chunker.
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            metadata: HashMap::new(),
        }
    }

    /// Attach base metadata copied into each chunk.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let config = self.config.normalized();
        let spans = split_paragraph_spans(text, 0, text.len())
            .into_iter()
            .flat_map(|span| fit_span(text, span, &config))
            .collect::<Vec<_>>();

        let aggregated = aggregate_segments(text, &spans, &config);
        spans_to_chunks(text, aggregated, &config, &self.metadata)
    }
}

/// Heuristic semantic chunker that groups topically related sentences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticChunker {
    /// Chunking configuration.
    pub config: ChunkerConfig,

    /// Minimum adjacent sentence similarity required to keep growing a chunk.
    pub similarity_threshold: f32,

    /// Base metadata copied into every chunk.
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

impl SemanticChunker {
    /// Create a semantic chunker.
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            similarity_threshold: 0.15,
            metadata: HashMap::new(),
        }
    }

    /// Override the semantic similarity threshold.
    pub fn with_similarity_threshold(mut self, similarity_threshold: f32) -> Self {
        self.similarity_threshold = similarity_threshold.clamp(0.0, 1.0);
        self
    }

    /// Attach base metadata copied into each chunk.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let config = self.config.normalized();
        let sentences = split_sentence_spans(text, 0, text.len())
            .into_iter()
            .flat_map(|span| fit_span(text, span, &config))
            .collect::<Vec<_>>();

        if sentences.is_empty() {
            return Vec::new();
        }

        let mut spans = Vec::new();
        let mut start_index = 0;

        while start_index < sentences.len() {
            let mut end_index = start_index;
            let mut previous_tokens = sentence_tokens(text, sentences[end_index]);

            while end_index + 1 < sentences.len() {
                let next_index = end_index + 1;
                let next_tokens = sentence_tokens(text, sentences[next_index]);
                let candidate_len = char_len(
                    text,
                    sentences[start_index].start,
                    sentences[next_index].end,
                );
                let current_len =
                    char_len(text, sentences[start_index].start, sentences[end_index].end);
                let similarity = jaccard_similarity(&previous_tokens, &next_tokens);
                let semantic_break =
                    similarity < self.similarity_threshold && current_len >= config.min_chunk_size;

                if candidate_len > config.chunk_size || semantic_break {
                    break;
                }

                end_index = next_index;
                previous_tokens = next_tokens;
            }

            spans.push(Span {
                start: sentences[start_index].start,
                end: sentences[end_index].end,
            });

            if end_index + 1 >= sentences.len() {
                break;
            }

            start_index =
                overlap_start_index(text, &sentences, start_index, end_index, config.overlap);
        }

        spans_to_chunks(text, spans, &config, &self.metadata)
    }
}

/// Recursive chunker that progressively falls back from larger to smaller units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecursiveChunker {
    /// Chunking configuration.
    pub config: ChunkerConfig,

    /// Base metadata copied into every chunk.
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

impl RecursiveChunker {
    /// Create a recursive chunker.
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            metadata: HashMap::new(),
        }
    }

    /// Attach base metadata copied into each chunk.
    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let config = self.config.normalized();
        let spans = recursive_atomic_spans(
            text,
            Span {
                start: 0,
                end: text.len(),
            },
            &config,
        );
        let aggregated = aggregate_segments(text, &spans, &config);
        spans_to_chunks(text, aggregated, &config, &self.metadata)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
}

fn spans_to_chunks(
    text: &str,
    mut spans: Vec<Span>,
    config: &ChunkerConfig,
    base_metadata: &ChunkMetadata,
) -> Vec<Chunk> {
    spans.retain(|span| span.start < span.end);

    if spans.len() > 1 {
        let last = *spans.last().expect("last span exists");
        if char_len(text, last.start, last.end) < config.min_chunk_size {
            let previous = spans.len() - 2;
            spans[previous].end = last.end;
            spans.pop();
        }
    }

    spans
        .into_iter()
        .map(|span| Chunk {
            content: text[span.start..span.end].to_string(),
            metadata: collect_metadata(text, span.start, span.end, base_metadata),
            start_offset: span.start,
            end_offset: span.end,
        })
        .collect()
}

fn aggregate_segments(text: &str, segments: &[Span], config: &ChunkerConfig) -> Vec<Span> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut start_index = 0;

    while start_index < segments.len() {
        let mut end_index = start_index;

        while end_index + 1 < segments.len() {
            let candidate_end = segments[end_index + 1].end;
            if char_len(text, segments[start_index].start, candidate_end) > config.chunk_size {
                break;
            }
            end_index += 1;
        }

        spans.push(Span {
            start: segments[start_index].start,
            end: segments[end_index].end,
        });

        if end_index + 1 >= segments.len() {
            break;
        }

        start_index = overlap_start_index(text, segments, start_index, end_index, config.overlap);
    }

    spans
}

fn overlap_start_index(
    text: &str,
    segments: &[Span],
    start_index: usize,
    end_index: usize,
    overlap: usize,
) -> usize {
    if overlap == 0 || start_index == end_index {
        return end_index + 1;
    }

    let mut next_index = end_index;

    for candidate in (start_index + 1..=end_index).rev() {
        if char_len(text, segments[candidate].start, segments[end_index].end) >= overlap {
            next_index = candidate;
        }
    }

    next_index.max(start_index + 1)
}

fn fit_span(text: &str, span: Span, config: &ChunkerConfig) -> Vec<Span> {
    if char_len(text, span.start, span.end) <= config.chunk_size {
        return vec![span];
    }

    recursive_atomic_spans(text, span, config)
}

fn recursive_atomic_spans(text: &str, span: Span, config: &ChunkerConfig) -> Vec<Span> {
    let span = match trim_span(text, span.start, span.end) {
        Some(span) => span,
        None => return Vec::new(),
    };

    if char_len(text, span.start, span.end) <= config.chunk_size {
        return vec![span];
    }

    for parts in [
        split_paragraph_spans(text, span.start, span.end),
        split_sentence_spans(text, span.start, span.end),
        split_line_spans(text, span.start, span.end),
        split_word_spans(text, span.start, span.end),
    ] {
        if parts.len() > 1 {
            return parts
                .into_iter()
                .flat_map(|part| recursive_atomic_spans(text, part, config))
                .collect();
        }
    }

    split_fixed_span(text, span, config.chunk_size)
}

fn split_fixed_span(text: &str, span: Span, chunk_size: usize) -> Vec<Span> {
    let slice = &text[span.start..span.end];
    let total_chars = slice.chars().count();
    let mut spans = Vec::new();
    let mut start_char = 0;

    while start_char < total_chars {
        let end_char = (start_char + chunk_size).min(total_chars);
        let start_offset = span.start + char_to_byte_offset(slice, start_char);
        let end_offset = span.start + char_to_byte_offset(slice, end_char);

        if let Some(trimmed) = trim_span(text, start_offset, end_offset) {
            spans.push(trimmed);
        }

        start_char = end_char;
    }

    spans
}

fn split_sentence_spans(text: &str, start: usize, end: usize) -> Vec<Span> {
    if start >= end {
        return Vec::new();
    }

    let slice = &text[start..end];
    let mut spans = Vec::new();
    let mut sentence_start = 0;

    for (index, character) in slice.char_indices() {
        if !matches!(character, '.' | '!' | '?') {
            continue;
        }

        let boundary = index + character.len_utf8();
        let remainder = &slice[boundary..];
        let next_char = remainder.chars().next();

        if next_char.is_none() || next_char.is_some_and(char::is_whitespace) {
            if let Some(span) = trim_span(text, start + sentence_start, start + boundary) {
                spans.push(span);
            }
            sentence_start = boundary;
        }
    }

    if let Some(span) = trim_span(text, start + sentence_start, end) {
        spans.push(span);
    }

    if spans.is_empty() {
        trim_span(text, start, end).into_iter().collect()
    } else {
        spans
    }
}

fn split_paragraph_spans(text: &str, start: usize, end: usize) -> Vec<Span> {
    split_on_blank_lines(text, start, end)
}

fn split_line_spans(text: &str, start: usize, end: usize) -> Vec<Span> {
    if start >= end {
        return Vec::new();
    }

    let slice = &text[start..end];
    let mut spans = Vec::new();
    let mut line_start = 0;

    for (index, character) in slice.char_indices() {
        if character == '\n' {
            if let Some(span) = trim_span(text, start + line_start, start + index) {
                spans.push(span);
            }
            line_start = index + 1;
        }
    }

    if let Some(span) = trim_span(text, start + line_start, end) {
        spans.push(span);
    }

    if spans.is_empty() {
        trim_span(text, start, end).into_iter().collect()
    } else {
        spans
    }
}

fn split_word_spans(text: &str, start: usize, end: usize) -> Vec<Span> {
    if start >= end {
        return Vec::new();
    }

    let slice = &text[start..end];
    let mut spans = Vec::new();
    let mut word_start = None;

    for (index, character) in slice.char_indices() {
        if character.is_whitespace() {
            if let Some(begin) = word_start.take() {
                spans.push(Span {
                    start: start + begin,
                    end: start + index,
                });
            }
        } else if word_start.is_none() {
            word_start = Some(index);
        }
    }

    if let Some(begin) = word_start {
        spans.push(Span {
            start: start + begin,
            end,
        });
    }

    spans
}

fn split_on_blank_lines(text: &str, start: usize, end: usize) -> Vec<Span> {
    if start >= end {
        return Vec::new();
    }

    let slice = &text[start..end];
    let bytes = slice.as_bytes();
    let mut spans = Vec::new();
    let mut segment_start = 0;
    let mut index = 0;

    while index + 1 < bytes.len() {
        if bytes[index] == b'\n' && bytes[index + 1] == b'\n' {
            if let Some(span) = trim_span(text, start + segment_start, start + index) {
                spans.push(span);
            }

            index += 2;
            while index < bytes.len() && matches!(bytes[index], b'\n' | b'\r' | b' ' | b'\t') {
                index += 1;
            }
            segment_start = index;
            continue;
        }

        index += 1;
    }

    if let Some(span) = trim_span(text, start + segment_start, end) {
        spans.push(span);
    }

    if spans.is_empty() {
        trim_span(text, start, end).into_iter().collect()
    } else {
        spans
    }
}

fn trim_span(text: &str, start: usize, end: usize) -> Option<Span> {
    if start >= end {
        return None;
    }

    let slice = &text[start..end];
    let trimmed_start = slice.len() - slice.trim_start().len();
    let trimmed_end = slice.trim_end().len();

    if trimmed_start >= trimmed_end {
        return None;
    }

    Some(Span {
        start: start + trimmed_start,
        end: start + trimmed_end,
    })
}

fn char_to_byte_offset(text: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }

    text.char_indices()
        .nth(char_index)
        .map(|(offset, _)| offset)
        .unwrap_or(text.len())
}

fn char_len(text: &str, start: usize, end: usize) -> usize {
    text[start..end].chars().count()
}

fn collect_metadata(
    text: &str,
    start_offset: usize,
    end_offset: usize,
    base_metadata: &ChunkMetadata,
) -> ChunkMetadata {
    let mut metadata = base_metadata.clone();
    let (page_number, section_header) = infer_context_metadata(text, start_offset, end_offset);

    if !metadata.contains_key("page_number") {
        if let Some(page_number) = page_number {
            metadata.insert("page_number".to_string(), Value::from(page_number));
        }
    }

    if !metadata.contains_key("section_header") {
        if let Some(section_header) = section_header {
            metadata.insert("section_header".to_string(), Value::String(section_header));
        }
    }

    metadata
}

fn infer_context_metadata(
    text: &str,
    start_offset: usize,
    end_offset: usize,
) -> (Option<u64>, Option<String>) {
    let mut page_number = if text[..start_offset].contains('\u{000C}') || start_offset == 0 {
        Some(text[..start_offset].matches('\u{000C}').count() as u64 + 1)
    } else {
        None
    };
    let mut section_header = None;
    let mut offset = 0;
    let scan_limit = end_offset.min(text.len());

    for line in text.split_inclusive('\n') {
        if offset > scan_limit {
            break;
        }

        let trimmed = line.trim();
        if let Some(parsed_page) = parse_page_number(trimmed) {
            page_number = Some(parsed_page);
        }
        if let Some(header) = parse_section_header(trimmed) {
            section_header = Some(header);
        }

        offset += line.len();
    }

    (page_number, section_header)
}

fn parse_page_number(line: &str) -> Option<u64> {
    let normalized = line
        .trim_matches(|character| matches!(character, '[' | ']'))
        .trim();
    let lower = normalized.to_ascii_lowercase();

    let remainder = lower
        .strip_prefix("page")
        .map(|value| {
            value
                .trim_start_matches(|character: char| character == ':' || character.is_whitespace())
        })
        .unwrap_or("")
        .trim();

    if remainder.is_empty() {
        return None;
    }

    remainder
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn parse_section_header(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let header = trimmed.trim_start_matches('#').trim();

    if trimmed.starts_with('#') && !header.is_empty() {
        Some(header.to_string())
    } else {
        None
    }
}

fn sentence_tokens(text: &str, span: Span) -> HashSet<String> {
    text[span.start..span.end]
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }

    let intersection = left.intersection(right).count() as f32;
    let union = left.union(right).count() as f32;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_chunk_len_and_empty() {
        let chunk = Chunk {
            content: "hello".to_string(),
            metadata: HashMap::new(),
            start_offset: 0,
            end_offset: 5,
        };

        assert_eq!(chunk.len(), 5);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_fixed_size_chunker_respects_size_overlap_and_offsets() {
        let chunker = FixedSizeChunker::new(ChunkerConfig {
            chunk_size: 10,
            overlap: 3,
            min_chunk_size: 1,
        });

        let chunks = chunker.chunk("abcdefghijklmnopqrstuvwxyz");

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].content, "abcdefghij");
        assert_eq!(chunks[1].content, "hijklmnopq");
        assert_eq!(chunks[2].content, "opqrstuvwx");
        assert_eq!(chunks[3].content, "vwxyz");
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks[0].end_offset, 10);
        assert_eq!(chunks[1].start_offset, 7);
        assert_eq!(chunks[1].end_offset, 17);
    }

    #[test]
    fn test_fixed_size_chunker_merges_small_trailing_chunk() {
        let chunker = FixedSizeChunker::new(ChunkerConfig {
            chunk_size: 10,
            overlap: 0,
            min_chunk_size: 5,
        });

        let chunks = chunker.chunk("abcdefghijkl");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "abcdefghijkl");
    }

    #[test]
    fn test_sentence_chunker_splits_on_sentence_boundaries() {
        let chunker = SentenceChunker::new(ChunkerConfig {
            chunk_size: 20,
            overlap: 0,
            min_chunk_size: 1,
        });

        let chunks = chunker.chunk("First sentence. Second one! Third item?");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "First sentence.");
        assert_eq!(chunks[1].content, "Second one!");
        assert_eq!(chunks[2].content, "Third item?");
    }

    #[test]
    fn test_paragraph_chunker_splits_on_blank_lines() {
        let chunker = ParagraphChunker::new(ChunkerConfig {
            chunk_size: 18,
            overlap: 0,
            min_chunk_size: 1,
        });

        let text = "alpha beta\n\ncharlie delta\n\necho foxtrot";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "alpha beta");
        assert_eq!(chunks[1].content, "charlie delta");
        assert_eq!(chunks[2].content, "echo foxtrot");
    }

    #[test]
    fn test_recursive_chunker_splits_large_text_and_applies_overlap() {
        let chunker = RecursiveChunker::new(ChunkerConfig {
            chunk_size: 24,
            overlap: 6,
            min_chunk_size: 5,
        });

        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|chunk| chunk.len() <= 24));
        assert!(chunks[1].start_offset < chunks[0].end_offset);
    }

    #[test]
    fn test_semantic_chunker_breaks_on_topic_shift() {
        let chunker = SemanticChunker::new(ChunkerConfig {
            chunk_size: 64,
            overlap: 0,
            min_chunk_size: 8,
        })
        .with_similarity_threshold(0.15);

        let text = "Cats purr softly. Cats chase yarn. Quantum fields oscillate. Quantum particles carry energy.";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Cats purr softly."));
        assert!(chunks[0].content.contains("Cats chase yarn."));
        assert!(chunks[1].content.contains("Quantum fields oscillate."));
        assert!(chunks[1]
            .content
            .contains("Quantum particles carry energy."));
    }

    #[test]
    fn test_metadata_preservation_and_inference() {
        let mut metadata = HashMap::new();
        metadata.insert("source_document".to_string(), json!("guide.md"));

        let chunker = FixedSizeChunker::new(ChunkerConfig {
            chunk_size: 64,
            overlap: 0,
            min_chunk_size: 1,
        })
        .with_metadata(metadata);

        let text = "# Introduction\nPage 2\nThis is a document section.";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].metadata.get("source_document"),
            Some(&json!("guide.md"))
        );
        assert_eq!(
            chunks[0].metadata.get("section_header"),
            Some(&json!("Introduction"))
        );
        assert_eq!(chunks[0].metadata.get("page_number"), Some(&json!(2)));
    }

    #[test]
    fn test_base_metadata_takes_precedence_over_inferred_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("page_number".to_string(), json!(9));
        metadata.insert("section_header".to_string(), json!("Override"));

        let chunker = FixedSizeChunker::new(ChunkerConfig::default()).with_metadata(metadata);
        let chunks = chunker.chunk("# Intro\nPage 2\nBody text.");

        assert_eq!(chunks[0].metadata.get("page_number"), Some(&json!(9)));
        assert_eq!(
            chunks[0].metadata.get("section_header"),
            Some(&json!("Override"))
        );
    }

    #[test]
    fn test_empty_input_returns_no_chunks() {
        let chunker = RecursiveChunker::new(ChunkerConfig::default());
        assert!(chunker.chunk("").is_empty());
    }

    #[test]
    fn test_unicode_safe_fixed_size_chunking() {
        let chunker = FixedSizeChunker::new(ChunkerConfig {
            chunk_size: 3,
            overlap: 1,
            min_chunk_size: 1,
        });

        let chunks = chunker.chunk("你好世界");

        assert_eq!(chunks[0].content, "你好世");
        assert_eq!(chunks[1].content, "世界");
    }
}
