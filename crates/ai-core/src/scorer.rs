use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::Lazy;

use crate::error::ScorerError;

/// Async scoring abstraction for quality metrics.
#[async_trait]
pub trait Scorer: Send + Sync {
    /// Score an output for the given input and context.
    async fn score(&self, input: &str, output: &str, context: &str) -> Result<f64, ScorerError>;

    /// Stable scorer name for reporting.
    fn name(&self) -> &str;
}

/// Scores relevance based on overlap between the input and output keywords.
#[derive(Debug, Default, Clone, Copy)]
pub struct RelevanceScorer;

/// Scores faithfulness based on how much of the output is grounded in the context.
#[derive(Debug, Default, Clone, Copy)]
pub struct FaithfulnessScorer;

/// Scores output length against an ideal word count.
#[derive(Debug, Clone)]
pub struct LengthScorer {
    ideal_length: usize,
}

impl LengthScorer {
    pub fn new(ideal_length: usize) -> Result<Self, ScorerError> {
        if ideal_length == 0 {
            return Err(ScorerError::Configuration(
                "ideal_length must be at least 1".to_string(),
            ));
        }

        Ok(Self { ideal_length })
    }

    pub fn ideal_length(&self) -> usize {
        self.ideal_length
    }
}

type ScorerFuture = Pin<Box<dyn Future<Output = Result<f64, ScorerError>> + Send>>;
type JudgeFn = dyn Fn(&str, &str, &str) -> ScorerFuture + Send + Sync;

/// Mockable LLM-as-judge scorer.
#[derive(Clone)]
pub struct LlmJudgeScorer {
    name: String,
    judge: Arc<JudgeFn>,
}

impl std::fmt::Debug for LlmJudgeScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmJudgeScorer")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl LlmJudgeScorer {
    pub fn new(
        name: impl Into<String>,
        judge: impl Fn(&str, &str, &str) -> ScorerFuture + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            judge: Arc::new(judge),
        }
    }
}

/// Individual scorer output.
#[derive(Debug, Clone, PartialEq)]
pub struct ScorerResult {
    pub name: String,
    pub score: f64,
}

/// Aggregated pipeline output.
#[derive(Debug, Clone, PartialEq)]
pub struct ScorerPipelineResult {
    pub scores: Vec<ScorerResult>,
    pub average: f64,
}

/// Chains multiple scorers and computes an aggregate score.
#[derive(Default)]
pub struct ScorerPipeline {
    scorers: Vec<Box<dyn Scorer>>,
}

impl std::fmt::Debug for ScorerPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScorerPipeline")
            .field("scorer_count", &self.scorers.len())
            .finish()
    }
}

impl ScorerPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scorer<S>(mut self, scorer: S) -> Self
    where
        S: Scorer + 'static,
    {
        self.push(scorer);
        self
    }

    pub fn push<S>(&mut self, scorer: S)
    where
        S: Scorer + 'static,
    {
        self.scorers.push(Box::new(scorer));
    }

    pub fn len(&self) -> usize {
        self.scorers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scorers.is_empty()
    }

    pub async fn score(
        &self,
        input: &str,
        output: &str,
        context: &str,
    ) -> Result<ScorerPipelineResult, ScorerError> {
        if self.scorers.is_empty() {
            return Err(ScorerError::EmptyPipeline);
        }

        let mut scores = Vec::with_capacity(self.scorers.len());
        let mut total = 0.0;

        for scorer in &self.scorers {
            let score = scorer.score(input, output, context).await?;
            total += score;
            scores.push(ScorerResult {
                name: scorer.name().to_string(),
                score,
            });
        }

        let average = total / scores.len() as f64;

        Ok(ScorerPipelineResult { scores, average })
    }
}

#[async_trait]
impl Scorer for RelevanceScorer {
    async fn score(&self, input: &str, output: &str, _context: &str) -> Result<f64, ScorerError> {
        Ok(overlap_ratio(&keywords(input), &keywords(output)))
    }

    fn name(&self) -> &str {
        "relevance"
    }
}

#[async_trait]
impl Scorer for FaithfulnessScorer {
    async fn score(&self, _input: &str, output: &str, context: &str) -> Result<f64, ScorerError> {
        Ok(overlap_ratio(&keywords(output), &keywords(context)))
    }

    fn name(&self) -> &str {
        "faithfulness"
    }
}

#[async_trait]
impl Scorer for LengthScorer {
    async fn score(&self, _input: &str, output: &str, _context: &str) -> Result<f64, ScorerError> {
        let actual = word_count(output);
        let deviation = actual.abs_diff(self.ideal_length) as f64 / self.ideal_length as f64;
        Ok((1.0 - deviation).clamp(0.0, 1.0))
    }

    fn name(&self) -> &str {
        "length"
    }
}

#[async_trait]
impl Scorer for LlmJudgeScorer {
    async fn score(&self, input: &str, output: &str, context: &str) -> Result<f64, ScorerError> {
        let score = (self.judge)(input, output, context).await?;
        validate_score(&self.name, score)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn overlap_ratio(source: &HashSet<String>, target: &HashSet<String>) -> f64 {
    if source.is_empty() {
        return 0.0;
    }

    let overlap = source.intersection(target).count();
    overlap as f64 / source.len() as f64
}

fn keywords(text: &str) -> HashSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_ascii_lowercase();
            if token.len() <= 1 || is_stopword(&token) {
                None
            } else {
                Some(token)
            }
        })
        .collect()
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn validate_score(name: &str, score: f64) -> Result<f64, ScorerError> {
    if (0.0..=1.0).contains(&score) {
        Ok(score)
    } else {
        Err(ScorerError::InvalidScore {
            scorer: name.to_string(),
            score,
        })
    }
}

static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
        "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which",
        "who", "why", "with",
    ]
    .into_iter()
    .collect()
});

fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn relevance_scorer_returns_high_score_for_relevant_input_output() {
        let scorer = RelevanceScorer;
        let score = scorer
            .score(
                "Explain Rust ownership borrowing safety",
                "Rust ownership and borrowing enable memory safety.",
                "",
            )
            .await
            .unwrap();

        assert!(score >= 0.75, "expected high relevance score, got {score}");
    }

    #[tokio::test]
    async fn relevance_scorer_returns_low_score_for_irrelevant_input_output() {
        let scorer = RelevanceScorer;
        let score = scorer
            .score(
                "Explain Rust ownership borrowing safety",
                "Bananas grow in tropical climates near the equator.",
                "",
            )
            .await
            .unwrap();

        assert!(score <= 0.1, "expected low relevance score, got {score}");
    }

    #[tokio::test]
    async fn faithfulness_scorer_returns_high_score_when_output_is_grounded_in_context() {
        let scorer = FaithfulnessScorer;
        let score = scorer
            .score(
                "",
                "Rust ownership prevents data races and enables memory safety.",
                "Rust ownership prevents data races while enabling memory safety without garbage collection.",
            )
            .await
            .unwrap();

        assert!(
            score >= 0.8,
            "expected high faithfulness score, got {score}"
        );
    }

    #[tokio::test]
    async fn faithfulness_scorer_returns_low_score_when_output_contradicts_context() {
        let scorer = FaithfulnessScorer;
        let score = scorer
            .score(
                "",
                "The warranty lasts thirty days and excludes repair coverage.",
                "The warranty lasts two years and includes free repair coverage for manufacturing defects.",
            )
            .await
            .unwrap();

        assert!(score < 0.7, "expected low faithfulness score, got {score}");
    }

    #[tokio::test]
    async fn length_scorer_returns_one_at_ideal_length() {
        let scorer = LengthScorer::new(4).unwrap();
        let score = scorer
            .score("", "alpha beta gamma delta", "")
            .await
            .unwrap();

        assert_eq!(score, 1.0);
    }

    #[tokio::test]
    async fn length_scorer_returns_lower_score_for_too_short_and_too_long_outputs() {
        let scorer = LengthScorer::new(4).unwrap();
        let short_score = scorer.score("", "alpha beta", "").await.unwrap();
        let long_score = scorer
            .score("", "alpha beta gamma delta epsilon zeta eta theta", "")
            .await
            .unwrap();

        assert!(short_score < 1.0);
        assert!(long_score < 1.0);
        assert_eq!(short_score, 0.5);
        assert_eq!(long_score, 0.0);
    }

    #[tokio::test]
    async fn scorer_pipeline_aggregates_scores_correctly() {
        let pipeline = ScorerPipeline::new()
            .with_scorer(RelevanceScorer)
            .with_scorer(LengthScorer::new(6).unwrap())
            .with_scorer(LlmJudgeScorer::new("judge", |_, _, _| {
                Box::pin(async { Ok(0.6) })
            }));

        let result = pipeline
            .score(
                "Rust ownership improves memory safety",
                "Rust ownership improves memory safety greatly",
                "Rust ownership improves memory safety greatly",
            )
            .await
            .unwrap();

        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.scores[0].name, "relevance");
        assert_eq!(result.scores[1].name, "length");
        assert_eq!(result.scores[2].name, "judge");
        assert!((result.average - ((1.0 + 1.0 + 0.6) / 3.0)).abs() < 1e-10);
    }

    #[tokio::test]
    async fn llm_judge_scorer_rejects_out_of_range_scores() {
        let scorer = LlmJudgeScorer::new("judge", |_, _, _| Box::pin(async { Ok(1.2) }));
        let err = scorer.score("", "", "").await.unwrap_err();

        assert!(matches!(
            err,
            ScorerError::InvalidScore {
                scorer,
                score: 1.2
            } if scorer == "judge"
        ));
    }
}
