//! Configurable content filtering for LLM inputs and outputs.
//!
//! Supports category-based filtering (violence, hate speech, sexual content,
//! self-harm) with configurable severity thresholds and custom regex rules.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::content_filter::{ContentFilter, FilterCategory, FilterSeverity, FilterAction};
//!
//! // Build a filter that blocks high-severity violence and warns on hate.
//! let filter = ContentFilter::new()
//!     .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block)
//!     .with_rule(FilterCategory::Hate, FilterSeverity::Medium, FilterAction::Warn);
//!
//! // check_input returns an action for the given text.
//! let action = filter.check("I will hurt you");
//! // The exact outcome depends on your configured rules and keyword matching.
//! let _ = action;
//! ```

use regex::Regex;

// ── Category ──────────────────────────────────────────────────────────────────

/// Content category for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterCategory {
    /// Violence or physical harm content.
    Violence,
    /// Hate speech or discrimination.
    Hate,
    /// Sexual or explicit content.
    Sexual,
    /// Self-harm or suicide content.
    SelfHarm,
    /// A custom, user-defined category.
    Custom,
}

impl FilterCategory {
    /// Return the human-readable name of this category.
    pub fn name(&self) -> &'static str {
        match self {
            FilterCategory::Violence => "violence",
            FilterCategory::Hate => "hate",
            FilterCategory::Sexual => "sexual",
            FilterCategory::SelfHarm => "self_harm",
            FilterCategory::Custom => "custom",
        }
    }
}

// ── Severity ──────────────────────────────────────────────────────────────────

/// Severity level for a content filter match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FilterSeverity {
    /// Low severity — unlikely to require action for most applications.
    Low,
    /// Medium severity — may require a warning.
    Medium,
    /// High severity — typically requires blocking.
    High,
    /// Critical severity — always block.
    Critical,
}

// ── Action ────────────────────────────────────────────────────────────────────

/// Action to take when a filter rule fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterAction {
    /// Allow the content through with no modification.
    Allow,
    /// Allow but attach a warning annotation.
    Warn(String),
    /// Block the content with the given reason.
    Block(String),
}

// ── Filter rule ───────────────────────────────────────────────────────────────

/// A single configured rule: if `category` at `min_severity` or above is detected,
/// take `action`.
#[derive(Debug, Clone)]
struct FilterRule {
    category: FilterCategory,
    min_severity: FilterSeverity,
    action: FilterAction,
}

// ── Built-in keyword sets ─────────────────────────────────────────────────────

fn violence_keywords_high() -> &'static [&'static str] {
    &[
        "kill", "murder", "torture", "assault", "stab", "shoot", "bomb", "explode",
        "massacre", "slaughter",
    ]
}

fn violence_keywords_medium() -> &'static [&'static str] {
    &["fight", "hurt", "harm", "attack", "punch", "beat", "hit", "wound"]
}

fn hate_keywords_high() -> &'static [&'static str] {
    &[
        "subhuman", "vermin", "parasite",
        "exterminate",
        "inferior race",
    ]
}

fn hate_keywords_medium() -> &'static [&'static str] {
    &["hate", "despise", "bigot", "racist", "slur"]
}

fn sexual_keywords_high() -> &'static [&'static str] {
    &["explicit sexual", "pornographic", "obscene"]
}

fn sexual_keywords_medium() -> &'static [&'static str] {
    &["explicit", "sexual content", "nude", "adult content"]
}

fn self_harm_keywords_high() -> &'static [&'static str] {
    &[
        "suicide method",
        "how to kill myself",
        "how to end my life",
        "overdose on pills",
        "slit wrist",
    ]
}

fn self_harm_keywords_medium() -> &'static [&'static str] {
    &["self harm", "self-harm", "cutting myself", "hurt myself", "want to die"]
}

// ── Custom regex rule ──────────────────────────────────────────────────────────

/// A user-supplied regex pattern rule.
struct RegexRule {
    pattern: Regex,
    category: FilterCategory,
    severity: FilterSeverity,
    description: String,
}

// ── Match result ──────────────────────────────────────────────────────────────

/// The result of running `ContentFilter::check`.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterResult {
    /// The action determined by the filter.
    pub action: FilterAction,
    /// Category that triggered the filter (if any).
    pub category: Option<FilterCategory>,
    /// Severity of the detected content (if any).
    pub severity: Option<FilterSeverity>,
    /// Human-readable reason for the action.
    pub reason: Option<String>,
}

impl FilterResult {
    fn allow() -> Self {
        Self {
            action: FilterAction::Allow,
            category: None,
            severity: None,
            reason: None,
        }
    }

    fn from_action(
        action: FilterAction,
        category: FilterCategory,
        severity: FilterSeverity,
    ) -> Self {
        let reason = Some(format!(
            "{} content detected at {:?} severity",
            category.name(),
            severity
        ));
        Self {
            action,
            category: Some(category),
            severity: Some(severity),
            reason,
        }
    }
}

// ── ContentFilter ─────────────────────────────────────────────────────────────

/// Configurable content filter for LLM inputs and outputs.
///
/// Rules are evaluated in registration order; the first matching rule determines
/// the action for a given category. If no rule matches, the content is allowed.
#[derive(Default)]
pub struct ContentFilter {
    rules: Vec<FilterRule>,
    regex_rules: Vec<RegexRule>,
}

impl ContentFilter {
    /// Create a filter with no rules (all content is allowed by default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a category-severity-action rule.
    ///
    /// The rule fires when content in `category` is detected with severity ≥
    /// `min_severity`.
    pub fn with_rule(
        mut self,
        category: FilterCategory,
        min_severity: FilterSeverity,
        action: FilterAction,
    ) -> Self {
        self.rules.push(FilterRule {
            category,
            min_severity,
            action,
        });
        self
    }

    /// Add a custom regex rule.
    ///
    /// If `pattern` matches the input, it is treated as `category` at `severity`.
    /// The applicable action is determined by the existing category rules.
    /// If no matching rule exists, the content is allowed.
    pub fn with_regex_rule(
        mut self,
        pattern: &str,
        category: FilterCategory,
        severity: FilterSeverity,
        description: impl Into<String>,
    ) -> Self {
        match Regex::new(pattern) {
            Ok(re) => {
                self.regex_rules.push(RegexRule {
                    pattern: re,
                    category,
                    severity,
                    description: description.into(),
                });
            }
            Err(e) => {
                eprintln!("ContentFilter: invalid regex '{}': {}", pattern, e);
            }
        }
        self
    }

    /// Check text and return the filter result.
    pub fn check(&self, text: &str) -> FilterResult {
        let lower = text.to_lowercase();

        // Check built-in category keywords first.
        let categories: &[(FilterCategory, &[&str], FilterSeverity)] = &[
            (FilterCategory::Violence, violence_keywords_high(), FilterSeverity::High),
            (FilterCategory::Violence, violence_keywords_medium(), FilterSeverity::Medium),
            (FilterCategory::Hate, hate_keywords_high(), FilterSeverity::High),
            (FilterCategory::Hate, hate_keywords_medium(), FilterSeverity::Medium),
            (FilterCategory::Sexual, sexual_keywords_high(), FilterSeverity::High),
            (FilterCategory::Sexual, sexual_keywords_medium(), FilterSeverity::Medium),
            (FilterCategory::SelfHarm, self_harm_keywords_high(), FilterSeverity::High),
            (FilterCategory::SelfHarm, self_harm_keywords_medium(), FilterSeverity::Medium),
        ];

        for (category, keywords, severity) in categories {
            if keywords.iter().any(|kw| lower.contains(kw)) {
                if let Some(action) = self.action_for(*category, *severity) {
                    return FilterResult::from_action(action, *category, *severity);
                }
            }
        }

        // Check user-supplied regex rules.
        for regex_rule in &self.regex_rules {
            if regex_rule.pattern.is_match(text) {
                if let Some(action) = self.action_for(regex_rule.category, regex_rule.severity) {
                    let mut result =
                        FilterResult::from_action(action, regex_rule.category, regex_rule.severity);
                    result.reason = Some(regex_rule.description.clone());
                    return result;
                }
            }
        }

        FilterResult::allow()
    }

    /// Find the first rule that applies to `category` at `severity`.
    fn action_for(&self, category: FilterCategory, severity: FilterSeverity) -> Option<FilterAction> {
        self.rules
            .iter()
            .find(|r| r.category == category && severity >= r.min_severity)
            .map(|r| r.action.clone())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-13.2: violence content is blocked when rule is configured
    #[test]
    fn test_violence_keyword_triggers_block() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block("Violence detected".to_string()));

        let result = filter.check("I will kill everyone here");
        assert_eq!(
            result.action,
            FilterAction::Block("Violence detected".to_string())
        );
        assert_eq!(result.category, Some(FilterCategory::Violence));
        assert_eq!(result.severity, Some(FilterSeverity::High));
    }

    // REQ-13.2: hate speech at high severity is blocked
    #[test]
    fn test_hate_speech_high_severity_blocked() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Hate, FilterSeverity::High, FilterAction::Block("Hate detected".to_string()));

        let result = filter.check("they are subhuman vermin");
        assert!(matches!(result.action, FilterAction::Block(_)));
        assert_eq!(result.category, Some(FilterCategory::Hate));
    }

    // REQ-13.2: self-harm content triggers block
    #[test]
    fn test_self_harm_content_blocked() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::SelfHarm, FilterSeverity::Medium, FilterAction::Block("Self-harm detected".to_string()));

        let result = filter.check("I want to hurt myself and engage in self harm");
        assert!(matches!(result.action, FilterAction::Block(_)));
        assert_eq!(result.category, Some(FilterCategory::SelfHarm));
    }

    // REQ-13.2: medium severity triggers warn when configured
    #[test]
    fn test_medium_severity_triggers_warn() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::Medium, FilterAction::Warn("Violence warning".to_string()));

        let result = filter.check("they got into a fight last night");
        assert!(matches!(result.action, FilterAction::Warn(_)));
        assert_eq!(result.category, Some(FilterCategory::Violence));
    }

    // REQ-13.2: benign content is allowed when no rule matches
    #[test]
    fn test_benign_content_is_allowed() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block("Violence".to_string()));

        let result = filter.check("The weather is nice today.");
        assert_eq!(result.action, FilterAction::Allow);
        assert!(result.category.is_none());
    }

    // REQ-13.2: content below threshold is allowed
    #[test]
    fn test_low_severity_below_threshold_is_allowed() {
        // Only block HIGH severity; medium keyword should pass through
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block("Violence".to_string()));

        // "fight" is a medium-severity keyword; HIGH threshold should allow it
        let result = filter.check("they got into a fight");
        assert_eq!(result.action, FilterAction::Allow);
    }

    // REQ-13.2: custom regex rule fires correctly
    #[test]
    fn test_custom_regex_rule_triggers_block() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Custom, FilterSeverity::Medium, FilterAction::Block("Custom rule matched".to_string()))
            .with_regex_rule(r"\bforbidden_word\b", FilterCategory::Custom, FilterSeverity::Medium, "custom policy violation");

        let result = filter.check("This message contains forbidden_word inside it.");
        assert!(matches!(result.action, FilterAction::Block(_)));
        assert_eq!(result.category, Some(FilterCategory::Custom));
        assert_eq!(result.reason, Some("custom policy violation".to_string()));
    }

    // REQ-13.2: custom regex rule with no matching category rule allows
    #[test]
    fn test_custom_regex_rule_with_no_category_rule_allows() {
        let filter = ContentFilter::new()
            // No rule for FilterCategory::Custom is registered
            .with_regex_rule(r"\bforbidden_word\b", FilterCategory::Custom, FilterSeverity::High, "custom");

        let result = filter.check("This contains forbidden_word.");
        assert_eq!(result.action, FilterAction::Allow);
    }

    // REQ-13.2: sexual content at high severity blocked
    #[test]
    fn test_sexual_content_high_severity_blocked() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Sexual, FilterSeverity::High, FilterAction::Block("Sexual content".to_string()));

        let result = filter.check("This is explicit sexual content");
        assert!(matches!(result.action, FilterAction::Block(_)));
        assert_eq!(result.category, Some(FilterCategory::Sexual));
    }

    // REQ-13.2: filter result contains reason
    #[test]
    fn test_filter_result_contains_reason() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block("Violence".to_string()));

        let result = filter.check("I will murder everyone");
        assert!(result.reason.is_some());
        assert!(result.reason.unwrap().contains("violence"));
    }

    // REQ-13.2: multiple rules first match wins
    #[test]
    fn test_first_matching_rule_wins() {
        let filter = ContentFilter::new()
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Block("First rule".to_string()))
            .with_rule(FilterCategory::Violence, FilterSeverity::High, FilterAction::Warn("Second rule".to_string()));

        let result = filter.check("I will kill you");
        // First rule should win
        assert_eq!(
            result.action,
            FilterAction::Block("First rule".to_string())
        );
    }
}
