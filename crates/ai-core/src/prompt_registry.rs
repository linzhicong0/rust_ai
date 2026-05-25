//! Prompt Registry with versioning and A/B testing (REQ-4.2).
//!
//! This module provides a centralized registry for prompts with support for
//! version history, activation, diff viewing, and A/B testing assignment.
//!
//! ## Features
//!
//! - `PromptRegistry`: `register()`, `get(version)`, `activate()`
//! - Version history with diff view capability
//! - A/B assignment by hash of (user, prompt_id)

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A single prompt version entry.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptVersion {
    /// Version number (1-indexed).
    pub version: u32,
    /// The prompt content.
    pub content: String,
    /// Optional description of this version.
    pub description: Option<String>,
}

/// A prompt entry with all its versions.
#[derive(Debug, Clone)]
struct PromptEntry {
    /// All versions of this prompt, ordered by version number.
    versions: Vec<PromptVersion>,
    /// The currently active version (index into `versions`).
    active_version: u32,
}

impl PromptEntry {
    fn new(content: String, description: Option<String>) -> Self {
        Self {
            versions: vec![PromptVersion {
                version: 1,
                content,
                description,
            }],
            active_version: 1,
        }
    }

    fn add_version(&mut self, content: String, description: Option<String>) -> u32 {
        let version = self.versions.len() as u32 + 1;
        self.versions.push(PromptVersion {
            version,
            content,
            description,
        });
        version
    }

    fn get(&self, version: Option<u32>) -> Option<&PromptVersion> {
        match version {
            Some(v) => self.versions.get((v - 1) as usize),
            None => self.versions.get((self.active_version - 1) as usize),
        }
    }

    fn activate(&mut self, version: u32) -> bool {
        if version >= 1 && version <= self.versions.len() as u32 {
            self.active_version = version;
            true
        } else {
            false
        }
    }
}

/// A/B test variant assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct AbVariant {
    /// The variant name (e.g., "A" or "B").
    pub variant: String,
    /// The prompt version assigned to this variant.
    pub version: u32,
}

/// Configuration for an A/B test on a prompt.
#[derive(Debug, Clone)]
pub struct AbTestConfig {
    /// Prompt ID being tested.
    pub prompt_id: String,
    /// Variants with their version numbers and traffic weights.
    pub variants: Vec<(String, u32, f64)>, // (variant_name, version, weight)
}

impl AbTestConfig {
    /// Create a 50/50 A/B test between two versions.
    pub fn fifty_fifty(prompt_id: impl Into<String>, version_a: u32, version_b: u32) -> Self {
        Self {
            prompt_id: prompt_id.into(),
            variants: vec![
                ("A".to_string(), version_a, 0.5),
                ("B".to_string(), version_b, 0.5),
            ],
        }
    }

    /// Assign a variant based on user ID using deterministic hashing.
    pub fn assign(&self, user_id: &str) -> AbVariant {
        let mut hasher = DefaultHasher::new();
        user_id.hash(&mut hasher);
        self.prompt_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Normalize hash to [0, 1)
        let normalized = (hash as f64) / (u64::MAX as f64);

        let mut cumulative = 0.0;
        for (variant_name, version, weight) in &self.variants {
            cumulative += weight;
            if normalized < cumulative {
                return AbVariant {
                    variant: variant_name.clone(),
                    version: *version,
                };
            }
        }

        // Fallback to last variant
        let last = self.variants.last().unwrap();
        AbVariant {
            variant: last.0.clone(),
            version: last.1,
        }
    }
}

/// A diff between two prompt versions.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptDiff {
    /// The from-version number.
    pub from_version: u32,
    /// The to-version number.
    pub to_version: u32,
    /// Human-readable diff output.
    pub diff: String,
}

/// Prompt registry with versioning and A/B testing support (REQ-4.2).
#[derive(Debug, Default)]
pub struct PromptRegistry {
    prompts: HashMap<String, PromptEntry>,
    ab_tests: HashMap<String, AbTestConfig>,
}

impl PromptRegistry {
    /// Create a new empty prompt registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new prompt or add a new version to an existing prompt.
    ///
    /// Returns the version number of the registered prompt.
    pub fn register(
        &mut self,
        prompt_id: impl Into<String>,
        content: impl Into<String>,
        description: Option<String>,
    ) -> u32 {
        let id = prompt_id.into();
        let content = content.into();

        if let Some(entry) = self.prompts.get_mut(&id) {
            entry.add_version(content, description)
        } else {
            self.prompts
                .insert(id, PromptEntry::new(content, description));
            1
        }
    }

    /// Get a specific version of a prompt.
    ///
    /// If `version` is None, returns the active version.
    pub fn get(&self, prompt_id: &str, version: Option<u32>) -> Option<&PromptVersion> {
        self.prompts.get(prompt_id).and_then(|e| e.get(version))
    }

    /// Get the prompt content for the active version.
    pub fn get_content(&self, prompt_id: &str) -> Option<&str> {
        self.get(prompt_id, None).map(|v| v.content.as_str())
    }

    /// Activate a specific version of a prompt.
    ///
    /// After activation, `get(prompt_id, None)` returns this version.
    pub fn activate(&mut self, prompt_id: &str, version: u32) -> bool {
        self.prompts
            .get_mut(prompt_id)
            .map(|e| e.activate(version))
            .unwrap_or(false)
    }

    /// Get all versions of a prompt.
    pub fn versions(&self, prompt_id: &str) -> Vec<&PromptVersion> {
        self.prompts
            .get(prompt_id)
            .map(|e| e.versions.iter().collect())
            .unwrap_or_default()
    }

    /// Compute a simple line-by-line diff between two versions.
    pub fn diff(&self, prompt_id: &str, from_version: u32, to_version: u32) -> Option<PromptDiff> {
        let entry = self.prompts.get(prompt_id)?;
        let from = entry.get(Some(from_version))?;
        let to = entry.get(Some(to_version))?;

        let diff = compute_simple_diff(&from.content, &to.content);
        Some(PromptDiff {
            from_version,
            to_version,
            diff,
        })
    }

    /// Configure an A/B test for a prompt.
    pub fn set_ab_test(&mut self, config: AbTestConfig) {
        self.ab_tests.insert(config.prompt_id.clone(), config);
    }

    /// Get the A/B variant assignment for a user and prompt.
    pub fn get_ab_variant(&self, prompt_id: &str, user_id: &str) -> Option<AbVariant> {
        self.ab_tests.get(prompt_id).map(|c| c.assign(user_id))
    }

    /// Get the prompt content for a user, respecting A/B test assignment.
    pub fn get_for_user(&self, prompt_id: &str, user_id: &str) -> Option<&str> {
        if let Some(variant) = self.get_ab_variant(prompt_id, user_id) {
            self.get(prompt_id, Some(variant.version))
                .map(|v| v.content.as_str())
        } else {
            self.get_content(prompt_id)
        }
    }

    /// List all registered prompt IDs.
    pub fn list(&self) -> Vec<String> {
        self.prompts.keys().cloned().collect()
    }
}

/// Compute a simple line-by-line diff between two strings.
fn compute_simple_diff(from: &str, to: &str) -> String {
    let from_lines: Vec<&str> = from.lines().collect();
    let to_lines: Vec<&str> = to.lines().collect();
    let mut output = String::new();

    let max_len = from_lines.len().max(to_lines.len());
    for i in 0..max_len {
        let from_line = from_lines.get(i);
        let to_line = to_lines.get(i);

        match (from_line, to_line) {
            (Some(f), Some(t)) if f == t => {
                output.push_str(&format!("  {}\n", f));
            }
            (Some(f), Some(t)) => {
                output.push_str(&format!("- {}\n", f));
                output.push_str(&format!("+ {}\n", t));
            }
            (Some(f), None) => {
                output.push_str(&format!("- {}\n", f));
            }
            (None, Some(t)) => {
                output.push_str(&format!("+ {}\n", t));
            }
            (None, None) => {}
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-4.2: Unit: register prompt v1, then v2 — get(2) returns v2 content
    #[test]
    fn test_register_and_get_versions() {
        let mut registry = PromptRegistry::new();

        let v1 = registry.register("greeting", "Hello, how can I help?", None);
        assert_eq!(v1, 1);

        let v2 = registry.register(
            "greeting",
            "Hi there! How can I assist you today?",
            Some("More friendly".to_string()),
        );
        assert_eq!(v2, 2);

        let version2 = registry.get("greeting", Some(2)).unwrap();
        assert_eq!(version2.content, "Hi there! How can I assist you today?");
        assert_eq!(version2.version, 2);
    }

    // REQ-4.2: Unit: activate(v2) makes v2 the default for get() without version
    #[test]
    fn test_activate_version() {
        let mut registry = PromptRegistry::new();

        registry.register("sys_prompt", "You are a helpful assistant.", None);
        registry.register("sys_prompt", "You are an expert Rust developer.", None);

        // Default is v1 (initial registration)
        let active = registry.get("sys_prompt", None).unwrap();
        assert_eq!(active.content, "You are a helpful assistant.");

        // Activate v2
        assert!(registry.activate("sys_prompt", 2));

        let active = registry.get("sys_prompt", None).unwrap();
        assert_eq!(active.content, "You are an expert Rust developer.");
    }

    // REQ-4.2: Unit: diff(v1, v2) returns a human-readable diff
    #[test]
    fn test_diff_between_versions() {
        let mut registry = PromptRegistry::new();

        registry.register("prompt", "Line 1\nLine 2\nLine 3", None);
        registry.register("prompt", "Line 1\nModified Line 2\nLine 3\nLine 4", None);

        let diff = registry.diff("prompt", 1, 2).unwrap();
        assert_eq!(diff.from_version, 1);
        assert_eq!(diff.to_version, 2);

        // Diff should contain changes
        assert!(diff.diff.contains("- Line 2"));
        assert!(diff.diff.contains("+ Modified Line 2"));
        assert!(diff.diff.contains("+ Line 4"));
    }

    // REQ-4.2: Unit: same user always gets the same variant for a given prompt_id
    #[test]
    fn test_consistent_ab_assignment() {
        let mut registry = PromptRegistry::new();

        registry.register("test_prompt", "Version A content", None);
        registry.register("test_prompt", "Version B content", None);

        registry.set_ab_test(AbTestConfig::fifty_fifty("test_prompt", 1, 2));

        // Same user should always get the same variant
        let variant1 = registry.get_ab_variant("test_prompt", "user_123").unwrap();
        let variant2 = registry.get_ab_variant("test_prompt", "user_123").unwrap();
        let variant3 = registry.get_ab_variant("test_prompt", "user_123").unwrap();

        assert_eq!(variant1, variant2);
        assert_eq!(variant2, variant3);
    }

    // REQ-4.2: Integration: traffic split 50/50 across two prompt variants
    #[test]
    fn test_traffic_split_approximately_even() {
        let config = AbTestConfig::fifty_fifty("test_prompt", 1, 2);

        let mut count_a = 0;
        let mut count_b = 0;

        // Test with 1000 different users
        for i in 0..1000 {
            let user_id = format!("user_{}", i);
            let variant = config.assign(&user_id);
            match variant.variant.as_str() {
                "A" => count_a += 1,
                "B" => count_b += 1,
                _ => panic!("Unexpected variant"),
            }
        }

        // With 1000 users and 50/50 split, each should get ~500
        // Allow 15% tolerance
        assert!(
            count_a > 350 && count_a < 650,
            "Expected ~500 for A, got {}",
            count_a
        );
        assert!(
            count_b > 350 && count_b < 650,
            "Expected ~500 for B, got {}",
            count_b
        );
    }

    // Test get_for_user with A/B test
    #[test]
    fn test_get_for_user_with_ab_test() {
        let mut registry = PromptRegistry::new();

        registry.register("prompt", "Content A", None);
        registry.register("prompt", "Content B", None);

        registry.set_ab_test(AbTestConfig::fifty_fifty("prompt", 1, 2));

        let content = registry.get_for_user("prompt", "user_42").unwrap();
        // Should be either "Content A" or "Content B"
        assert!(content == "Content A" || content == "Content B");
    }

    // Test version history
    #[test]
    fn test_version_history() {
        let mut registry = PromptRegistry::new();

        registry.register("prompt", "v1", None);
        registry.register("prompt", "v2", Some("Updated".to_string()));
        registry.register("prompt", "v3", Some("Final".to_string()));

        let versions = registry.versions("prompt");
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);
        assert_eq!(versions[2].version, 3);
        assert_eq!(versions[2].description, Some("Final".to_string()));
    }

    // Test non-existent prompt
    #[test]
    fn test_get_nonexistent_prompt() {
        let registry = PromptRegistry::new();
        assert!(registry.get("nonexistent", None).is_none());
        assert!(registry.get_content("nonexistent").is_none());
    }

    // Test activate invalid version
    #[test]
    fn test_activate_invalid_version() {
        let mut registry = PromptRegistry::new();
        registry.register("prompt", "content", None);

        assert!(!registry.activate("prompt", 5)); // Version 5 doesn't exist
        assert!(!registry.activate("nonexistent", 1)); // Prompt doesn't exist
    }

    // Test list
    #[test]
    fn test_list_prompts() {
        let mut registry = PromptRegistry::new();
        registry.register("prompt_a", "content a", None);
        registry.register("prompt_b", "content b", None);

        let mut list = registry.list();
        list.sort();
        assert_eq!(list, vec!["prompt_a", "prompt_b"]);
    }
}
