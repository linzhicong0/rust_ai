// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Audit Trail (REQ-13.5)
//!
//! Immutable, append-only audit trail of all agent actions, tool calls, and data access.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::audit_trail::{AuditTrail, AuditEntry, AuditAction};
//!
//! let mut trail = AuditTrail::new();
//! trail.log(AuditEntry::new("agent-1", AuditAction::ToolCall {
//!     tool_name: "web_search".into(),
//!     input_hash: "abc123".into(),
//!     output_hash: "def456".into(),
//! }));
//!
//! assert_eq!(trail.len(), 1);
//! let entries = trail.query_by_agent("agent-1");
//! assert_eq!(entries.len(), 1);
//! ```

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ── AuditAction ───────────────────────────────────────────────────────────────

/// The type of action being audited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    /// An agent invoked a tool.
    ToolCall {
        tool_name: String,
        input_hash: String,
        output_hash: String,
    },
    /// An agent accessed data.
    DataAccess { resource: String, operation: String },
    /// An agent produced a response.
    AgentResponse { model: String, token_count: u32 },
    /// A model was invoked.
    ModelInvocation { model: String, provider: String },
    /// A configuration was changed.
    ConfigChange {
        key: String,
        old_value: String,
        new_value: String,
    },
    /// A custom action.
    Custom {
        action_type: String,
        details: String,
    },
}

impl AuditAction {
    /// Return the action type as a string.
    pub fn action_type(&self) -> &str {
        match self {
            AuditAction::ToolCall { .. } => "tool_call",
            AuditAction::DataAccess { .. } => "data_access",
            AuditAction::AgentResponse { .. } => "agent_response",
            AuditAction::ModelInvocation { .. } => "model_invocation",
            AuditAction::ConfigChange { .. } => "config_change",
            AuditAction::Custom { action_type, .. } => action_type.as_str(),
        }
    }
}

// ── AuditEntry ────────────────────────────────────────────────────────────────

/// A single entry in the audit trail.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Unique entry identifier.
    pub id: String,
    /// Timestamp of the action.
    pub timestamp: SystemTime,
    /// The agent or actor that performed the action.
    pub agent_id: String,
    /// The action that was performed.
    pub action: AuditAction,
    /// Optional correlation ID to group related entries.
    pub correlation_id: Option<String>,
    /// Optional metadata.
    pub metadata: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry with the current timestamp.
    pub fn new(agent_id: impl Into<String>, action: AuditAction) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            agent_id: agent_id.into(),
            action,
            correlation_id: None,
            metadata: None,
        }
    }

    /// Set a correlation ID.
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

// ── RetentionPolicy ───────────────────────────────────────────────────────────

/// Retention policy for audit entries.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum number of entries to retain.
    pub max_entries: Option<usize>,
    /// Maximum age of entries.
    pub max_age: Option<Duration>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_entries: None,
            max_age: None,
        }
    }
}

impl RetentionPolicy {
    /// Create a policy with a maximum number of entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = Some(max);
        self
    }

    /// Create a policy with a maximum age.
    pub fn with_max_age(mut self, age: Duration) -> Self {
        self.max_age = Some(age);
        self
    }
}

// ── AuditTrail ────────────────────────────────────────────────────────────────

/// Append-only audit trail for tracking all agent actions.
#[derive(Debug)]
pub struct AuditTrail {
    entries: VecDeque<AuditEntry>,
    retention: RetentionPolicy,
}

impl AuditTrail {
    /// Create a new empty audit trail.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            retention: RetentionPolicy::default(),
        }
    }

    /// Create an audit trail with a retention policy.
    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Append a new entry to the trail (immutable — cannot be modified after).
    pub fn log(&mut self, entry: AuditEntry) {
        self.entries.push_back(entry);
        self.apply_retention();
    }

    /// Return the number of entries in the trail.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query entries by agent ID.
    pub fn query_by_agent(&self, agent_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.agent_id == agent_id)
            .collect()
    }

    /// Query entries by action type.
    pub fn query_by_action_type(&self, action_type: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.action.action_type() == action_type)
            .collect()
    }

    /// Query entries by correlation ID.
    pub fn query_by_correlation_id(&self, correlation_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.correlation_id.as_deref() == Some(correlation_id))
            .collect()
    }

    /// Query entries within a time range.
    pub fn query_by_time_range(&self, start: SystemTime, end: SystemTime) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Get all entries as a slice (for export/serialization).
    pub fn entries(&self) -> impl Iterator<Item = &AuditEntry> {
        self.entries.iter()
    }

    /// Apply the retention policy, removing old entries.
    fn apply_retention(&mut self) {
        // Apply max entries limit
        if let Some(max) = self.retention.max_entries {
            while self.entries.len() > max {
                self.entries.pop_front();
            }
        }

        // Apply max age limit
        if let Some(max_age) = self.retention.max_age {
            let cutoff = SystemTime::now() - max_age;
            while let Some(front) = self.entries.front() {
                if front.timestamp < cutoff {
                    self.entries.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-13.5: Append-only log of every action
    #[test]
    fn test_append_only_log() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "search".into(),
                input_hash: "hash1".into(),
                output_hash: "hash2".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::AgentResponse {
                model: "gpt-4".into(),
                token_count: 100,
            },
        ));

        assert_eq!(trail.len(), 2);
    }

    // REQ-13.5: Structured format — timestamp, agent, action, input/output hash
    #[test]
    fn test_structured_entry_format() {
        let entry = AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "calculator".into(),
                input_hash: "abc123".into(),
                output_hash: "def456".into(),
            },
        );

        assert!(!entry.id.is_empty());
        assert_eq!(entry.agent_id, "agent-1");
        assert_eq!(entry.action.action_type(), "tool_call");
        // Timestamp should be recent
        let elapsed = entry.timestamp.elapsed().unwrap();
        assert!(elapsed < Duration::from_secs(1));
    }

    // REQ-13.5: Query by agent
    #[test]
    fn test_query_by_agent() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "a".into(),
                input_hash: "".into(),
                output_hash: "".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "agent-2",
            AuditAction::ToolCall {
                tool_name: "b".into(),
                input_hash: "".into(),
                output_hash: "".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::DataAccess {
                resource: "db".into(),
                operation: "read".into(),
            },
        ));

        let results = trail.query_by_agent("agent-1");
        assert_eq!(results.len(), 2);
        let results = trail.query_by_agent("agent-2");
        assert_eq!(results.len(), 1);
    }

    // REQ-13.5: Query by action type
    #[test]
    fn test_query_by_action_type() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "a".into(),
                input_hash: "".into(),
                output_hash: "".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::DataAccess {
                resource: "db".into(),
                operation: "read".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "b".into(),
                input_hash: "".into(),
                output_hash: "".into(),
            },
        ));

        let results = trail.query_by_action_type("tool_call");
        assert_eq!(results.len(), 2);
        let results = trail.query_by_action_type("data_access");
        assert_eq!(results.len(), 1);
    }

    // REQ-13.5: Retention policy with max entries
    #[test]
    fn test_retention_max_entries() {
        let policy = RetentionPolicy::default().with_max_entries(2);
        let mut trail = AuditTrail::new().with_retention(policy);

        trail.log(AuditEntry::new(
            "a1",
            AuditAction::Custom {
                action_type: "t".into(),
                details: "first".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "a1",
            AuditAction::Custom {
                action_type: "t".into(),
                details: "second".into(),
            },
        ));
        trail.log(AuditEntry::new(
            "a1",
            AuditAction::Custom {
                action_type: "t".into(),
                details: "third".into(),
            },
        ));

        assert_eq!(trail.len(), 2);
        // First entry should be dropped
        let entries: Vec<_> = trail.entries().collect();
        if let AuditAction::Custom { details, .. } = &entries[0].action {
            assert_eq!(details, "second");
        }
    }

    // REQ-13.5: Correlation ID grouping
    #[test]
    fn test_correlation_id_query() {
        let mut trail = AuditTrail::new();
        trail.log(
            AuditEntry::new(
                "agent-1",
                AuditAction::ToolCall {
                    tool_name: "a".into(),
                    input_hash: "".into(),
                    output_hash: "".into(),
                },
            )
            .with_correlation_id("req-123"),
        );
        trail.log(
            AuditEntry::new(
                "agent-1",
                AuditAction::AgentResponse {
                    model: "gpt-4".into(),
                    token_count: 50,
                },
            )
            .with_correlation_id("req-123"),
        );
        trail.log(AuditEntry::new(
            "agent-2",
            AuditAction::ToolCall {
                tool_name: "b".into(),
                input_hash: "".into(),
                output_hash: "".into(),
            },
        ));

        let results = trail.query_by_correlation_id("req-123");
        assert_eq!(results.len(), 2);
    }

    // REQ-13.5: Data access tracking
    #[test]
    fn test_data_access_tracking() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::DataAccess {
                resource: "knowledge_base/docs".into(),
                operation: "read".into(),
            },
        ));

        let entries = trail.query_by_action_type("data_access");
        assert_eq!(entries.len(), 1);
        if let AuditAction::DataAccess {
            resource,
            operation,
        } = &entries[0].action
        {
            assert_eq!(resource, "knowledge_base/docs");
            assert_eq!(operation, "read");
        }
    }

    // REQ-13.5: Model invocation logging
    #[test]
    fn test_model_invocation_logging() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "agent-1",
            AuditAction::ModelInvocation {
                model: "gpt-4".into(),
                provider: "openai".into(),
            },
        ));

        let entries = trail.query_by_action_type("model_invocation");
        assert_eq!(entries.len(), 1);
    }

    // REQ-13.5: Metadata support
    #[test]
    fn test_metadata_support() {
        let entry = AuditEntry::new(
            "agent-1",
            AuditAction::ToolCall {
                tool_name: "search".into(),
                input_hash: "h1".into(),
                output_hash: "h2".into(),
            },
        )
        .with_metadata("user_ip=192.168.1.1");

        assert_eq!(entry.metadata, Some("user_ip=192.168.1.1".to_string()));
    }

    // REQ-13.5: Empty trail
    #[test]
    fn test_empty_trail() {
        let trail = AuditTrail::new();
        assert!(trail.is_empty());
        assert_eq!(trail.len(), 0);
        assert_eq!(trail.query_by_agent("any").len(), 0);
    }

    // REQ-13.5: Config change auditing
    #[test]
    fn test_config_change_auditing() {
        let mut trail = AuditTrail::new();
        trail.log(AuditEntry::new(
            "admin-1",
            AuditAction::ConfigChange {
                key: "max_tokens".into(),
                old_value: "1000".into(),
                new_value: "2000".into(),
            },
        ));

        let entries = trail.query_by_action_type("config_change");
        assert_eq!(entries.len(), 1);
        if let AuditAction::ConfigChange {
            key,
            old_value,
            new_value,
        } = &entries[0].action
        {
            assert_eq!(key, "max_tokens");
            assert_eq!(old_value, "1000");
            assert_eq!(new_value, "2000");
        }
    }
}
