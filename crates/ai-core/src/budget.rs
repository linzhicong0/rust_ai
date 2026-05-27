// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Budget Management (REQ-18.3)
//!
//! Support for budget limits per project, per agent, and per user with
//! automatic enforcement and alerts. Includes budget definitions with periods,
//! real-time spend tracking, and configurable actions on limit (warn, throttle, block).
//!
//! ## Example
//!
//! ```rust
//! use ai_core::budget::{
//!     Budget, BudgetManager, BudgetPeriod, BudgetScope, LimitAction,
//! };
//!
//! let mut manager = BudgetManager::new();
//! manager.create_budget(Budget::new("project-alpha", BudgetScope::Project("alpha".into()))
//!     .with_limit(100.0)
//!     .with_period(BudgetPeriod::Monthly)
//!     .with_action(LimitAction::Block));
//!
//! // Track spending
//! manager.record_spend("project-alpha", 25.0).unwrap();
//! let status = manager.get_status("project-alpha").unwrap();
//! assert_eq!(status.spent, 25.0);
//! assert_eq!(status.remaining, 75.0);
//! ```

use std::collections::HashMap;

// ── BudgetPeriod ──────────────────────────────────────────────────────────────

/// Budget period for tracking spending.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BudgetPeriod {
    /// Daily budget period.
    Daily,
    /// Weekly budget period.
    Weekly,
    /// Monthly budget period.
    Monthly,
    /// Custom period in hours.
    Custom { hours: u32 },
}

impl BudgetPeriod {
    /// Return the period name.
    pub fn name(&self) -> &str {
        match self {
            BudgetPeriod::Daily => "daily",
            BudgetPeriod::Weekly => "weekly",
            BudgetPeriod::Monthly => "monthly",
            BudgetPeriod::Custom { .. } => "custom",
        }
    }

    /// Return the period duration in hours.
    pub fn hours(&self) -> u32 {
        match self {
            BudgetPeriod::Daily => 24,
            BudgetPeriod::Weekly => 168,
            BudgetPeriod::Monthly => 720, // Approximate
            BudgetPeriod::Custom { hours } => *hours,
        }
    }
}

// ── BudgetScope ───────────────────────────────────────────────────────────────

/// Scope to which a budget applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BudgetScope {
    /// Budget per project.
    Project(String),
    /// Budget per agent.
    Agent(String),
    /// Budget per user.
    User(String),
    /// Global budget (applies to everything).
    Global,
}

impl BudgetScope {
    /// Return a description of the scope.
    pub fn description(&self) -> String {
        match self {
            BudgetScope::Project(name) => format!("project:{}", name),
            BudgetScope::Agent(name) => format!("agent:{}", name),
            BudgetScope::User(name) => format!("user:{}", name),
            BudgetScope::Global => "global".to_string(),
        }
    }

    /// Return the scope type name.
    pub fn scope_type(&self) -> &str {
        match self {
            BudgetScope::Project(_) => "project",
            BudgetScope::Agent(_) => "agent",
            BudgetScope::User(_) => "user",
            BudgetScope::Global => "global",
        }
    }
}

// ── LimitAction ───────────────────────────────────────────────────────────────

/// Action to take when a budget limit is reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitAction {
    /// Emit a warning but allow requests to continue.
    Warn,
    /// Throttle requests (reduce rate/quality).
    Throttle {
        /// Factor to reduce throughput by (0.0 to 1.0).
        factor: u32, // stored as percentage (0-100)
    },
    /// Block all further requests.
    Block,
}

impl LimitAction {
    /// Return the action name.
    pub fn name(&self) -> &str {
        match self {
            LimitAction::Warn => "warn",
            LimitAction::Throttle { .. } => "throttle",
            LimitAction::Block => "block",
        }
    }
}

// ── AlertThreshold ────────────────────────────────────────────────────────────

/// Threshold at which to trigger an alert.
#[derive(Debug, Clone, PartialEq)]
pub struct AlertThreshold {
    /// Percentage of budget consumed (0.0 to 1.0).
    pub percentage: f64,
    /// Whether this alert has been triggered.
    pub triggered: bool,
    /// Label for this threshold.
    pub label: String,
}

impl AlertThreshold {
    /// Create a new alert threshold.
    pub fn new(percentage: f64, label: impl Into<String>) -> Self {
        Self {
            percentage,
            triggered: false,
            label: label.into(),
        }
    }
}

// ── Budget ────────────────────────────────────────────────────────────────────

/// A budget definition with limit, period, scope, and enforcement actions.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Unique budget identifier.
    pub id: String,
    /// Scope this budget applies to.
    pub scope: BudgetScope,
    /// Maximum spending limit in USD.
    pub limit: f64,
    /// Budget period.
    pub period: BudgetPeriod,
    /// Action to take when limit is reached.
    pub limit_action: LimitAction,
    /// Alert thresholds.
    pub alert_thresholds: Vec<AlertThreshold>,
    /// Whether the budget is active.
    pub active: bool,
}

impl Budget {
    /// Create a new budget.
    pub fn new(id: impl Into<String>, scope: BudgetScope) -> Self {
        Self {
            id: id.into(),
            scope,
            limit: 0.0,
            period: BudgetPeriod::Monthly,
            limit_action: LimitAction::Warn,
            alert_thresholds: vec![
                AlertThreshold::new(0.5, "50% consumed"),
                AlertThreshold::new(0.8, "80% consumed"),
                AlertThreshold::new(0.95, "95% consumed"),
            ],
            active: true,
        }
    }

    /// Set the spending limit.
    pub fn with_limit(mut self, limit: f64) -> Self {
        self.limit = limit;
        self
    }

    /// Set the budget period.
    pub fn with_period(mut self, period: BudgetPeriod) -> Self {
        self.period = period;
        self
    }

    /// Set the limit action.
    pub fn with_action(mut self, action: LimitAction) -> Self {
        self.limit_action = action;
        self
    }

    /// Set custom alert thresholds.
    pub fn with_thresholds(mut self, thresholds: Vec<AlertThreshold>) -> Self {
        self.alert_thresholds = thresholds;
        self
    }

    /// Deactivate the budget.
    pub fn deactivate(mut self) -> Self {
        self.active = false;
        self
    }
}

// ── BudgetStatus ──────────────────────────────────────────────────────────────

/// Real-time status of a budget.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetStatus {
    /// Budget identifier.
    pub budget_id: String,
    /// Total amount spent in current period.
    pub spent: f64,
    /// Remaining budget.
    pub remaining: f64,
    /// Budget limit.
    pub limit: f64,
    /// Utilization percentage (0.0 to 1.0).
    pub utilization: f64,
    /// Whether the budget is exceeded.
    pub exceeded: bool,
    /// Current enforcement action (if limit reached).
    pub enforcement: Option<LimitAction>,
    /// Alerts that have been triggered.
    pub triggered_alerts: Vec<String>,
}

// ── BudgetAlert ───────────────────────────────────────────────────────────────

/// An alert generated when a budget threshold is crossed.
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Budget identifier.
    pub budget_id: String,
    /// Alert threshold label.
    pub threshold_label: String,
    /// Current utilization when alert was triggered.
    pub utilization: f64,
    /// Budget scope description.
    pub scope: String,
}

// ── BudgetError ───────────────────────────────────────────────────────────────

/// Errors in budget management.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// Budget not found.
    #[error("Budget not found: {0}")]
    NotFound(String),
    /// Budget limit exceeded.
    #[error("Budget limit exceeded for '{budget_id}': spent {spent:.2}, limit {limit:.2}")]
    LimitExceeded {
        budget_id: String,
        spent: f64,
        limit: f64,
    },
    /// Invalid budget configuration.
    #[error("Invalid budget config: {0}")]
    InvalidConfig(String),
    /// Budget is inactive.
    #[error("Budget '{0}' is inactive")]
    Inactive(String),
}

// ── BudgetManager ─────────────────────────────────────────────────────────────

/// Manages budgets, tracks spending, and enforces limits.
pub struct BudgetManager {
    /// All budgets, keyed by ID.
    budgets: HashMap<String, Budget>,
    /// Current spending per budget ID.
    spending: HashMap<String, f64>,
    /// Alert history.
    alerts: Vec<BudgetAlert>,
    /// Alert callback (optional): called when alert is triggered.
    alert_callbacks: Vec<Box<dyn Fn(&BudgetAlert) + Send + Sync>>,
}

impl BudgetManager {
    /// Create a new budget manager.
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
            spending: HashMap::new(),
            alerts: Vec::new(),
            alert_callbacks: Vec::new(),
        }
    }

    /// Create and register a new budget.
    pub fn create_budget(&mut self, budget: Budget) -> Result<(), BudgetError> {
        if budget.id.is_empty() {
            return Err(BudgetError::InvalidConfig("budget id is required".into()));
        }
        if budget.limit < 0.0 {
            return Err(BudgetError::InvalidConfig(
                "budget limit must be non-negative".into(),
            ));
        }
        self.spending.insert(budget.id.clone(), 0.0);
        self.budgets.insert(budget.id.clone(), budget);
        Ok(())
    }

    /// Remove a budget.
    pub fn remove_budget(&mut self, budget_id: &str) -> Option<Budget> {
        self.spending.remove(budget_id);
        self.budgets.remove(budget_id)
    }

    /// Get a budget by ID.
    pub fn get_budget(&self, budget_id: &str) -> Option<&Budget> {
        self.budgets.get(budget_id)
    }

    /// List all budget IDs.
    pub fn budget_ids(&self) -> Vec<&str> {
        self.budgets.keys().map(|k| k.as_str()).collect()
    }

    /// Record spending against a budget.
    pub fn record_spend(&mut self, budget_id: &str, amount: f64) -> Result<(), BudgetError> {
        let budget = self
            .budgets
            .get(budget_id)
            .ok_or_else(|| BudgetError::NotFound(budget_id.to_string()))?;

        if !budget.active {
            return Err(BudgetError::Inactive(budget_id.to_string()));
        }

        let current_spend = self.spending.entry(budget_id.to_string()).or_insert(0.0);
        let new_spend = *current_spend + amount;

        // Check if limit would be exceeded and action is Block
        if new_spend > budget.limit && budget.limit_action == LimitAction::Block {
            return Err(BudgetError::LimitExceeded {
                budget_id: budget_id.to_string(),
                spent: new_spend,
                limit: budget.limit,
            });
        }

        *current_spend = new_spend;

        // Check alert thresholds
        let utilization = if budget.limit > 0.0 {
            new_spend / budget.limit
        } else {
            0.0
        };

        // Clone the data we need to avoid borrow issues
        let thresholds: Vec<(f64, String, bool)> = self
            .budgets
            .get(budget_id)
            .map(|b| {
                b.alert_thresholds
                    .iter()
                    .map(|t| (t.percentage, t.label.clone(), t.triggered))
                    .collect()
            })
            .unwrap_or_default();

        for (i, (percentage, label, triggered)) in thresholds.iter().enumerate() {
            if utilization >= *percentage && !triggered {
                // Mark as triggered
                if let Some(budget) = self.budgets.get_mut(budget_id) {
                    if let Some(threshold) = budget.alert_thresholds.get_mut(i) {
                        threshold.triggered = true;
                    }
                }

                let alert = BudgetAlert {
                    budget_id: budget_id.to_string(),
                    threshold_label: label.clone(),
                    utilization,
                    scope: self
                        .budgets
                        .get(budget_id)
                        .map(|b| b.scope.description())
                        .unwrap_or_default(),
                };

                // Invoke callbacks
                for callback in &self.alert_callbacks {
                    callback(&alert);
                }

                self.alerts.push(alert);
            }
        }

        Ok(())
    }

    /// Get the current status of a budget.
    pub fn get_status(&self, budget_id: &str) -> Result<BudgetStatus, BudgetError> {
        let budget = self
            .budgets
            .get(budget_id)
            .ok_or_else(|| BudgetError::NotFound(budget_id.to_string()))?;

        let spent = self.spending.get(budget_id).copied().unwrap_or(0.0);
        let remaining = (budget.limit - spent).max(0.0);
        let utilization = if budget.limit > 0.0 {
            spent / budget.limit
        } else {
            0.0
        };
        let exceeded = spent > budget.limit;

        let enforcement = if exceeded {
            Some(budget.limit_action.clone())
        } else {
            None
        };

        let triggered_alerts: Vec<String> = budget
            .alert_thresholds
            .iter()
            .filter(|t| t.triggered)
            .map(|t| t.label.clone())
            .collect();

        Ok(BudgetStatus {
            budget_id: budget_id.to_string(),
            spent,
            remaining,
            limit: budget.limit,
            utilization,
            exceeded,
            enforcement,
            triggered_alerts,
        })
    }

    /// Check if a spend amount would be allowed under the budget.
    pub fn can_spend(&self, budget_id: &str, amount: f64) -> Result<bool, BudgetError> {
        let budget = self
            .budgets
            .get(budget_id)
            .ok_or_else(|| BudgetError::NotFound(budget_id.to_string()))?;

        if !budget.active {
            return Err(BudgetError::Inactive(budget_id.to_string()));
        }

        let current_spend = self.spending.get(budget_id).copied().unwrap_or(0.0);
        let would_exceed = (current_spend + amount) > budget.limit;

        if would_exceed && budget.limit_action == LimitAction::Block {
            return Ok(false);
        }

        Ok(true)
    }

    /// Reset spending for a budget (e.g., at the start of a new period).
    pub fn reset_spending(&mut self, budget_id: &str) -> Result<(), BudgetError> {
        if !self.budgets.contains_key(budget_id) {
            return Err(BudgetError::NotFound(budget_id.to_string()));
        }

        self.spending.insert(budget_id.to_string(), 0.0);

        // Reset alert triggers
        if let Some(budget) = self.budgets.get_mut(budget_id) {
            for threshold in &mut budget.alert_thresholds {
                threshold.triggered = false;
            }
        }

        Ok(())
    }

    /// Get all alerts that have been generated.
    pub fn alerts(&self) -> &[BudgetAlert] {
        &self.alerts
    }

    /// Register an alert callback.
    pub fn on_alert(&mut self, callback: impl Fn(&BudgetAlert) + Send + Sync + 'static) {
        self.alert_callbacks.push(Box::new(callback));
    }

    /// Get total spending across all budgets.
    pub fn total_spending(&self) -> f64 {
        self.spending.values().sum()
    }
}

impl Default for BudgetManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-18.3: Budget definitions with period (daily, monthly)
    #[test]
    fn test_budget_period_definitions() {
        let daily = BudgetPeriod::Daily;
        assert_eq!(daily.name(), "daily");
        assert_eq!(daily.hours(), 24);

        let monthly = BudgetPeriod::Monthly;
        assert_eq!(monthly.name(), "monthly");
        assert_eq!(monthly.hours(), 720);

        let custom = BudgetPeriod::Custom { hours: 48 };
        assert_eq!(custom.name(), "custom");
        assert_eq!(custom.hours(), 48);
    }

    // REQ-18.3: Budget per project
    #[test]
    fn test_budget_per_project() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("proj-alpha", BudgetScope::Project("alpha".into()))
                    .with_limit(100.0)
                    .with_period(BudgetPeriod::Monthly),
            )
            .unwrap();

        let budget = manager.get_budget("proj-alpha").unwrap();
        assert_eq!(budget.scope, BudgetScope::Project("alpha".into()));
        assert_eq!(budget.limit, 100.0);
        assert_eq!(budget.period, BudgetPeriod::Monthly);
    }

    // REQ-18.3: Budget per agent
    #[test]
    fn test_budget_per_agent() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("agent-chatbot", BudgetScope::Agent("chatbot".into()))
                    .with_limit(50.0)
                    .with_period(BudgetPeriod::Daily),
            )
            .unwrap();

        let budget = manager.get_budget("agent-chatbot").unwrap();
        assert_eq!(budget.scope, BudgetScope::Agent("chatbot".into()));
        assert_eq!(budget.limit, 50.0);
        assert_eq!(budget.period, BudgetPeriod::Daily);
    }

    // REQ-18.3: Budget per user
    #[test]
    fn test_budget_per_user() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("user-alice", BudgetScope::User("alice".into()))
                    .with_limit(200.0)
                    .with_period(BudgetPeriod::Monthly),
            )
            .unwrap();

        let budget = manager.get_budget("user-alice").unwrap();
        assert_eq!(budget.scope, BudgetScope::User("alice".into()));
        assert_eq!(budget.scope.scope_type(), "user");
    }

    // REQ-18.3: Real-time spend tracking against budget
    #[test]
    fn test_real_time_spend_tracking() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("proj-1", BudgetScope::Project("one".into())).with_limit(100.0),
            )
            .unwrap();

        manager.record_spend("proj-1", 25.0).unwrap();
        manager.record_spend("proj-1", 15.0).unwrap();

        let status = manager.get_status("proj-1").unwrap();
        assert_eq!(status.spent, 40.0);
        assert_eq!(status.remaining, 60.0);
        assert!((status.utilization - 0.4).abs() < f64::EPSILON);
        assert!(!status.exceeded);
    }

    // REQ-18.3: Actions on limit - warn
    #[test]
    fn test_limit_action_warn() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("warn-budget", BudgetScope::Global)
                    .with_limit(10.0)
                    .with_action(LimitAction::Warn),
            )
            .unwrap();

        // Should succeed even when exceeding (warn only)
        manager.record_spend("warn-budget", 8.0).unwrap();
        manager.record_spend("warn-budget", 5.0).unwrap(); // Exceeds limit

        let status = manager.get_status("warn-budget").unwrap();
        assert!(status.exceeded);
        assert_eq!(status.enforcement, Some(LimitAction::Warn));
    }

    // REQ-18.3: Actions on limit - throttle
    #[test]
    fn test_limit_action_throttle() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("throttle-budget", BudgetScope::Agent("bot".into()))
                    .with_limit(50.0)
                    .with_action(LimitAction::Throttle { factor: 50 }),
            )
            .unwrap();

        // Should succeed even when exceeding (throttle only)
        manager.record_spend("throttle-budget", 60.0).unwrap();

        let status = manager.get_status("throttle-budget").unwrap();
        assert!(status.exceeded);
        assert_eq!(
            status.enforcement,
            Some(LimitAction::Throttle { factor: 50 })
        );
    }

    // REQ-18.3: Actions on limit - block
    #[test]
    fn test_limit_action_block() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("block-budget", BudgetScope::User("bob".into()))
                    .with_limit(20.0)
                    .with_action(LimitAction::Block),
            )
            .unwrap();

        manager.record_spend("block-budget", 15.0).unwrap();
        // This should fail - would exceed limit
        let result = manager.record_spend("block-budget", 10.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BudgetError::LimitExceeded { .. }
        ));
    }

    // REQ-18.3: can_spend check
    #[test]
    fn test_can_spend() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("check-budget", BudgetScope::Global)
                    .with_limit(100.0)
                    .with_action(LimitAction::Block),
            )
            .unwrap();

        manager.record_spend("check-budget", 80.0).unwrap();

        assert!(manager.can_spend("check-budget", 10.0).unwrap()); // 90 < 100
        assert!(!manager.can_spend("check-budget", 30.0).unwrap()); // 110 > 100
    }

    // REQ-18.3: Alert callbacks for integration with monitoring
    #[test]
    fn test_alert_thresholds() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("alert-budget", BudgetScope::Project("demo".into()))
                    .with_limit(100.0)
                    .with_thresholds(vec![
                        AlertThreshold::new(0.5, "50% warning"),
                        AlertThreshold::new(0.8, "80% critical"),
                    ]),
            )
            .unwrap();

        // Spend 60% - should trigger 50% alert
        manager.record_spend("alert-budget", 60.0).unwrap();

        let status = manager.get_status("alert-budget").unwrap();
        assert!(status.triggered_alerts.contains(&"50% warning".to_string()));
        assert!(!status
            .triggered_alerts
            .contains(&"80% critical".to_string()));

        // Spend more to trigger 80% alert
        manager.record_spend("alert-budget", 25.0).unwrap();

        let status = manager.get_status("alert-budget").unwrap();
        assert!(status
            .triggered_alerts
            .contains(&"80% critical".to_string()));
    }

    // REQ-18.3: Alert callback invocation
    #[test]
    fn test_alert_callback() {
        use std::sync::{Arc, Mutex};

        let alerts_received = Arc::new(Mutex::new(Vec::new()));
        let alerts_clone = alerts_received.clone();

        let mut manager = BudgetManager::new();
        manager.on_alert(move |alert| {
            alerts_clone
                .lock()
                .unwrap()
                .push(alert.threshold_label.clone());
        });

        manager
            .create_budget(
                Budget::new("cb-budget", BudgetScope::Global)
                    .with_limit(100.0)
                    .with_thresholds(vec![AlertThreshold::new(0.5, "half spent")]),
            )
            .unwrap();

        manager.record_spend("cb-budget", 60.0).unwrap();

        let received = alerts_received.lock().unwrap();
        assert!(received.contains(&"half spent".to_string()));
    }

    // REQ-18.3: Reset spending for new period
    #[test]
    fn test_reset_spending() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(Budget::new("reset-budget", BudgetScope::Global).with_limit(100.0))
            .unwrap();

        manager.record_spend("reset-budget", 75.0).unwrap();
        assert_eq!(manager.get_status("reset-budget").unwrap().spent, 75.0);

        manager.reset_spending("reset-budget").unwrap();
        assert_eq!(manager.get_status("reset-budget").unwrap().spent, 0.0);
    }

    // REQ-18.3: Budget not found error
    #[test]
    fn test_budget_not_found() {
        let manager = BudgetManager::new();
        let result = manager.get_status("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BudgetError::NotFound(_)));
    }

    // REQ-18.3: Invalid budget config
    #[test]
    fn test_invalid_budget_config() {
        let mut manager = BudgetManager::new();

        // Empty ID
        let result = manager.create_budget(Budget::new("", BudgetScope::Global).with_limit(100.0));
        assert!(result.is_err());

        // Negative limit
        let result =
            manager.create_budget(Budget::new("neg", BudgetScope::Global).with_limit(-10.0));
        assert!(result.is_err());
    }

    // REQ-18.3: Inactive budget
    #[test]
    fn test_inactive_budget() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(
                Budget::new("inactive", BudgetScope::Global)
                    .with_limit(100.0)
                    .deactivate(),
            )
            .unwrap();

        let result = manager.record_spend("inactive", 10.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BudgetError::Inactive(_)));
    }

    // REQ-18.3: Total spending across all budgets
    #[test]
    fn test_total_spending() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(Budget::new("b1", BudgetScope::Project("a".into())).with_limit(100.0))
            .unwrap();
        manager
            .create_budget(Budget::new("b2", BudgetScope::Project("b".into())).with_limit(200.0))
            .unwrap();

        manager.record_spend("b1", 30.0).unwrap();
        manager.record_spend("b2", 50.0).unwrap();

        assert_eq!(manager.total_spending(), 80.0);
    }

    // REQ-18.3: Remove budget
    #[test]
    fn test_remove_budget() {
        let mut manager = BudgetManager::new();
        manager
            .create_budget(Budget::new("removable", BudgetScope::Global).with_limit(100.0))
            .unwrap();

        assert!(manager.get_budget("removable").is_some());
        let removed = manager.remove_budget("removable");
        assert!(removed.is_some());
        assert!(manager.get_budget("removable").is_none());
    }

    // REQ-18.3: Scope descriptions
    #[test]
    fn test_scope_descriptions() {
        assert_eq!(
            BudgetScope::Project("alpha".into()).description(),
            "project:alpha"
        );
        assert_eq!(BudgetScope::Agent("bot".into()).description(), "agent:bot");
        assert_eq!(
            BudgetScope::User("alice".into()).description(),
            "user:alice"
        );
        assert_eq!(BudgetScope::Global.description(), "global");
    }
}
