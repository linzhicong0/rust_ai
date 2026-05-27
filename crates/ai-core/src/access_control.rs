// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Access Control (REQ-13.4)
//!
//! Role-based access control (RBAC) for agents, tools, models, and knowledge bases.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::access_control::{AccessControl, Role, Permission, ResourceType};
//!
//! let mut ac = AccessControl::new();
//! ac.grant(Role::Developer, Permission::Use, ResourceType::Tool("web_search".into()));
//! ac.grant(Role::Admin, Permission::Manage, ResourceType::Model("gpt-4".into()));
//!
//! assert!(ac.is_allowed(&Role::Developer, &Permission::Use, &ResourceType::Tool("web_search".into())));
//! assert!(!ac.is_allowed(&Role::User, &Permission::Manage, &ResourceType::Model("gpt-4".into())));
//! ```

use std::collections::HashSet;

// ── Role ──────────────────────────────────────────────────────────────────────

/// Predefined roles for access control.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full access to all resources.
    Admin,
    /// Can create and configure agents, tools, and pipelines.
    Developer,
    /// Can use agents and tools but not configure them.
    User,
    /// An AI agent acting on behalf of a user.
    Agent,
    /// A custom role with an arbitrary name.
    Custom(String),
}

impl Role {
    /// Return the human-readable name of this role.
    pub fn name(&self) -> &str {
        match self {
            Role::Admin => "admin",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Agent => "agent",
            Role::Custom(name) => name.as_str(),
        }
    }
}

// ── Permission ────────────────────────────────────────────────────────────────

/// Permission levels for resource access.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    /// Can use/invoke the resource.
    Use,
    /// Can read/view the resource configuration.
    Read,
    /// Can modify/configure the resource.
    Write,
    /// Can manage (create, delete, grant) the resource.
    Manage,
    /// A custom permission with an arbitrary name.
    Custom(String),
}

impl Permission {
    /// Return the human-readable name of this permission.
    pub fn name(&self) -> &str {
        match self {
            Permission::Use => "use",
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Manage => "manage",
            Permission::Custom(name) => name.as_str(),
        }
    }
}

// ── ResourceType ──────────────────────────────────────────────────────────────

/// Types of resources that can be protected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// A tool resource identified by name.
    Tool(String),
    /// A model resource identified by name.
    Model(String),
    /// An agent resource identified by name.
    Agent(String),
    /// A knowledge base resource identified by name.
    KnowledgeBase(String),
    /// A pipeline resource identified by name.
    Pipeline(String),
    /// Any resource of a given type (wildcard).
    Any(String),
}

impl ResourceType {
    /// Return the type category name.
    pub fn category(&self) -> &str {
        match self {
            ResourceType::Tool(_) => "tool",
            ResourceType::Model(_) => "model",
            ResourceType::Agent(_) => "agent",
            ResourceType::KnowledgeBase(_) => "knowledge_base",
            ResourceType::Pipeline(_) => "pipeline",
            ResourceType::Any(_) => "any",
        }
    }

    /// Return the resource identifier.
    pub fn id(&self) -> &str {
        match self {
            ResourceType::Tool(id)
            | ResourceType::Model(id)
            | ResourceType::Agent(id)
            | ResourceType::KnowledgeBase(id)
            | ResourceType::Pipeline(id)
            | ResourceType::Any(id) => id.as_str(),
        }
    }
}

// ── Policy Entry ──────────────────────────────────────────────────────────────

/// A single access control policy entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PolicyEntry {
    role: Role,
    permission: Permission,
    resource: ResourceType,
}

// ── AccessControlError ────────────────────────────────────────────────────────

/// Errors from access control checks.
#[derive(Debug, thiserror::Error)]
pub enum AccessControlError {
    /// The action is denied by policy.
    #[error("Access denied: role '{role}' does not have '{permission}' on {resource_type}/{resource_id}")]
    Denied {
        role: String,
        permission: String,
        resource_type: String,
        resource_id: String,
    },
    /// The requested resource is not registered.
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
}

// ── AccessDecision ────────────────────────────────────────────────────────────

/// Result of an access control check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDecision {
    /// Access is allowed.
    Allow,
    /// Access is denied with a reason.
    Deny(String),
}

// ── AccessControl ─────────────────────────────────────────────────────────────

/// Role-based access control manager.
///
/// Maintains a set of policies (role, permission, resource) and enforces them
/// at agent and pipeline level.
#[derive(Debug, Clone)]
pub struct AccessControl {
    policies: HashSet<PolicyEntry>,
    /// If true, Admin role bypasses all checks.
    admin_bypass: bool,
}

impl AccessControl {
    /// Create a new empty access control manager with admin bypass enabled.
    pub fn new() -> Self {
        Self {
            policies: HashSet::new(),
            admin_bypass: true,
        }
    }

    /// Disable admin bypass (admin must have explicit grants).
    pub fn without_admin_bypass(mut self) -> Self {
        self.admin_bypass = false;
        self
    }

    /// Grant a role permission on a resource.
    pub fn grant(&mut self, role: Role, permission: Permission, resource: ResourceType) {
        self.policies.insert(PolicyEntry {
            role,
            permission,
            resource,
        });
    }

    /// Revoke a previously granted permission.
    pub fn revoke(&mut self, role: &Role, permission: &Permission, resource: &ResourceType) {
        self.policies.retain(|entry| {
            !(&entry.role == role && &entry.permission == permission && &entry.resource == resource)
        });
    }

    /// Check if a role has a specific permission on a resource.
    pub fn is_allowed(
        &self,
        role: &Role,
        permission: &Permission,
        resource: &ResourceType,
    ) -> bool {
        // Admin bypass
        if self.admin_bypass && *role == Role::Admin {
            return true;
        }

        // Direct policy match
        self.policies.contains(&PolicyEntry {
            role: role.clone(),
            permission: permission.clone(),
            resource: resource.clone(),
        })
    }

    /// Enforce access control, returning an error if denied.
    pub fn enforce(
        &self,
        role: &Role,
        permission: &Permission,
        resource: &ResourceType,
    ) -> Result<(), AccessControlError> {
        if self.is_allowed(role, permission, resource) {
            Ok(())
        } else {
            Err(AccessControlError::Denied {
                role: role.name().to_string(),
                permission: permission.name().to_string(),
                resource_type: resource.category().to_string(),
                resource_id: resource.id().to_string(),
            })
        }
    }

    /// Check access and return a decision.
    pub fn check(
        &self,
        role: &Role,
        permission: &Permission,
        resource: &ResourceType,
    ) -> AccessDecision {
        if self.is_allowed(role, permission, resource) {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny(format!(
                "Role '{}' does not have '{}' permission on {}/{}",
                role.name(),
                permission.name(),
                resource.category(),
                resource.id()
            ))
        }
    }

    /// List all permissions granted to a specific role.
    pub fn list_permissions(&self, role: &Role) -> Vec<(&Permission, &ResourceType)> {
        self.policies
            .iter()
            .filter(|entry| &entry.role == role)
            .map(|entry| (&entry.permission, &entry.resource))
            .collect()
    }

    /// Return the number of policy entries.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for AccessControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-13.4: Role definitions (admin, developer, user, agent)
    #[test]
    fn test_role_definitions() {
        assert_eq!(Role::Admin.name(), "admin");
        assert_eq!(Role::Developer.name(), "developer");
        assert_eq!(Role::User.name(), "user");
        assert_eq!(Role::Agent.name(), "agent");
        assert_eq!(Role::Custom("reviewer".into()).name(), "reviewer");
    }

    // REQ-13.4: Permission matrix — grant and check
    #[test]
    fn test_grant_and_check_permission() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::Developer,
            Permission::Use,
            ResourceType::Tool("web_search".into()),
        );

        assert!(ac.is_allowed(
            &Role::Developer,
            &Permission::Use,
            &ResourceType::Tool("web_search".into())
        ));
        // Different resource should be denied
        assert!(!ac.is_allowed(
            &Role::Developer,
            &Permission::Use,
            &ResourceType::Tool("file_write".into())
        ));
    }

    // REQ-13.4: Admin bypass — admin has access to everything
    #[test]
    fn test_admin_bypass() {
        let ac = AccessControl::new();
        // Admin can access anything without explicit grants
        assert!(ac.is_allowed(
            &Role::Admin,
            &Permission::Manage,
            &ResourceType::Model("gpt-4".into())
        ));
    }

    // REQ-13.4: Admin bypass can be disabled
    #[test]
    fn test_admin_bypass_disabled() {
        let ac = AccessControl::new().without_admin_bypass();
        // Without bypass, admin needs explicit grants
        assert!(!ac.is_allowed(
            &Role::Admin,
            &Permission::Manage,
            &ResourceType::Model("gpt-4".into())
        ));
    }

    // REQ-13.4: User role cannot manage models
    #[test]
    fn test_user_cannot_manage_models() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::User,
            Permission::Use,
            ResourceType::Model("gpt-4".into()),
        );

        assert!(ac.is_allowed(
            &Role::User,
            &Permission::Use,
            &ResourceType::Model("gpt-4".into())
        ));
        assert!(!ac.is_allowed(
            &Role::User,
            &Permission::Manage,
            &ResourceType::Model("gpt-4".into())
        ));
    }

    // REQ-13.4: Agent role access control
    #[test]
    fn test_agent_role_access() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::Agent,
            Permission::Use,
            ResourceType::Tool("calculator".into()),
        );
        ac.grant(
            Role::Agent,
            Permission::Read,
            ResourceType::KnowledgeBase("docs".into()),
        );

        assert!(ac.is_allowed(
            &Role::Agent,
            &Permission::Use,
            &ResourceType::Tool("calculator".into())
        ));
        assert!(ac.is_allowed(
            &Role::Agent,
            &Permission::Read,
            &ResourceType::KnowledgeBase("docs".into())
        ));
        // Agent cannot write to knowledge base
        assert!(!ac.is_allowed(
            &Role::Agent,
            &Permission::Write,
            &ResourceType::KnowledgeBase("docs".into())
        ));
    }

    // REQ-13.4: Revoke permission
    #[test]
    fn test_revoke_permission() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::Developer,
            Permission::Use,
            ResourceType::Tool("dangerous_tool".into()),
        );
        assert!(ac.is_allowed(
            &Role::Developer,
            &Permission::Use,
            &ResourceType::Tool("dangerous_tool".into())
        ));

        ac.revoke(
            &Role::Developer,
            &Permission::Use,
            &ResourceType::Tool("dangerous_tool".into()),
        );
        assert!(!ac.is_allowed(
            &Role::Developer,
            &Permission::Use,
            &ResourceType::Tool("dangerous_tool".into())
        ));
    }

    // REQ-13.4: Enforce returns error on denied access
    #[test]
    fn test_enforce_returns_error() {
        let ac = AccessControl::new().without_admin_bypass();
        let result = ac.enforce(
            &Role::User,
            &Permission::Manage,
            &ResourceType::Agent("my_agent".into()),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Access denied"));
    }

    // REQ-13.4: Policy enforcement at pipeline level
    #[test]
    fn test_pipeline_resource_access() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::Developer,
            Permission::Write,
            ResourceType::Pipeline("data_pipeline".into()),
        );

        assert!(ac.is_allowed(
            &Role::Developer,
            &Permission::Write,
            &ResourceType::Pipeline("data_pipeline".into())
        ));
        assert!(!ac.is_allowed(
            &Role::User,
            &Permission::Write,
            &ResourceType::Pipeline("data_pipeline".into())
        ));
    }

    // REQ-13.4: Access decision check
    #[test]
    fn test_access_decision() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::User,
            Permission::Use,
            ResourceType::Tool("search".into()),
        );

        let decision = ac.check(
            &Role::User,
            &Permission::Use,
            &ResourceType::Tool("search".into()),
        );
        assert_eq!(decision, AccessDecision::Allow);

        let decision = ac.check(
            &Role::User,
            &Permission::Manage,
            &ResourceType::Tool("search".into()),
        );
        assert!(matches!(decision, AccessDecision::Deny(_)));
    }

    // REQ-13.4: List permissions for a role
    #[test]
    fn test_list_permissions() {
        let mut ac = AccessControl::new();
        ac.grant(
            Role::Developer,
            Permission::Use,
            ResourceType::Tool("a".into()),
        );
        ac.grant(
            Role::Developer,
            Permission::Read,
            ResourceType::Model("b".into()),
        );
        ac.grant(Role::User, Permission::Use, ResourceType::Tool("c".into()));

        let perms = ac.list_permissions(&Role::Developer);
        assert_eq!(perms.len(), 2);
    }

    // REQ-13.4: Multiple roles same resource
    #[test]
    fn test_multiple_roles_same_resource() {
        let mut ac = AccessControl::new();
        let resource = ResourceType::Model("gpt-4".into());
        ac.grant(Role::Developer, Permission::Use, resource.clone());
        ac.grant(Role::User, Permission::Use, resource.clone());

        assert!(ac.is_allowed(&Role::Developer, &Permission::Use, &resource));
        assert!(ac.is_allowed(&Role::User, &Permission::Use, &resource));
        assert!(!ac.is_allowed(&Role::Agent, &Permission::Use, &resource));
    }
}
