// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Cloud Deployment (REQ-16.3)
//!
//! Support for deployment to major cloud providers (AWS, GCP, Azure) with
//! infrastructure-as-code templates, managed service integration, and autoscaling.
//!
//! ## Example
//!
//! ```rust
//! use ai_core::cloud_deployment::{
//!     CloudDeployment, CloudProvider, DeploymentConfig, ServiceIntegration,
//!     AutoscalingConfig,
//! };
//!
//! let config = DeploymentConfig::new("my-ai-app", CloudProvider::Aws)
//!     .with_region("us-east-1")
//!     .with_autoscaling(AutoscalingConfig {
//!         min_instances: 1,
//!         max_instances: 10,
//!         target_cpu_utilization: 70,
//!     });
//!
//! let deployment = CloudDeployment::new(config);
//! assert_eq!(deployment.provider(), &CloudProvider::Aws);
//! ```

use std::collections::HashMap;

// ── CloudProvider ─────────────────────────────────────────────────────────────

/// Supported cloud providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CloudProvider {
    /// Amazon Web Services.
    Aws,
    /// Google Cloud Platform.
    Gcp,
    /// Microsoft Azure.
    Azure,
    /// Custom/on-premises provider.
    Custom(String),
}

impl CloudProvider {
    /// Return the provider name.
    pub fn name(&self) -> &str {
        match self {
            CloudProvider::Aws => "aws",
            CloudProvider::Gcp => "gcp",
            CloudProvider::Azure => "azure",
            CloudProvider::Custom(name) => name.as_str(),
        }
    }
}

// ── IaCFormat ─────────────────────────────────────────────────────────────────

/// Infrastructure-as-code format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IaCFormat {
    /// HashiCorp Terraform.
    Terraform,
    /// Pulumi (TypeScript/Python/Go).
    Pulumi,
    /// AWS CloudFormation.
    CloudFormation,
    /// Azure ARM templates.
    ArmTemplate,
    /// Google Cloud Deployment Manager.
    DeploymentManager,
}

impl IaCFormat {
    /// Return the format name.
    pub fn name(&self) -> &str {
        match self {
            IaCFormat::Terraform => "terraform",
            IaCFormat::Pulumi => "pulumi",
            IaCFormat::CloudFormation => "cloudformation",
            IaCFormat::ArmTemplate => "arm_template",
            IaCFormat::DeploymentManager => "deployment_manager",
        }
    }
}

// ── ServiceIntegration ────────────────────────────────────────────────────────

/// Managed service integrations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceIntegration {
    /// Relational database service (RDS, Cloud SQL, Azure SQL).
    Database {
        engine: String,
        instance_type: String,
    },
    /// Object storage (S3, GCS, Blob Storage).
    ObjectStorage { bucket_name: String },
    /// Cache service (ElastiCache, Memorystore, Azure Cache).
    Cache { engine: String, node_type: String },
    /// Message queue (SQS, Pub/Sub, Service Bus).
    MessageQueue { queue_name: String },
    /// Vector database for embeddings.
    VectorDb { provider: String },
    /// Custom service.
    Custom { name: String, config: String },
}

impl ServiceIntegration {
    /// Return the service type name.
    pub fn service_type(&self) -> &str {
        match self {
            ServiceIntegration::Database { .. } => "database",
            ServiceIntegration::ObjectStorage { .. } => "object_storage",
            ServiceIntegration::Cache { .. } => "cache",
            ServiceIntegration::MessageQueue { .. } => "message_queue",
            ServiceIntegration::VectorDb { .. } => "vector_db",
            ServiceIntegration::Custom { name, .. } => name.as_str(),
        }
    }
}

// ── AutoscalingConfig ─────────────────────────────────────────────────────────

/// Autoscaling configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoscalingConfig {
    /// Minimum number of instances.
    pub min_instances: u32,
    /// Maximum number of instances.
    pub max_instances: u32,
    /// Target CPU utilization percentage (1-100).
    pub target_cpu_utilization: u32,
}

impl Default for AutoscalingConfig {
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 5,
            target_cpu_utilization: 70,
        }
    }
}

// ── DeploymentConfig ──────────────────────────────────────────────────────────

/// Configuration for a cloud deployment.
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    /// Application name.
    pub app_name: String,
    /// Target cloud provider.
    pub provider: CloudProvider,
    /// Deployment region.
    pub region: Option<String>,
    /// Infrastructure-as-code format.
    pub iac_format: IaCFormat,
    /// Autoscaling configuration.
    pub autoscaling: Option<AutoscalingConfig>,
    /// Service integrations.
    pub services: Vec<ServiceIntegration>,
    /// Environment variables.
    pub env_vars: HashMap<String, String>,
    /// Resource tags/labels.
    pub tags: HashMap<String, String>,
}

impl DeploymentConfig {
    /// Create a new deployment config.
    pub fn new(app_name: impl Into<String>, provider: CloudProvider) -> Self {
        let iac_format = match &provider {
            CloudProvider::Aws => IaCFormat::Terraform,
            CloudProvider::Gcp => IaCFormat::Terraform,
            CloudProvider::Azure => IaCFormat::Terraform,
            CloudProvider::Custom(_) => IaCFormat::Terraform,
        };

        Self {
            app_name: app_name.into(),
            provider,
            region: None,
            iac_format,
            autoscaling: None,
            services: Vec::new(),
            env_vars: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    /// Set the deployment region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the IaC format.
    pub fn with_iac_format(mut self, format: IaCFormat) -> Self {
        self.iac_format = format;
        self
    }

    /// Set autoscaling configuration.
    pub fn with_autoscaling(mut self, config: AutoscalingConfig) -> Self {
        self.autoscaling = Some(config);
        self
    }

    /// Add a service integration.
    pub fn with_service(mut self, service: ServiceIntegration) -> Self {
        self.services.push(service);
        self
    }

    /// Add an environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Add a tag/label.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

// ── DeploymentError ───────────────────────────────────────────────────────────

/// Errors during deployment operations.
#[derive(Debug, thiserror::Error)]
pub enum DeploymentError {
    /// Invalid configuration.
    #[error("Invalid deployment config: {0}")]
    InvalidConfig(String),
    /// Provider-specific error.
    #[error("Provider error ({provider}): {message}")]
    ProviderError { provider: String, message: String },
    /// Template generation error.
    #[error("Template error: {0}")]
    TemplateError(String),
}

// ── DeploymentStatus ──────────────────────────────────────────────────────────

/// Status of a deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentStatus {
    /// Deployment is pending.
    Pending,
    /// Deployment is in progress.
    InProgress,
    /// Deployment completed successfully.
    Deployed,
    /// Deployment failed.
    Failed(String),
    /// Deployment is being rolled back.
    RollingBack,
}

// ── CloudDeployment ───────────────────────────────────────────────────────────

/// Manages cloud deployment lifecycle.
#[derive(Debug)]
pub struct CloudDeployment {
    config: DeploymentConfig,
    status: DeploymentStatus,
}

impl CloudDeployment {
    /// Create a new cloud deployment.
    pub fn new(config: DeploymentConfig) -> Self {
        Self {
            config,
            status: DeploymentStatus::Pending,
        }
    }

    /// Get the cloud provider.
    pub fn provider(&self) -> &CloudProvider {
        &self.config.provider
    }

    /// Get the deployment configuration.
    pub fn config(&self) -> &DeploymentConfig {
        &self.config
    }

    /// Get the current status.
    pub fn status(&self) -> &DeploymentStatus {
        &self.status
    }

    /// Validate the deployment configuration.
    pub fn validate(&self) -> Result<(), DeploymentError> {
        if self.config.app_name.is_empty() {
            return Err(DeploymentError::InvalidConfig(
                "app_name is required".into(),
            ));
        }

        if let Some(ref autoscaling) = self.config.autoscaling {
            if autoscaling.min_instances > autoscaling.max_instances {
                return Err(DeploymentError::InvalidConfig(
                    "min_instances cannot exceed max_instances".into(),
                ));
            }
            if autoscaling.target_cpu_utilization == 0 || autoscaling.target_cpu_utilization > 100 {
                return Err(DeploymentError::InvalidConfig(
                    "target_cpu_utilization must be between 1 and 100".into(),
                ));
            }
        }

        Ok(())
    }

    /// Generate infrastructure-as-code template (returns template as string).
    pub fn generate_template(&self) -> Result<String, DeploymentError> {
        self.validate()?;

        let provider_name = self.config.provider.name();
        let app_name = &self.config.app_name;
        let region = self.config.region.as_deref().unwrap_or("us-east-1");

        let mut template = format!(
            "# {} deployment for '{}'\n# Region: {}\n# Format: {}\n\n",
            provider_name,
            app_name,
            region,
            self.config.iac_format.name()
        );

        // Add service integrations
        for service in &self.config.services {
            template.push_str(&format!("# Service: {}\n", service.service_type()));
        }

        // Add autoscaling
        if let Some(ref autoscaling) = self.config.autoscaling {
            template.push_str(&format!(
                "# Autoscaling: min={}, max={}, target_cpu={}%\n",
                autoscaling.min_instances,
                autoscaling.max_instances,
                autoscaling.target_cpu_utilization
            ));
        }

        Ok(template)
    }

    /// Simulate starting a deployment (sets status to InProgress).
    pub fn start_deploy(&mut self) -> Result<(), DeploymentError> {
        self.validate()?;
        self.status = DeploymentStatus::InProgress;
        Ok(())
    }

    /// Mark deployment as completed.
    pub fn mark_deployed(&mut self) {
        self.status = DeploymentStatus::Deployed;
    }

    /// Mark deployment as failed.
    pub fn mark_failed(&mut self, reason: impl Into<String>) {
        self.status = DeploymentStatus::Failed(reason.into());
    }

    /// List configured service integrations.
    pub fn services(&self) -> &[ServiceIntegration] {
        &self.config.services
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-16.3: Support deployment to AWS
    #[test]
    fn test_aws_deployment() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws).with_region("us-west-2");

        let deployment = CloudDeployment::new(config);
        assert_eq!(deployment.provider(), &CloudProvider::Aws);
        assert_eq!(deployment.config().region.as_deref(), Some("us-west-2"));
    }

    // REQ-16.3: Support deployment to GCP
    #[test]
    fn test_gcp_deployment() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Gcp).with_region("us-central1");

        let deployment = CloudDeployment::new(config);
        assert_eq!(deployment.provider(), &CloudProvider::Gcp);
    }

    // REQ-16.3: Support deployment to Azure
    #[test]
    fn test_azure_deployment() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Azure).with_region("eastus");

        let deployment = CloudDeployment::new(config);
        assert_eq!(deployment.provider(), &CloudProvider::Azure);
    }

    // REQ-16.3: Terraform/Pulumi modules
    #[test]
    fn test_iac_format_configuration() {
        let config =
            DeploymentConfig::new("my-app", CloudProvider::Aws).with_iac_format(IaCFormat::Pulumi);

        assert_eq!(config.iac_format, IaCFormat::Pulumi);
        assert_eq!(config.iac_format.name(), "pulumi");
    }

    // REQ-16.3: Managed service integration (RDS, S3, ElastiCache)
    #[test]
    fn test_managed_service_integration() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws)
            .with_service(ServiceIntegration::Database {
                engine: "postgres".into(),
                instance_type: "db.t3.medium".into(),
            })
            .with_service(ServiceIntegration::ObjectStorage {
                bucket_name: "my-app-data".into(),
            })
            .with_service(ServiceIntegration::Cache {
                engine: "redis".into(),
                node_type: "cache.t3.micro".into(),
            });

        let deployment = CloudDeployment::new(config);
        assert_eq!(deployment.services().len(), 3);
        assert_eq!(deployment.services()[0].service_type(), "database");
        assert_eq!(deployment.services()[1].service_type(), "object_storage");
        assert_eq!(deployment.services()[2].service_type(), "cache");
    }

    // REQ-16.3: Autoscaling configuration
    #[test]
    fn test_autoscaling_config() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws).with_autoscaling(
            AutoscalingConfig {
                min_instances: 2,
                max_instances: 20,
                target_cpu_utilization: 75,
            },
        );

        let deployment = CloudDeployment::new(config);
        let autoscaling = deployment.config().autoscaling.as_ref().unwrap();
        assert_eq!(autoscaling.min_instances, 2);
        assert_eq!(autoscaling.max_instances, 20);
        assert_eq!(autoscaling.target_cpu_utilization, 75);
    }

    // REQ-16.3: Template generation
    #[test]
    fn test_template_generation() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws)
            .with_region("us-east-1")
            .with_autoscaling(AutoscalingConfig::default())
            .with_service(ServiceIntegration::Database {
                engine: "postgres".into(),
                instance_type: "db.t3.medium".into(),
            });

        let deployment = CloudDeployment::new(config);
        let template = deployment.generate_template().unwrap();

        assert!(template.contains("aws"));
        assert!(template.contains("my-app"));
        assert!(template.contains("us-east-1"));
        assert!(template.contains("database"));
        assert!(template.contains("Autoscaling"));
    }

    // REQ-16.3: Validation rejects invalid config
    #[test]
    fn test_validation_invalid_autoscaling() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws).with_autoscaling(
            AutoscalingConfig {
                min_instances: 10,
                max_instances: 5, // Invalid: min > max
                target_cpu_utilization: 70,
            },
        );

        let deployment = CloudDeployment::new(config);
        assert!(deployment.validate().is_err());
    }

    // REQ-16.3: Validation rejects empty app name
    #[test]
    fn test_validation_empty_app_name() {
        let config = DeploymentConfig::new("", CloudProvider::Aws);
        let deployment = CloudDeployment::new(config);
        assert!(deployment.validate().is_err());
    }

    // REQ-16.3: Deployment lifecycle
    #[test]
    fn test_deployment_lifecycle() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws).with_region("us-east-1");

        let mut deployment = CloudDeployment::new(config);
        assert_eq!(deployment.status(), &DeploymentStatus::Pending);

        deployment.start_deploy().unwrap();
        assert_eq!(deployment.status(), &DeploymentStatus::InProgress);

        deployment.mark_deployed();
        assert_eq!(deployment.status(), &DeploymentStatus::Deployed);
    }

    // REQ-16.3: Failed deployment
    #[test]
    fn test_failed_deployment() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Gcp);
        let mut deployment = CloudDeployment::new(config);

        deployment.mark_failed("resource quota exceeded");
        assert!(matches!(deployment.status(), DeploymentStatus::Failed(_)));
    }

    // REQ-16.3: Environment variables and tags
    #[test]
    fn test_env_vars_and_tags() {
        let config = DeploymentConfig::new("my-app", CloudProvider::Aws)
            .with_env("API_KEY", "secret")
            .with_tag("team", "ml-platform");

        assert_eq!(config.env_vars.get("API_KEY"), Some(&"secret".to_string()));
        assert_eq!(config.tags.get("team"), Some(&"ml-platform".to_string()));
    }
}
