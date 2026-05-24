//! Plugin system for extending request/response processing.
//!
//! Plugins can participate in framework lifecycle events, mutate requests
//! before provider execution, and inspect or transform responses afterwards.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::types::{Message, Usage};

/// Context shared with plugins for the current request/response lifecycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PluginContext {
    /// Arbitrary metadata associated with the current processing lifecycle.
    pub metadata: HashMap<String, Value>,
}

impl PluginContext {
    /// Create an empty plugin context.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A request being processed by the plugin pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRequest {
    /// Target model identifier.
    pub model: String,

    /// Conversation messages being sent to the model.
    pub messages: Vec<Message>,

    /// Plugin-specific metadata associated with this request.
    pub metadata: HashMap<String, Value>,
}

impl PluginRequest {
    /// Create a new plugin request.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            metadata: HashMap::new(),
        }
    }
}

/// A response being processed by the plugin pipeline.
#[derive(Debug, Clone)]
pub struct PluginResponse {
    /// Text content returned by the model.
    pub content: String,

    /// Token usage associated with the response.
    pub usage: Usage,

    /// Plugin-specific metadata associated with this response.
    pub metadata: HashMap<String, Value>,
}

impl PluginResponse {
    /// Create a new plugin response.
    pub fn new(content: impl Into<String>, usage: Usage) -> Self {
        Self {
            content: content.into(),
            usage,
            metadata: HashMap::new(),
        }
    }
}

/// Errors produced by plugins or plugin discovery/registration.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// A plugin reported a lifecycle failure.
    #[error("plugin lifecycle error: {0}")]
    Lifecycle(String),

    /// Plugin discovery failed due to invalid configuration or layout.
    #[error("plugin discovery error: {0}")]
    Discovery(String),

    /// File-system level error while discovering plugins.
    #[error("plugin I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to serialize or deserialize plugin metadata.
    #[error("plugin serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// A framework plugin.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Stable plugin name.
    fn name(&self) -> &str;

    /// Semantic version of the plugin implementation.
    fn version(&self) -> &str;

    /// Called when the plugin is loaded into the registry.
    async fn on_load(&mut self) -> Result<(), PluginError>;

    /// Called before a request is sent to a provider.
    async fn on_request(&self, request: &mut PluginRequest) -> Result<(), PluginError>;

    /// Called after a response is produced by a provider.
    async fn on_response(&self, response: &mut PluginResponse) -> Result<(), PluginError>;

    /// Called when the plugin is removed or the registry shuts down.
    async fn on_unload(&mut self) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Registry that owns and executes plugins.
#[derive(Default)]
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
    plugin_directories: Vec<PathBuf>,
}

impl PluginRegistry {
    /// Create an empty plugin registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin instance.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether the registry contains no plugins.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Register a directory to scan for plugins during startup discovery.
    pub fn add_plugin_directory<P: Into<PathBuf>>(&mut self, directory: P) {
        self.plugin_directories.push(directory.into());
    }

    /// Configure multiple directories to scan for plugins.
    pub fn with_plugin_directories<I, P>(mut self, directories: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.plugin_directories = directories.into_iter().map(Into::into).collect();
        self
    }

    /// Returns the configured plugin directories.
    pub fn plugin_directories(&self) -> &[PathBuf] {
        &self.plugin_directories
    }

    /// Discover plugin candidates from configured directories.
    ///
    /// Discovery is path-based so callers can decide how discovered entries map
    /// to concrete plugin implementations.
    pub fn discover_plugins(&self) -> Result<Vec<PathBuf>, PluginError> {
        let mut discovered = Vec::new();
        let mut seen = HashSet::new();

        for directory in &self.plugin_directories {
            if !directory.exists() {
                return Err(PluginError::Discovery(format!(
                    "configured plugin directory does not exist: {}",
                    directory.display()
                )));
            }

            if !directory.is_dir() {
                return Err(PluginError::Discovery(format!(
                    "configured plugin path is not a directory: {}",
                    directory.display()
                )));
            }

            for entry in std::fs::read_dir(directory)? {
                let path = entry?.path();
                if (path.is_file() || path.is_dir()) && seen.insert(path.clone()) {
                    discovered.push(path);
                }
            }
        }

        discovered.sort();
        Ok(discovered)
    }

    /// Discover and register plugins using a caller-provided loader.
    pub async fn discover_and_register<F, Fut>(
        &mut self,
        mut loader: F,
    ) -> Result<usize, PluginError>
    where
        F: FnMut(&Path) -> Fut,
        Fut: Future<Output = Result<Box<dyn Plugin>, PluginError>>,
    {
        let discovered = self.discover_plugins()?;
        let count = discovered.len();

        for path in discovered {
            let plugin = loader(&path).await?;
            self.register(plugin);
        }

        Ok(count)
    }

    /// Load all registered plugins.
    pub async fn load_all(&mut self) -> Result<(), PluginError> {
        for plugin in &mut self.plugins {
            debug!(
                plugin = plugin.name(),
                version = plugin.version(),
                "loading plugin"
            );
            plugin.on_load().await?;
        }

        Ok(())
    }

    /// Process a request through all plugins in registration order.
    pub async fn process_request(&self, request: &mut PluginRequest) -> Result<(), PluginError> {
        for plugin in &self.plugins {
            debug!(
                plugin = plugin.name(),
                version = plugin.version(),
                "processing request"
            );
            plugin.on_request(request).await?;
        }

        Ok(())
    }

    /// Process a response through all plugins in registration order.
    pub async fn process_response(&self, response: &mut PluginResponse) -> Result<(), PluginError> {
        for plugin in &self.plugins {
            debug!(
                plugin = plugin.name(),
                version = plugin.version(),
                "processing response"
            );
            plugin.on_response(response).await?;
        }

        Ok(())
    }

    /// Unload all registered plugins in reverse registration order.
    pub async fn unload_all(&mut self) -> Result<(), PluginError> {
        for plugin in self.plugins.iter_mut().rev() {
            debug!(
                plugin = plugin.name(),
                version = plugin.version(),
                "unloading plugin"
            );
            plugin.on_unload().await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use crate::types::{Content, Role};

    #[derive(Default)]
    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(test_name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after unix epoch")
                .as_nanos();
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("plugin-tests")
                .join(format!("{test_name}-{unique}"));
            std::fs::create_dir_all(&path).expect("test directory should be created");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            if self.path.exists() {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    struct RecordingPlugin {
        name: &'static str,
        version: &'static str,
        events: Arc<Mutex<Vec<String>>>,
        fail_on_load: bool,
        fail_on_request: bool,
        fail_on_response: bool,
        fail_on_unload: bool,
    }

    impl RecordingPlugin {
        fn new(name: &'static str, events: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                name,
                version: "1.0.0",
                events,
                fail_on_load: false,
                fail_on_request: false,
                fail_on_response: false,
                fail_on_unload: false,
            }
        }

        fn with_load_failure(mut self) -> Self {
            self.fail_on_load = true;
            self
        }

        fn with_request_failure(mut self) -> Self {
            self.fail_on_request = true;
            self
        }

        fn with_response_failure(mut self) -> Self {
            self.fail_on_response = true;
            self
        }

        fn with_unload_failure(mut self) -> Self {
            self.fail_on_unload = true;
            self
        }
    }

    #[async_trait]
    impl Plugin for RecordingPlugin {
        fn name(&self) -> &str {
            self.name
        }

        fn version(&self) -> &str {
            self.version
        }

        async fn on_load(&mut self) -> Result<(), PluginError> {
            self.events
                .lock()
                .expect("events should be lockable")
                .push(format!("load:{}", self.name));

            if self.fail_on_load {
                return Err(PluginError::Lifecycle(format!(
                    "{} failed to load",
                    self.name
                )));
            }

            Ok(())
        }

        async fn on_request(&self, request: &mut PluginRequest) -> Result<(), PluginError> {
            self.events
                .lock()
                .expect("events should be lockable")
                .push(format!("request:{}", self.name));

            if self.fail_on_request {
                return Err(PluginError::Lifecycle(format!(
                    "{} failed to process request",
                    self.name
                )));
            }

            request.model.push_str(&format!("|{}", self.name));
            request
                .metadata
                .insert(self.name.to_string(), json!("request-processed"));
            Ok(())
        }

        async fn on_response(&self, response: &mut PluginResponse) -> Result<(), PluginError> {
            self.events
                .lock()
                .expect("events should be lockable")
                .push(format!("response:{}", self.name));

            if self.fail_on_response {
                return Err(PluginError::Lifecycle(format!(
                    "{} failed to process response",
                    self.name
                )));
            }

            response.content.push_str(&format!("|{}", self.name));
            response
                .metadata
                .insert(self.name.to_string(), json!("response-processed"));
            Ok(())
        }

        async fn on_unload(&mut self) -> Result<(), PluginError> {
            self.events
                .lock()
                .expect("events should be lockable")
                .push(format!("unload:{}", self.name));

            if self.fail_on_unload {
                return Err(PluginError::Lifecycle(format!(
                    "{} failed to unload",
                    self.name
                )));
            }

            Ok(())
        }
    }

    struct MinimalPlugin;

    #[async_trait]
    impl Plugin for MinimalPlugin {
        fn name(&self) -> &str {
            "minimal"
        }

        fn version(&self) -> &str {
            "0.1.0"
        }

        async fn on_load(&mut self) -> Result<(), PluginError> {
            Ok(())
        }

        async fn on_request(&self, _request: &mut PluginRequest) -> Result<(), PluginError> {
            Ok(())
        }

        async fn on_response(&self, _response: &mut PluginResponse) -> Result<(), PluginError> {
            Ok(())
        }
    }

    fn sample_request() -> PluginRequest {
        PluginRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: Content::Text("hello".to_string()),
            }],
            metadata: HashMap::new(),
        }
    }

    fn sample_response() -> PluginResponse {
        PluginResponse {
            content: "hello back".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 12,
                total_tokens: 22,
            },
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_registry_loads_processes_and_unloads_plugins_in_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(RecordingPlugin::new("alpha", events.clone())));
        registry.register(Box::new(RecordingPlugin::new("beta", events.clone())));

        registry.load_all().await.expect("plugins should load");

        let mut request = sample_request();
        registry
            .process_request(&mut request)
            .await
            .expect("request should process");
        assert_eq!(request.model, "gpt-4o-mini|alpha|beta");
        assert_eq!(
            request.metadata.get("alpha"),
            Some(&json!("request-processed"))
        );
        assert_eq!(
            request.metadata.get("beta"),
            Some(&json!("request-processed"))
        );

        let mut response = sample_response();
        registry
            .process_response(&mut response)
            .await
            .expect("response should process");
        assert_eq!(response.content, "hello back|alpha|beta");
        assert_eq!(
            response.metadata.get("alpha"),
            Some(&json!("response-processed"))
        );
        assert_eq!(
            response.metadata.get("beta"),
            Some(&json!("response-processed"))
        );

        registry.unload_all().await.expect("plugins should unload");

        let recorded = events.lock().expect("events should be lockable").clone();
        assert_eq!(
            recorded,
            vec![
                "load:alpha",
                "load:beta",
                "request:alpha",
                "request:beta",
                "response:alpha",
                "response:beta",
                "unload:beta",
                "unload:alpha",
            ]
        );
    }

    #[tokio::test]
    async fn test_load_all_stops_on_first_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(RecordingPlugin::new("alpha", events.clone())));
        registry.register(Box::new(
            RecordingPlugin::new("beta", events.clone()).with_load_failure(),
        ));
        registry.register(Box::new(RecordingPlugin::new("gamma", events.clone())));

        let err = registry.load_all().await.expect_err("load should fail");
        assert!(matches!(err, PluginError::Lifecycle(message) if message == "beta failed to load"));

        let recorded = events.lock().expect("events should be lockable").clone();
        assert_eq!(recorded, vec!["load:alpha", "load:beta"]);
    }

    #[tokio::test]
    async fn test_process_request_stops_on_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(RecordingPlugin::new("alpha", events.clone())));
        registry.register(Box::new(
            RecordingPlugin::new("beta", events.clone()).with_request_failure(),
        ));
        registry.register(Box::new(RecordingPlugin::new("gamma", events.clone())));

        let mut request = sample_request();
        let err = registry
            .process_request(&mut request)
            .await
            .expect_err("request processing should fail");
        assert!(
            matches!(err, PluginError::Lifecycle(message) if message == "beta failed to process request")
        );
        assert_eq!(request.model, "gpt-4o-mini|alpha");
        assert!(request.metadata.contains_key("alpha"));
        assert!(!request.metadata.contains_key("beta"));
        assert!(!request.metadata.contains_key("gamma"));
    }

    #[tokio::test]
    async fn test_process_response_stops_on_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(RecordingPlugin::new("alpha", events.clone())));
        registry.register(Box::new(
            RecordingPlugin::new("beta", events.clone()).with_response_failure(),
        ));
        registry.register(Box::new(RecordingPlugin::new("gamma", events.clone())));

        let mut response = sample_response();
        let err = registry
            .process_response(&mut response)
            .await
            .expect_err("response processing should fail");
        assert!(
            matches!(err, PluginError::Lifecycle(message) if message == "beta failed to process response")
        );
        assert_eq!(response.content, "hello back|alpha");
        assert!(response.metadata.contains_key("alpha"));
        assert!(!response.metadata.contains_key("beta"));
        assert!(!response.metadata.contains_key("gamma"));
    }

    #[tokio::test]
    async fn test_unload_all_propagates_errors() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(RecordingPlugin::new("alpha", events.clone())));
        registry.register(Box::new(
            RecordingPlugin::new("beta", events.clone()).with_unload_failure(),
        ));

        let err = registry
            .unload_all()
            .await
            .expect_err("unload should fail for beta");
        assert!(
            matches!(err, PluginError::Lifecycle(message) if message == "beta failed to unload")
        );

        let recorded = events.lock().expect("events should be lockable").clone();
        assert_eq!(recorded, vec!["unload:beta"]);
    }

    #[tokio::test]
    async fn test_default_on_unload_is_noop() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(MinimalPlugin));

        registry.load_all().await.expect("plugin should load");
        registry.unload_all().await.expect("plugin should unload");
    }

    #[test]
    fn test_discover_plugins_from_configured_directories() {
        let dir_a = TestDir::new("discover-a");
        let dir_b = TestDir::new("discover-b");
        let plugin_a = dir_a.path().join("alpha.json");
        let plugin_b = dir_b.path().join("beta");
        std::fs::write(&plugin_a, "{}").expect("plugin manifest should be written");
        std::fs::create_dir_all(&plugin_b).expect("plugin directory should be created");

        let registry = PluginRegistry::new()
            .with_plugin_directories([dir_a.path().to_path_buf(), dir_b.path().to_path_buf()]);

        let discovered = registry
            .discover_plugins()
            .expect("plugins should be discovered");
        assert_eq!(discovered, vec![plugin_a, plugin_b]);
    }

    #[test]
    fn test_discover_plugins_errors_for_missing_directory() {
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("plugin-tests")
            .join("missing-directory");
        let registry = PluginRegistry::new().with_plugin_directories([missing.clone()]);

        let err = registry
            .discover_plugins()
            .expect_err("missing directory should error");
        assert!(
            matches!(err, PluginError::Discovery(message) if message.contains(&missing.display().to_string()))
        );
    }

    #[tokio::test]
    async fn test_discover_and_register_uses_loader() {
        let dir = TestDir::new("discover-register");
        let plugin_a = dir.path().join("alpha.json");
        let plugin_b = dir.path().join("beta.json");
        std::fs::write(&plugin_a, r#"{"name":"alpha"}"#).expect("alpha manifest should be written");
        std::fs::write(&plugin_b, r#"{"name":"beta"}"#).expect("beta manifest should be written");

        let mut registry =
            PluginRegistry::new().with_plugin_directories([dir.path().to_path_buf()]);
        let loaded = registry
            .discover_and_register(|path| {
                let path = path.to_path_buf();
                async move {
                    let manifest = std::fs::read_to_string(&path)?;
                    let value: Value = serde_json::from_str(&manifest)?;
                    let name = value["name"].as_str().ok_or_else(|| {
                        PluginError::Discovery("plugin manifest missing name".to_string())
                    })?;
                    Ok(Box::new(RecordingPlugin::new(
                        if name == "alpha" { "alpha" } else { "beta" },
                        Arc::new(Mutex::new(Vec::new())),
                    )) as Box<dyn Plugin>)
                }
            })
            .await
            .expect("plugins should register");

        assert_eq!(loaded, 2);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_plugin_data_structures() {
        let mut context = PluginContext::new();
        context
            .metadata
            .insert("request_id".to_string(), json!("req-1"));
        assert_eq!(context.metadata.get("request_id"), Some(&json!("req-1")));

        let request = PluginRequest::new("gpt-4o-mini", vec![Message::user("hello")]);
        assert_eq!(request.model, "gpt-4o-mini");
        assert_eq!(request.messages.len(), 1);

        let response = PluginResponse::new("done", Usage::default());
        assert_eq!(response.content, "done");
        assert_eq!(response.usage.total_tokens, 0);
    }
}
