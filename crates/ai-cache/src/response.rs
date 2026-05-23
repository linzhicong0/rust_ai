use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ai_core::error::ProviderError;
use ai_core::provider::Provider;
use ai_core::tool::ToolDescriptor;
use ai_core::types::{CompletionResponse, Message, ModelConfig, StreamChunk};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tracing::debug;

/// Configuration for the in-memory response cache.
#[derive(Debug, Clone)]
pub struct ResponseCacheConfig {
	/// Maximum number of entries stored at once.
	pub max_entries: usize,

	/// Time-to-live for each cache entry.
	pub ttl: Duration,
}

impl ResponseCacheConfig {
	/// Create a new cache configuration.
	pub fn new(max_entries: usize, ttl: Duration) -> Self {
		Self { max_entries, ttl }
	}

	/// Override the maximum number of cached entries.
	pub fn with_max_entries(mut self, max_entries: usize) -> Self {
		self.max_entries = max_entries;
		self
	}

	/// Override the cache TTL.
	pub fn with_ttl(mut self, ttl: Duration) -> Self {
		self.ttl = ttl;
		self
	}
}

impl Default for ResponseCacheConfig {
	fn default() -> Self {
		Self {
			max_entries: 1_024,
			ttl: Duration::from_secs(300),
		}
	}
}

/// Strategy for building deterministic response cache keys.
pub trait CacheKeyBuilder: Send + Sync {
	/// Build a cache key from a request envelope.
	fn build_key(
		&self,
		provider_name: &str,
		messages: &[Message],
		config: &ModelConfig,
		tools: &[ToolDescriptor],
	) -> String;
}

/// Configurable default cache key builder.
#[derive(Debug, Clone)]
pub struct ResponseCacheKeyBuilder {
	include_provider: bool,
	include_messages: bool,
	include_model: bool,
	include_config: bool,
	include_tools: bool,
}

impl Default for ResponseCacheKeyBuilder {
	fn default() -> Self {
		Self {
			include_provider: true,
			include_messages: true,
			include_model: true,
			include_config: true,
			include_tools: true,
		}
	}
}

impl ResponseCacheKeyBuilder {
	/// Create the default key builder.
	pub fn new() -> Self {
		Self::default()
	}

	/// Control whether the provider name is part of the key.
	pub fn include_provider(mut self, include_provider: bool) -> Self {
		self.include_provider = include_provider;
		self
	}

	/// Control whether the message list is part of the key.
	pub fn include_messages(mut self, include_messages: bool) -> Self {
		self.include_messages = include_messages;
		self
	}

	/// Control whether the model name is part of the key.
	pub fn include_model(mut self, include_model: bool) -> Self {
		self.include_model = include_model;
		self
	}

	/// Control whether generation parameters are part of the key.
	pub fn include_config(mut self, include_config: bool) -> Self {
		self.include_config = include_config;
		self
	}

	/// Control whether tool descriptors are part of the key.
	pub fn include_tools(mut self, include_tools: bool) -> Self {
		self.include_tools = include_tools;
		self
	}

	fn tool_value(descriptor: &ToolDescriptor) -> Value {
		json!({
			"name": descriptor.name,
			"description": descriptor.description,
			"input_schema": descriptor.input_schema,
			"output_schema": descriptor.output_schema,
		})
	}
}

impl CacheKeyBuilder for ResponseCacheKeyBuilder {
	fn build_key(
		&self,
		provider_name: &str,
		messages: &[Message],
		config: &ModelConfig,
		tools: &[ToolDescriptor],
	) -> String {
		let envelope = json!({
			"provider": self.include_provider.then_some(provider_name),
			"messages": self.include_messages.then_some(messages),
			"model": self.include_model.then_some(config.model.clone()),
			"config": self.include_config.then_some(json!({
				"temperature": config.temperature,
				"max_tokens": config.max_tokens,
				"top_p": config.top_p,
				"frequency_penalty": config.frequency_penalty,
				"presence_penalty": config.presence_penalty,
				"stop_sequences": config.stop_sequences,
			})),
			"tools": self.include_tools.then_some(
				tools.iter().map(Self::tool_value).collect::<Vec<_>>()
			),
		});

		let serialized = serde_json::to_string(&envelope)
			.expect("cache key envelope should serialize deterministically");
		let mut hasher = std::collections::hash_map::DefaultHasher::new();
		serialized.hash(&mut hasher);
		format!("{:016x}", hasher.finish())
	}
}

#[derive(Debug, Clone)]
struct CacheEntry {
	response: CompletionResponse,
	inserted_at: Instant,
}

#[derive(Debug)]
struct ResponseCacheState {
	entries: HashMap<String, CacheEntry>,
	lru_order: VecDeque<String>,
}

impl ResponseCacheState {
	fn new() -> Self {
		Self {
			entries: HashMap::new(),
			lru_order: VecDeque::new(),
		}
	}

	fn touch(&mut self, key: &str) {
		self.lru_order.retain(|existing| existing != key);
		self.lru_order.push_back(key.to_string());
	}

	fn remove(&mut self, key: &str) {
		self.entries.remove(key);
		self.lru_order.retain(|existing| existing != key);
	}
}

/// In-memory response cache with TTL and LRU eviction.
#[derive(Clone)]
pub struct ResponseCache {
	config: ResponseCacheConfig,
	key_builder: Arc<dyn CacheKeyBuilder>,
	inner: Arc<RwLock<ResponseCacheState>>,
}

impl std::fmt::Debug for ResponseCache {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ResponseCache")
			.field("config", &self.config)
			.finish_non_exhaustive()
	}
}

impl Default for ResponseCache {
	fn default() -> Self {
		Self::new(ResponseCacheConfig::default())
	}
}

impl ResponseCache {
	/// Create a new in-memory response cache.
	pub fn new(config: ResponseCacheConfig) -> Self {
		Self {
			config,
			key_builder: Arc::new(ResponseCacheKeyBuilder::default()),
			inner: Arc::new(RwLock::new(ResponseCacheState::new())),
		}
	}

	/// Override the cache key builder.
	pub fn with_key_builder(mut self, key_builder: impl CacheKeyBuilder + 'static) -> Self {
		self.key_builder = Arc::new(key_builder);
		self
	}

	/// Build the cache key for a request.
	pub fn build_key(
		&self,
		provider_name: &str,
		messages: &[Message],
		config: &ModelConfig,
		tools: &[ToolDescriptor],
	) -> String {
		self.key_builder
			.build_key(provider_name, messages, config, tools)
	}

	/// Get a cached response if it exists and has not expired.
	pub async fn get(&self, key: &str) -> Option<CompletionResponse> {
		let mut state = self.inner.write().await;

		let expired = state
			.entries
			.get(key)
			.map(|entry| entry.inserted_at.elapsed() > self.config.ttl)
			.unwrap_or(false);

		if expired {
			state.remove(key);
			return None;
		}

		let response = state.entries.get(key).map(|entry| entry.response.clone());
		if response.is_some() {
			state.touch(key);
		}
		response
	}

	/// Insert or update a cached response.
	pub async fn insert(&self, key: String, response: CompletionResponse) {
		let mut state = self.inner.write().await;

		state.entries.insert(
			key.clone(),
			CacheEntry {
				response,
				inserted_at: Instant::now(),
			},
		);
		state.touch(&key);

		while state.entries.len() > self.config.max_entries {
			if let Some(oldest) = state.lru_order.pop_front() {
				state.entries.remove(&oldest);
			} else {
				break;
			}
		}
	}

	/// Return whether a cache entry exists and is still valid.
	pub async fn contains(&self, key: &str) -> bool {
		self.get(key).await.is_some()
	}

	/// Return the current number of valid entries.
	pub async fn len(&self) -> usize {
		self.prune_expired().await;
		let state = self.inner.read().await;
		state.entries.len()
	}

	/// Clear all cached responses.
	pub async fn clear(&self) {
		let mut state = self.inner.write().await;
		state.entries.clear();
		state.lru_order.clear();
	}

	async fn prune_expired(&self) {
		let mut state = self.inner.write().await;
		let expired: Vec<String> = state
			.entries
			.iter()
			.filter_map(|(key, entry)| {
				(entry.inserted_at.elapsed() > self.config.ttl).then_some(key.clone())
			})
			.collect();

		for key in expired {
			state.remove(&key);
		}
	}
}

/// Provider wrapper that caches completion responses.
#[derive(Debug, Clone)]
pub struct CachedProvider<P> {
	provider: Arc<P>,
	cache: ResponseCache,
}

impl<P> CachedProvider<P>
where
	P: Provider,
{
	/// Wrap a provider with the default in-memory response cache.
	pub fn new(provider: P) -> Self {
		Self::with_cache(provider, ResponseCache::default())
	}

	/// Wrap a provider with a custom response cache.
	pub fn with_cache(provider: P, cache: ResponseCache) -> Self {
		Self {
			provider: Arc::new(provider),
			cache,
		}
	}

	/// Get the underlying response cache.
	pub fn cache(&self) -> &ResponseCache {
		&self.cache
	}

	/// Get the inner provider.
	pub fn provider(&self) -> &P {
		&self.provider
	}
}

#[async_trait]
impl<P> Provider for CachedProvider<P>
where
	P: Provider + 'static,
{
	async fn complete(
		&self,
		messages: Vec<Message>,
		config: &ModelConfig,
		tools: &[ToolDescriptor],
	) -> Result<CompletionResponse, ProviderError> {
		let key = self
			.cache
			.build_key(self.provider.name(), &messages, config, tools);

		if let Some(cached) = self.cache.get(&key).await {
			debug!(cache_key = %key, provider = %self.provider.name(), "Cache hit");
			return Ok(cached);
		}

		debug!(cache_key = %key, provider = %self.provider.name(), "Cache miss");
		let response = self.provider.complete(messages, config, tools).await?;
		self.cache.insert(key, response.clone()).await;
		Ok(response)
	}

	fn stream(
		&self,
		messages: Vec<Message>,
		config: &ModelConfig,
		tools: &[ToolDescriptor],
	) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
		self.provider.stream(messages, config, tools)
	}

	async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
		self.provider.embed(texts).await
	}

	fn name(&self) -> &str {
		self.provider.name()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use ai_core::types::{FinishReason, Usage};
	use std::sync::atomic::{AtomicUsize, Ordering};

	#[derive(Debug)]
	struct CountingProvider {
		calls: AtomicUsize,
	}

	impl CountingProvider {
		fn new() -> Self {
			Self {
				calls: AtomicUsize::new(0),
			}
		}

		fn calls(&self) -> usize {
			self.calls.load(Ordering::SeqCst)
		}
	}

	#[async_trait]
	impl Provider for CountingProvider {
		async fn complete(
			&self,
			messages: Vec<Message>,
			_config: &ModelConfig,
			_tools: &[ToolDescriptor],
		) -> Result<CompletionResponse, ProviderError> {
			let call_number = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
			let content = messages[0].content.as_text().unwrap_or_default();
			Ok(CompletionResponse {
				content: format!("{}:{}", content, call_number),
				tool_calls: vec![],
				usage: Usage {
					prompt_tokens: 10,
					completion_tokens: 5,
					total_tokens: 15,
				},
				finish_reason: FinishReason::Stop,
			})
		}

		fn stream(
			&self,
			_messages: Vec<Message>,
			_config: &ModelConfig,
			_tools: &[ToolDescriptor],
		) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
			Box::pin(futures::stream::empty())
		}

		async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
			Ok(texts.into_iter().map(|_| vec![0.0; 3]).collect())
		}

		fn name(&self) -> &str {
			"counting"
		}
	}

	#[tokio::test]
	async fn test_cache_key_changes_with_prompt_and_config() {
		let cache = ResponseCache::default();
		let key1 = cache.build_key(
			"mock",
			&[Message::user("hello")],
			&ModelConfig::new("gpt-4"),
			&[],
		);
		let key2 = cache.build_key(
			"mock",
			&[Message::user("hello")],
			&ModelConfig::new("gpt-4").with_temperature(0.9),
			&[],
		);
		let key3 = cache.build_key(
			"mock",
			&[Message::user("goodbye")],
			&ModelConfig::new("gpt-4"),
			&[],
		);

		assert_ne!(key1, key2);
		assert_ne!(key1, key3);
	}

	#[tokio::test]
	async fn test_identical_request_within_ttl_hits_cache() {
		let provider = CachedProvider::new(CountingProvider::new());
		let messages = vec![Message::user("hello")];
		let config = ModelConfig::new("gpt-4");

		let first = provider.complete(messages.clone(), &config, &[]).await.unwrap();
		let second = provider.complete(messages, &config, &[]).await.unwrap();

		assert_eq!(first.content, second.content);
		assert_eq!(provider.provider().calls(), 1);
	}

	#[tokio::test]
	async fn test_different_prompt_is_cache_miss() {
		let provider = CachedProvider::new(CountingProvider::new());
		let config = ModelConfig::new("gpt-4");

		let first = provider
			.complete(vec![Message::user("hello")], &config, &[])
			.await
			.unwrap();
		let second = provider
			.complete(vec![Message::user("goodbye")], &config, &[])
			.await
			.unwrap();

		assert_ne!(first.content, second.content);
		assert_eq!(provider.provider().calls(), 2);
	}

	#[tokio::test]
	async fn test_expired_ttl_is_cache_miss() {
		let cache = ResponseCache::new(ResponseCacheConfig::new(32, Duration::from_millis(5)));
		let provider = CachedProvider::with_cache(CountingProvider::new(), cache);
		let config = ModelConfig::new("gpt-4");
		let messages = vec![Message::user("hello")];

		let first = provider.complete(messages.clone(), &config, &[]).await.unwrap();
		tokio::time::sleep(Duration::from_millis(10)).await;
		let second = provider.complete(messages, &config, &[]).await.unwrap();

		assert_ne!(first.content, second.content);
		assert_eq!(provider.provider().calls(), 2);
	}

	#[tokio::test]
	async fn test_max_size_evicts_least_recently_used_entry() {
		let cache = ResponseCache::new(ResponseCacheConfig::new(2, Duration::from_secs(60)));
		let key_a = cache.build_key("mock", &[Message::user("a")], &ModelConfig::new("gpt-4"), &[]);
		let key_b = cache.build_key("mock", &[Message::user("b")], &ModelConfig::new("gpt-4"), &[]);
		let key_c = cache.build_key("mock", &[Message::user("c")], &ModelConfig::new("gpt-4"), &[]);

		let response = |content: &str| CompletionResponse {
			content: content.to_string(),
			tool_calls: vec![],
			usage: Usage {
				prompt_tokens: 1,
				completion_tokens: 1,
				total_tokens: 2,
			},
			finish_reason: FinishReason::Stop,
		};

		cache.insert(key_a.clone(), response("a")).await;
		cache.insert(key_b.clone(), response("b")).await;
		assert!(cache.get(&key_a).await.is_some());
		cache.insert(key_c.clone(), response("c")).await;

		assert!(cache.get(&key_a).await.is_some());
		assert!(cache.get(&key_c).await.is_some());
		assert!(cache.get(&key_b).await.is_none());
		assert_eq!(cache.len().await, 2);
	}
}
