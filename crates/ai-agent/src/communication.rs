// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Inter-agent communication through structured protocols.
//!
//! This module provides a [`MessageBus`] for agents to communicate via
//! request/response, publish/subscribe, and broadcast patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

/// A typed message envelope for inter-agent communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Unique message ID.
    pub id: String,

    /// Sender agent ID.
    pub from: String,

    /// Target agent ID (None for broadcast).
    pub to: Option<String>,

    /// Topic/channel for pub/sub routing.
    pub topic: Option<String>,

    /// The message payload as JSON.
    pub payload: serde_json::Value,

    /// Timestamp (ISO 8601).
    pub timestamp: String,
}

impl MessageEnvelope {
    /// Create a new directed message.
    pub fn new(from: impl Into<String>, to: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.into(),
            to: Some(to.into()),
            topic: None,
            payload,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a new broadcast message.
    pub fn broadcast(from: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.into(),
            to: None,
            topic: None,
            payload,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Create a new topic-based message.
    pub fn publish(
        from: impl Into<String>,
        topic: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.into(),
            to: None,
            topic: Some(topic.into()),
            payload,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// A message bus for inter-agent communication.
///
/// Supports:
/// - Direct messaging: `send()` to a specific agent
/// - Broadcasting: `broadcast()` to all agents
/// - Pub/Sub: `subscribe()` to a topic, `publish()` to a topic
pub struct MessageBus {
    /// Direct message channels per agent.
    agents: Arc<RwLock<HashMap<String, mpsc::Sender<MessageEnvelope>>>>,

    /// Broadcast channel.
    broadcast_tx: broadcast::Sender<MessageEnvelope>,

    /// Topic subscriptions: topic → list of subscriber agent IDs.
    subscriptions: Arc<RwLock<HashMap<String, Vec<String>>>>,

    /// Buffer size for agent channels.
    buffer_size: usize,
}

/// Handle returned when an agent registers with the bus.
pub struct AgentMailbox {
    /// The agent's ID.
    pub agent_id: String,

    /// Receiver for direct messages.
    pub receiver: mpsc::Receiver<MessageEnvelope>,

    /// Receiver for broadcast messages.
    pub broadcast_receiver: broadcast::Receiver<MessageEnvelope>,
}

impl MessageBus {
    /// Create a new message bus with a default buffer size.
    pub fn new() -> Self {
        Self::with_buffer_size(100)
    }

    /// Create a new message bus with a custom buffer size.
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        let (broadcast_tx, _) = broadcast::channel(buffer_size);
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            buffer_size,
        }
    }

    /// Register an agent with the bus and get a mailbox.
    pub async fn register(&self, agent_id: impl Into<String>) -> AgentMailbox {
        let agent_id = agent_id.into();
        let (tx, rx) = mpsc::channel(self.buffer_size);
        let broadcast_rx = self.broadcast_tx.subscribe();

        self.agents.write().await.insert(agent_id.clone(), tx);

        AgentMailbox {
            agent_id,
            receiver: rx,
            broadcast_receiver: broadcast_rx,
        }
    }

    /// Unregister an agent from the bus.
    pub async fn unregister(&self, agent_id: &str) {
        self.agents.write().await.remove(agent_id);

        // Remove from all subscriptions
        let mut subs = self.subscriptions.write().await;
        for subscribers in subs.values_mut() {
            subscribers.retain(|id| id != agent_id);
        }
    }

    /// Send a direct message to a specific agent.
    ///
    /// Returns `true` if the message was delivered, `false` if the agent is not registered.
    pub async fn send(&self, message: MessageEnvelope) -> bool {
        let target = match &message.to {
            Some(to) => to.clone(),
            None => return false,
        };

        let agents = self.agents.read().await;
        if let Some(tx) = agents.get(&target) {
            tx.send(message).await.is_ok()
        } else {
            false
        }
    }

    /// Broadcast a message to all registered agents.
    ///
    /// Returns the number of receivers that got the message.
    pub fn broadcast(&self, message: MessageEnvelope) -> usize {
        self.broadcast_tx.send(message).unwrap_or(0)
    }

    /// Subscribe an agent to a topic.
    pub async fn subscribe(&self, agent_id: impl Into<String>, topic: impl Into<String>) {
        let agent_id = agent_id.into();
        let topic = topic.into();

        let mut subs = self.subscriptions.write().await;
        let subscribers = subs.entry(topic).or_default();
        if !subscribers.contains(&agent_id) {
            subscribers.push(agent_id);
        }
    }

    /// Unsubscribe an agent from a topic.
    pub async fn unsubscribe(&self, agent_id: &str, topic: &str) {
        let mut subs = self.subscriptions.write().await;
        if let Some(subscribers) = subs.get_mut(topic) {
            subscribers.retain(|id| id != agent_id);
        }
    }

    /// Publish a message to all subscribers of a topic.
    ///
    /// Returns the number of agents the message was sent to.
    pub async fn publish(&self, message: MessageEnvelope) -> usize {
        let topic = match &message.topic {
            Some(t) => t.clone(),
            None => return 0,
        };

        let subs = self.subscriptions.read().await;
        let subscribers = match subs.get(&topic) {
            Some(s) => s.clone(),
            None => return 0,
        };
        drop(subs);

        let agents = self.agents.read().await;
        let mut count = 0;

        for subscriber_id in &subscribers {
            // Don't send to the sender
            if subscriber_id == &message.from {
                continue;
            }
            if let Some(tx) = agents.get(subscriber_id) {
                if tx.send(message.clone()).await.is_ok() {
                    count += 1;
                }
            }
        }

        count
    }

    /// Get the list of registered agent IDs.
    pub async fn registered_agents(&self) -> Vec<String> {
        self.agents.read().await.keys().cloned().collect()
    }

    /// Get subscribers for a topic.
    pub async fn topic_subscribers(&self, topic: &str) -> Vec<String> {
        self.subscriptions
            .read()
            .await
            .get(topic)
            .cloned()
            .unwrap_or_default()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_register_and_unregister() {
        let bus = MessageBus::new();

        let _mailbox = bus.register("agent1").await;
        assert!(bus
            .registered_agents()
            .await
            .contains(&"agent1".to_string()));

        bus.unregister("agent1").await;
        assert!(!bus
            .registered_agents()
            .await
            .contains(&"agent1".to_string()));
    }

    #[tokio::test]
    async fn test_send_direct_message() {
        let bus = MessageBus::new();
        let mut mailbox = bus.register("agent1").await;
        let _sender_mailbox = bus.register("agent2").await;

        let msg = MessageEnvelope::new("agent2", "agent1", json!({"hello": "world"}));
        let sent = bus.send(msg).await;
        assert!(sent);

        let received = mailbox.receiver.recv().await.unwrap();
        assert_eq!(received.from, "agent2");
        assert_eq!(received.to, Some("agent1".to_string()));
        assert_eq!(received.payload, json!({"hello": "world"}));
    }

    #[tokio::test]
    async fn test_send_to_unregistered_agent() {
        let bus = MessageBus::new();
        let _mailbox = bus.register("agent1").await;

        let msg = MessageEnvelope::new("agent1", "nonexistent", json!("test"));
        let sent = bus.send(msg).await;
        assert!(!sent);
    }

    #[tokio::test]
    async fn test_broadcast() {
        let bus = MessageBus::new();
        let mut mailbox1 = bus.register("agent1").await;
        let mut mailbox2 = bus.register("agent2").await;

        let msg = MessageEnvelope::broadcast("agent3", json!({"broadcast": true}));
        let count = bus.broadcast(msg);
        assert_eq!(count, 2);

        let received1 = mailbox1.broadcast_receiver.recv().await.unwrap();
        assert_eq!(received1.from, "agent3");
        assert_eq!(received1.payload, json!({"broadcast": true}));

        let received2 = mailbox2.broadcast_receiver.recv().await.unwrap();
        assert_eq!(received2.from, "agent3");
    }

    #[tokio::test]
    async fn test_subscribe_and_publish() {
        let bus = MessageBus::new();
        let mut mailbox1 = bus.register("agent1").await;
        let _mailbox2 = bus.register("agent2").await;

        bus.subscribe("agent1", "weather").await;
        bus.subscribe("agent2", "weather").await;

        let msg = MessageEnvelope::publish("agent1", "weather", json!({"temp": 72}));
        let count = bus.publish(msg).await;

        // Should send to agent2 only (not back to sender agent1)
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let bus = MessageBus::new();
        let _mailbox1 = bus.register("agent1").await;

        bus.subscribe("agent1", "news").await;
        assert_eq!(bus.topic_subscribers("news").await.len(), 1);

        bus.unsubscribe("agent1", "news").await;
        assert_eq!(bus.topic_subscribers("news").await.len(), 0);
    }

    #[tokio::test]
    async fn test_message_envelope_new() {
        let msg = MessageEnvelope::new("sender", "receiver", json!(42));
        assert_eq!(msg.from, "sender");
        assert_eq!(msg.to, Some("receiver".to_string()));
        assert_eq!(msg.payload, json!(42));
        assert!(msg.topic.is_none());
        assert!(!msg.id.is_empty());
        assert!(!msg.timestamp.is_empty());
    }

    #[tokio::test]
    async fn test_message_envelope_broadcast() {
        let msg = MessageEnvelope::broadcast("sender", json!("hello all"));
        assert_eq!(msg.from, "sender");
        assert!(msg.to.is_none());
        assert!(msg.topic.is_none());
    }

    #[tokio::test]
    async fn test_message_envelope_publish() {
        let msg = MessageEnvelope::publish("sender", "topic1", json!("data"));
        assert_eq!(msg.from, "sender");
        assert!(msg.to.is_none());
        assert_eq!(msg.topic, Some("topic1".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_subscriptions() {
        let bus = MessageBus::new();
        let _m1 = bus.register("a1").await;
        let _m2 = bus.register("a2").await;
        let _m3 = bus.register("a3").await;

        bus.subscribe("a1", "topic_a").await;
        bus.subscribe("a2", "topic_a").await;
        bus.subscribe("a3", "topic_a").await;
        bus.subscribe("a1", "topic_b").await;

        assert_eq!(bus.topic_subscribers("topic_a").await.len(), 3);
        assert_eq!(bus.topic_subscribers("topic_b").await.len(), 1);
    }

    #[tokio::test]
    async fn test_unregister_removes_subscriptions() {
        let bus = MessageBus::new();
        let _m1 = bus.register("agent1").await;

        bus.subscribe("agent1", "topic1").await;
        bus.subscribe("agent1", "topic2").await;

        bus.unregister("agent1").await;

        assert_eq!(bus.topic_subscribers("topic1").await.len(), 0);
        assert_eq!(bus.topic_subscribers("topic2").await.len(), 0);
    }

    #[tokio::test]
    async fn test_message_envelope_serialization() {
        let msg = MessageEnvelope::new("a", "b", json!({"key": "value"}));
        let json_str = serde_json::to_string(&msg).unwrap();
        let deserialized: MessageEnvelope = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.from, "a");
        assert_eq!(deserialized.to, Some("b".to_string()));
        assert_eq!(deserialized.payload, json!({"key": "value"}));
    }
}
