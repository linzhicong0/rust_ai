// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Communication Platforms (REQ-17.2)
//!
//! Provides the `Channel` trait for integrating with communication platforms:
//! Slack, Discord, Email, SMS, WhatsApp.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors that can occur in communication operations.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("Send error: {0}")]
    Send(String),
    #[error("Receive error: {0}")]
    Receive(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Authentication error: {0}")]
    Authentication(String),
    #[error("Rate limited: retry after {0} seconds")]
    RateLimited(u64),
    #[error("Invalid recipient: {0}")]
    InvalidRecipient(String),
}

/// The type of communication platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformType {
    Slack,
    Discord,
    Email,
    Sms,
    WhatsApp,
    Custom,
}

/// A normalized message that works across all platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMessage {
    /// Unique message ID.
    pub id: String,
    /// The sender identifier.
    pub sender: String,
    /// The recipient (user, channel, group).
    pub recipient: String,
    /// Message body text.
    pub body: String,
    /// Platform this message originated from.
    pub platform: PlatformType,
    /// Optional thread/reply-to ID.
    pub thread_id: Option<String>,
    /// Attachments (file URLs, images, etc.).
    pub attachments: Vec<Attachment>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
}

impl ChannelMessage {
    /// Create a new message.
    pub fn new(
        sender: impl Into<String>,
        recipient: impl Into<String>,
        body: impl Into<String>,
        platform: PlatformType,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            sender: sender.into(),
            recipient: recipient.into(),
            body: body.into(),
            platform,
            thread_id: None,
            attachments: Vec::new(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Set thread ID for threaded replies.
    pub fn with_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Add an attachment.
    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        self.attachments.push(attachment);
        self
    }
}

/// An attachment to a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    /// File name.
    pub name: String,
    /// MIME type.
    pub content_type: String,
    /// URL or file path.
    pub url: String,
    /// File size in bytes (if known).
    pub size: Option<u64>,
}

/// Configuration for sending a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendOptions {
    /// Whether to send as a reply in a thread.
    pub thread_reply: bool,
    /// Whether to notify/mention the recipient.
    pub notify: bool,
    /// Custom platform-specific options.
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for SendOptions {
    fn default() -> Self {
        Self {
            thread_reply: false,
            notify: true,
            extra: HashMap::new(),
        }
    }
}

/// Trait for communication platform connectors.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Get the platform type for this channel.
    fn platform(&self) -> PlatformType;

    /// Send a message to a recipient.
    async fn send(&self, message: ChannelMessage, options: SendOptions) -> Result<String, ChannelError>;

    /// Receive messages (poll-based). Returns new messages since the given cursor.
    async fn receive(&self, cursor: Option<&str>) -> Result<Vec<ChannelMessage>, ChannelError>;

    /// Check if the channel connection is healthy.
    async fn health_check(&self) -> Result<bool, ChannelError>;
}

/// In-memory channel implementation for testing.
pub struct InMemoryChannel {
    platform: PlatformType,
    messages: std::sync::Arc<tokio::sync::Mutex<Vec<ChannelMessage>>>,
}

impl InMemoryChannel {
    /// Create a new in-memory channel for the given platform.
    pub fn new(platform: PlatformType) -> Self {
        Self {
            platform,
            messages: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Channel for InMemoryChannel {
    fn platform(&self) -> PlatformType {
        self.platform
    }

    async fn send(&self, message: ChannelMessage, _options: SendOptions) -> Result<String, ChannelError> {
        if message.recipient.is_empty() {
            return Err(ChannelError::InvalidRecipient(
                "Recipient cannot be empty".to_string(),
            ));
        }
        let id = message.id.clone();
        let mut messages = self.messages.lock().await;
        messages.push(message);
        Ok(id)
    }

    async fn receive(&self, cursor: Option<&str>) -> Result<Vec<ChannelMessage>, ChannelError> {
        let messages = self.messages.lock().await;
        if let Some(cursor) = cursor {
            // Return messages after the cursor (by ID)
            let pos = messages.iter().position(|m| m.id == cursor);
            if let Some(pos) = pos {
                Ok(messages[pos + 1..].to_vec())
            } else {
                Ok(messages.clone())
            }
        } else {
            Ok(messages.clone())
        }
    }

    async fn health_check(&self) -> Result<bool, ChannelError> {
        Ok(true)
    }
}

/// Normalize a message from any platform into a unified format.
/// This handles platform-specific quirks.
pub fn normalize_message(mut message: ChannelMessage) -> ChannelMessage {
    // Trim whitespace from body
    message.body = message.body.trim().to_string();

    // Normalize sender/recipient: trim and lowercase for consistency
    message.sender = message.sender.trim().to_lowercase();
    message.recipient = message.recipient.trim().to_lowercase();

    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_message_creation() {
        let msg = ChannelMessage::new("bot", "#general", "Hello!", PlatformType::Slack);
        assert_eq!(msg.sender, "bot");
        assert_eq!(msg.recipient, "#general");
        assert_eq!(msg.body, "Hello!");
        assert_eq!(msg.platform, PlatformType::Slack);
        assert!(msg.thread_id.is_none());
        assert!(msg.attachments.is_empty());
    }

    #[tokio::test]
    async fn test_channel_message_with_thread() {
        let msg = ChannelMessage::new("bot", "#general", "Reply", PlatformType::Slack)
            .with_thread("thread-123");
        assert_eq!(msg.thread_id.unwrap(), "thread-123");
    }

    #[tokio::test]
    async fn test_channel_message_with_attachment() {
        let attachment = Attachment {
            name: "report.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            url: "https://example.com/report.pdf".to_string(),
            size: Some(1024),
        };
        let msg = ChannelMessage::new("bot", "user@example.com", "See attached", PlatformType::Email)
            .with_attachment(attachment);
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(msg.attachments[0].name, "report.pdf");
    }

    #[tokio::test]
    async fn test_in_memory_channel_send_and_receive() {
        let channel = InMemoryChannel::new(PlatformType::Slack);

        let msg = ChannelMessage::new("bot", "#general", "Hello!", PlatformType::Slack);
        let id = channel.send(msg, SendOptions::default()).await.unwrap();
        assert!(!id.is_empty());

        let messages = channel.receive(None).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].body, "Hello!");
    }

    #[tokio::test]
    async fn test_in_memory_channel_send_empty_recipient_fails() {
        let channel = InMemoryChannel::new(PlatformType::Discord);

        let msg = ChannelMessage::new("bot", "", "Hello!", PlatformType::Discord);
        let result = channel.send(msg, SendOptions::default()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChannelError::InvalidRecipient(_)));
    }

    #[tokio::test]
    async fn test_in_memory_channel_receive_with_cursor() {
        let channel = InMemoryChannel::new(PlatformType::Email);

        let msg1 = ChannelMessage::new("bot", "user@test.com", "First", PlatformType::Email);
        let msg1_id = msg1.id.clone();
        channel.send(msg1, SendOptions::default()).await.unwrap();

        let msg2 = ChannelMessage::new("bot", "user@test.com", "Second", PlatformType::Email);
        channel.send(msg2, SendOptions::default()).await.unwrap();

        // Receive after the first message
        let messages = channel.receive(Some(&msg1_id)).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].body, "Second");
    }

    #[tokio::test]
    async fn test_in_memory_channel_platform() {
        let channel = InMemoryChannel::new(PlatformType::Discord);
        assert_eq!(channel.platform(), PlatformType::Discord);
    }

    #[tokio::test]
    async fn test_in_memory_channel_health_check() {
        let channel = InMemoryChannel::new(PlatformType::Sms);
        assert!(channel.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_normalize_message() {
        let msg = ChannelMessage::new("  BOT  ", "#General", "  Hello World  ", PlatformType::Slack);
        let normalized = normalize_message(msg);
        assert_eq!(normalized.body, "Hello World");
        assert_eq!(normalized.sender, "bot");
        assert_eq!(normalized.recipient, "#general");
    }

    #[tokio::test]
    async fn test_send_options_default() {
        let opts = SendOptions::default();
        assert!(!opts.thread_reply);
        assert!(opts.notify);
        assert!(opts.extra.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_channels() {
        let slack = InMemoryChannel::new(PlatformType::Slack);
        let discord = InMemoryChannel::new(PlatformType::Discord);

        let msg1 = ChannelMessage::new("bot", "#channel", "Slack msg", PlatformType::Slack);
        let msg2 = ChannelMessage::new("bot", "#channel", "Discord msg", PlatformType::Discord);

        slack.send(msg1, SendOptions::default()).await.unwrap();
        discord.send(msg2, SendOptions::default()).await.unwrap();

        let slack_msgs = slack.receive(None).await.unwrap();
        let discord_msgs = discord.receive(None).await.unwrap();

        assert_eq!(slack_msgs.len(), 1);
        assert_eq!(discord_msgs.len(), 1);
        assert_eq!(slack_msgs[0].body, "Slack msg");
        assert_eq!(discord_msgs[0].body, "Discord msg");
    }
}
