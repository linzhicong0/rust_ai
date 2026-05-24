// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Memory implementations for conversation storage.
//!
//! This crate provides implementations of the [`Memory`] trait for storing
//! and retrieving conversation history.
//!
//! ## Implementations
//!
//! - [`InMemoryMemory`] — In-memory storage for single-threaded use
//! - [`ThreadScopedMemory`] — Thread-scoped storage for multiple conversations
//! - [`SharedMemory`] — Thread-safe shared memory with Arc
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_memory::ThreadScopedMemory;
//! use ai_core::memory::{MemoryEntry, ScopedMemory};
//! use ai_core::types::Role;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Thread-scoped memory maintains separate conversation histories
//! let memory = ThreadScopedMemory::new(100);
//!
//! let session_id = "user-123-session-1";
//!
//! // Add messages to this specific session
//! memory.add_to_scope(
//!     session_id,
//!     MemoryEntry::new(Role::User, "Hello!")
//! ).await?;
//!
//! // Retrieve only this session's history
//! let history = memory.get_from_scope(session_id, None).await?;
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod in_memory;
pub mod long_term;
pub mod scoped;
pub mod shared;

pub use agent::AgentMemory;
pub use in_memory::InMemoryMemory;
pub use long_term::{
    InMemoryLongTermStore, LongTermMemory, LongTermMemoryConfig, LongTermMemoryEntry,
};
pub use scoped::ThreadScopedMemory;
pub use shared::SharedMemory;
