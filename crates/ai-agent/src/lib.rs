// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! AI Agent system with ReAct loop and planning.
//!
//! This crate provides the [`Agent`] type which orchestrates LLM interactions
//! with tools and memory to accomplish tasks through iterative reasoning.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_agent::{Agent, AgentBuilder};
//! use ai_memory::InMemoryMemory;
//! use ai_provider_openai::OpenAiProvider;
//! use ai_core::{Tool, ToolDescriptor, ToolOutput};
//! use serde_json::json;
//! # use async_trait::async_trait;
//! # struct WeatherTool;
//! # #[async_trait::async_trait]
//! # impl Tool for WeatherTool {
//! #     fn descriptor(&self) -> ToolDescriptor {
//! #         ToolDescriptor::new("weather", "Get weather", json!({}))
//! #     }
//! #     async fn execute(&self, _input: serde_json::Value) -> Result<ToolOutput, ai_core::error::ToolError> {
//! #         Ok(ToolOutput::success("72°F and sunny"))
//! #     }
//! # }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let memory = InMemoryMemory::new(100);
//! let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
//!
//! let agent = AgentBuilder::new(memory)
//!     .provider(provider)
//!     .role("You are a helpful assistant with access to weather data.")
//!     .tool(WeatherTool)
//!     .build()?;
//!
//! let response = agent.run("What's the weather in San Francisco?").await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod builder;
pub mod delegate;
pub mod planner;

pub use agent::Agent;
pub use builder::AgentBuilder;
pub use delegate::DelegateTool;
pub use planner::{Plan, PlanResult, PlanStep, PlanStepStatus};
