//! Debugging tools for AI agent execution.
//!
//! Provides step-through agent execution inspection, prompt preview before
//! sending to an LLM, and context window (token budget) visualization.
//!
//! ## Example
//!
//! ```rust
//! use ai_observability::debug::{AgentDebugger, DebugHook, DebugEvent, ContextWindowView};
//!
//! // Build a debugger with a custom hook that records events.
//! let mut events: Vec<DebugEvent> = Vec::new();
//! let debugger = AgentDebugger::new();
//!
//! // Preview a prompt before it would be sent to the LLM.
//! let preview = debugger.preview_prompt("You are helpful.", "What is 2+2?");
//! assert!(preview.contains("You are helpful."));
//!
//! // Inspect token budget usage.
//! let view = ContextWindowView::new(4096)
//!     .with_system_tokens(50)
//!     .with_history_tokens(200)
//!     .with_input_tokens(30);
//!
//! assert!(view.remaining_tokens() > 0);
//! assert!(view.usage_fraction() < 1.0);
//! ```

use std::fmt;
use std::sync::{Arc, Mutex};

// ── Debug events ─────────────────────────────────────────────────────────────

/// A discrete event emitted during agent execution.
#[derive(Debug, Clone, PartialEq)]
pub enum DebugEvent {
    /// The agent is about to send a prompt to the LLM.
    ///
    /// Contains the assembled prompt text.
    PromptAssembled {
        /// The full prompt text (system + history + current input).
        prompt: String,
    },

    /// A tool call was made by the agent.
    ToolCall {
        /// The name of the tool.
        tool_name: String,
        /// The raw JSON input passed to the tool.
        input: String,
        /// The raw JSON output returned by the tool.
        output: String,
    },

    /// The LLM returned a response.
    LlmResponse {
        /// The model's text response.
        content: String,
        /// Token usage, if available.
        tokens_used: Option<u32>,
    },

    /// The agent reasoning step (chain-of-thought text).
    Reasoning {
        /// The reasoning / thought text produced by the agent.
        thought: String,
        /// The step index within the current agent run.
        step: usize,
    },

    /// A breakpoint was hit.
    BreakpointHit {
        /// Name of the breakpoint.
        name: String,
    },

    /// The agent run finished.
    Finished {
        /// Final agent output.
        output: String,
        /// Total steps taken.
        total_steps: usize,
    },
}

impl fmt::Display for DebugEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DebugEvent::PromptAssembled { prompt } => {
                write!(f, "[PROMPT] {:.200}…", prompt)
            }
            DebugEvent::ToolCall {
                tool_name,
                input,
                output,
            } => write!(f, "[TOOL] {}({}) → {}", tool_name, input, output),
            DebugEvent::LlmResponse {
                content,
                tokens_used,
            } => {
                if let Some(t) = tokens_used {
                    write!(f, "[LLM] {} ({} tokens)", content, t)
                } else {
                    write!(f, "[LLM] {}", content)
                }
            }
            DebugEvent::Reasoning { thought, step } => {
                write!(f, "[THOUGHT step={}] {}", step, thought)
            }
            DebugEvent::BreakpointHit { name } => write!(f, "[BREAKPOINT] {}", name),
            DebugEvent::Finished {
                output,
                total_steps,
            } => write!(f, "[DONE steps={}] {}", total_steps, output),
        }
    }
}

// ── Debug hook ───────────────────────────────────────────────────────────────

/// A callback invoked for each [`DebugEvent`] emitted by the debugger.
pub trait DebugHook: Send + Sync {
    /// Called when a debug event is emitted.
    fn on_event(&self, event: &DebugEvent);
}

/// A [`DebugHook`] that collects events into an in-memory buffer.
///
/// Useful in tests to assert on the exact sequence of events.
#[derive(Debug, Default, Clone)]
pub struct RecordingHook {
    events: Arc<Mutex<Vec<DebugEvent>>>,
}

impl RecordingHook {
    /// Create an empty recording hook.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a snapshot of all recorded events.
    pub fn events(&self) -> Vec<DebugEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Return the number of recorded events.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Return `true` if no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DebugHook for RecordingHook {
    fn on_event(&self, event: &DebugEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

// ── Context window view ───────────────────────────────────────────────────────

/// Visualization of context window (token budget) usage.
///
/// Tracks how the token budget is allocated across system prompt, conversation
/// history, and the current user input.
#[derive(Debug, Clone)]
pub struct ContextWindowView {
    /// Total token capacity of the model's context window.
    total_tokens: u32,
    /// Tokens consumed by the system prompt.
    system_tokens: u32,
    /// Tokens consumed by conversation history.
    history_tokens: u32,
    /// Tokens consumed by the current user input.
    input_tokens: u32,
    /// Tokens reserved for the model's output.
    reserved_output_tokens: u32,
}

impl ContextWindowView {
    /// Create a context window view with the given total capacity.
    pub fn new(total_tokens: u32) -> Self {
        Self {
            total_tokens,
            system_tokens: 0,
            history_tokens: 0,
            input_tokens: 0,
            reserved_output_tokens: 0,
        }
    }

    /// Set the system prompt token count.
    pub fn with_system_tokens(mut self, tokens: u32) -> Self {
        self.system_tokens = tokens;
        self
    }

    /// Set the history token count.
    pub fn with_history_tokens(mut self, tokens: u32) -> Self {
        self.history_tokens = tokens;
        self
    }

    /// Set the current input token count.
    pub fn with_input_tokens(mut self, tokens: u32) -> Self {
        self.input_tokens = tokens;
        self
    }

    /// Reserve tokens for the model's output.
    pub fn with_reserved_output(mut self, tokens: u32) -> Self {
        self.reserved_output_tokens = tokens;
        self
    }

    /// Total token capacity.
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Total tokens currently used (system + history + input).
    pub fn used_tokens(&self) -> u32 {
        self.system_tokens
            .saturating_add(self.history_tokens)
            .saturating_add(self.input_tokens)
    }

    /// Remaining tokens after deducting used and reserved output tokens.
    pub fn remaining_tokens(&self) -> u32 {
        self.total_tokens
            .saturating_sub(self.used_tokens())
            .saturating_sub(self.reserved_output_tokens)
    }

    /// Fraction of the context window used (0.0 – 1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.total_tokens == 0 {
            return 1.0;
        }
        self.used_tokens() as f64 / self.total_tokens as f64
    }

    /// Return `true` if the current usage would exceed the context window.
    pub fn is_overflow(&self) -> bool {
        self.used_tokens() > self.total_tokens
    }

    /// Render a human-readable budget report.
    pub fn render(&self) -> String {
        format!(
            "Context window: {}/{} tokens used ({:.1}%)\n  system: {}\n  history: {}\n  input: {}\n  reserved output: {}\n  remaining: {}",
            self.used_tokens(),
            self.total_tokens,
            self.usage_fraction() * 100.0,
            self.system_tokens,
            self.history_tokens,
            self.input_tokens,
            self.reserved_output_tokens,
            self.remaining_tokens(),
        )
    }
}

// ── Breakpoint registry ───────────────────────────────────────────────────────

/// A named breakpoint that pauses execution if enabled.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    name: String,
    enabled: bool,
}

impl Breakpoint {
    /// Create a new enabled breakpoint.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
        }
    }

    /// Enable or disable this breakpoint.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Return `true` if this breakpoint is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Return the breakpoint name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ── Agent debugger ────────────────────────────────────────────────────────────

/// Central debugger for AI agent execution.
///
/// The `AgentDebugger` provides:
/// - **Prompt preview** — inspect the assembled prompt before it reaches the LLM.
/// - **Event emission** — broadcast [`DebugEvent`]s to registered [`DebugHook`]s.
/// - **Breakpoint management** — pause execution at named checkpoints.
/// - **Context window visualization** — inspect token budget allocation.
#[derive(Default)]
pub struct AgentDebugger {
    hooks: Vec<Box<dyn DebugHook>>,
    breakpoints: Vec<Breakpoint>,
}

impl AgentDebugger {
    /// Create a new debugger with no hooks or breakpoints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a hook that will receive all future events.
    pub fn add_hook(&mut self, hook: impl DebugHook + 'static) -> &mut Self {
        self.hooks.push(Box::new(hook));
        self
    }

    /// Register a breakpoint.
    pub fn add_breakpoint(&mut self, bp: Breakpoint) -> &mut Self {
        self.breakpoints.push(bp);
        self
    }

    /// Emit an event to all registered hooks.
    pub fn emit(&self, event: DebugEvent) {
        for hook in &self.hooks {
            hook.on_event(&event);
        }
    }

    /// Return `true` if the named breakpoint is registered and enabled.
    pub fn is_breakpoint_active(&self, name: &str) -> bool {
        self.breakpoints
            .iter()
            .any(|bp| bp.name() == name && bp.is_enabled())
    }

    /// Emit a [`DebugEvent::BreakpointHit`] if the named breakpoint is active.
    ///
    /// Returns `true` if the breakpoint was hit (and the event was emitted).
    pub fn check_breakpoint(&self, name: &str) -> bool {
        if self.is_breakpoint_active(name) {
            self.emit(DebugEvent::BreakpointHit {
                name: name.to_string(),
            });
            return true;
        }
        false
    }

    /// Assemble and return a prompt preview string.
    ///
    /// This does **not** send anything to an LLM; it simply formats the prompt
    /// so it can be inspected before dispatch.
    pub fn preview_prompt(&self, system: &str, user_input: &str) -> String {
        let preview = format!("[SYSTEM]\n{}\n\n[USER]\n{}", system, user_input);
        self.emit(DebugEvent::PromptAssembled {
            prompt: preview.clone(),
        });
        preview
    }

    /// Emit a reasoning step event.
    pub fn emit_reasoning(&self, thought: impl Into<String>, step: usize) {
        self.emit(DebugEvent::Reasoning {
            thought: thought.into(),
            step,
        });
    }

    /// Emit a tool call event.
    pub fn emit_tool_call(
        &self,
        tool_name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
    ) {
        self.emit(DebugEvent::ToolCall {
            tool_name: tool_name.into(),
            input: input.into(),
            output: output.into(),
        });
    }

    /// Emit an LLM response event.
    pub fn emit_llm_response(&self, content: impl Into<String>, tokens_used: Option<u32>) {
        self.emit(DebugEvent::LlmResponse {
            content: content.into(),
            tokens_used,
        });
    }

    /// Emit a finished event.
    pub fn emit_finished(&self, output: impl Into<String>, total_steps: usize) {
        self.emit(DebugEvent::Finished {
            output: output.into(),
            total_steps,
        });
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-11.4: prompt preview is returned and event is emitted
    #[test]
    fn test_prompt_preview_emits_event_and_returns_preview() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        let preview = debugger.preview_prompt("You are helpful.", "What is 2+2?");

        assert!(preview.contains("You are helpful."));
        assert!(preview.contains("What is 2+2?"));

        let events = hook.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DebugEvent::PromptAssembled { prompt } => {
                assert!(prompt.contains("You are helpful."));
                assert!(prompt.contains("What is 2+2?"));
            }
            _ => panic!("Expected PromptAssembled event"),
        }
    }

    // REQ-11.4: reasoning step events are emitted in order
    #[test]
    fn test_reasoning_events_are_emitted_in_order() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        debugger.emit_reasoning("I need to search for the answer", 0);
        debugger.emit_reasoning("I found the answer: 42", 1);

        let events = hook.events();
        assert_eq!(events.len(), 2);

        match &events[0] {
            DebugEvent::Reasoning { thought, step } => {
                assert_eq!(*step, 0);
                assert!(thought.contains("search"));
            }
            _ => panic!("Expected Reasoning event at index 0"),
        }

        match &events[1] {
            DebugEvent::Reasoning { thought, step } => {
                assert_eq!(*step, 1);
                assert!(thought.contains("42"));
            }
            _ => panic!("Expected Reasoning event at index 1"),
        }
    }

    // REQ-11.4: tool call events are recorded
    #[test]
    fn test_tool_call_events_are_recorded() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        debugger.emit_tool_call("web_search", r#"{"query":"rust"}"#, r#"{"results":[]}"#);

        let events = hook.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DebugEvent::ToolCall {
                tool_name,
                input,
                output,
            } => {
                assert_eq!(tool_name, "web_search");
                assert!(input.contains("rust"));
                assert!(output.contains("results"));
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    // REQ-11.4: breakpoint fires event when active
    #[test]
    fn test_breakpoint_hit_event_emitted_when_active() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());
        debugger.add_breakpoint(Breakpoint::new("before_tool_call"));

        let hit = debugger.check_breakpoint("before_tool_call");
        assert!(hit);

        let events = hook.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DebugEvent::BreakpointHit { name } => assert_eq!(name, "before_tool_call"),
            _ => panic!("Expected BreakpointHit event"),
        }
    }

    // REQ-11.4: disabled breakpoint does not fire
    #[test]
    fn test_disabled_breakpoint_does_not_fire() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());
        let mut bp = Breakpoint::new("disabled_bp");
        bp.set_enabled(false);
        debugger.add_breakpoint(bp);

        let hit = debugger.check_breakpoint("disabled_bp");
        assert!(!hit);
        assert!(hook.is_empty());
    }

    // REQ-11.4: unknown breakpoint does not fire
    #[test]
    fn test_unknown_breakpoint_does_not_fire() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        let hit = debugger.check_breakpoint("nonexistent");
        assert!(!hit);
        assert!(hook.is_empty());
    }

    // REQ-11.4: context window view tracks token budget
    #[test]
    fn test_context_window_view_token_budget() {
        let view = ContextWindowView::new(4096)
            .with_system_tokens(100)
            .with_history_tokens(500)
            .with_input_tokens(80);

        assert_eq!(view.used_tokens(), 680);
        assert_eq!(view.remaining_tokens(), 3416);
        assert!(!view.is_overflow());

        let fraction = view.usage_fraction();
        assert!(fraction > 0.0 && fraction < 1.0);
    }

    // REQ-11.4: context window overflow is detected
    #[test]
    fn test_context_window_overflow_is_detected() {
        let view = ContextWindowView::new(100)
            .with_system_tokens(60)
            .with_history_tokens(60);

        assert!(view.is_overflow());
    }

    // REQ-11.4: context window render contains key information
    #[test]
    fn test_context_window_render_contains_usage() {
        let view = ContextWindowView::new(8192)
            .with_system_tokens(200)
            .with_history_tokens(1000)
            .with_input_tokens(50)
            .with_reserved_output(512);

        let rendered = view.render();
        assert!(rendered.contains("1250")); // used tokens
        assert!(rendered.contains("8192")); // total
        assert!(rendered.contains("512")); // reserved
    }

    // REQ-11.4: multiple hooks all receive events
    #[test]
    fn test_multiple_hooks_all_receive_events() {
        let hook1 = RecordingHook::new();
        let hook2 = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook1.clone());
        debugger.add_hook(hook2.clone());

        debugger.emit_reasoning("thinking", 0);

        assert_eq!(hook1.len(), 1);
        assert_eq!(hook2.len(), 1);
    }

    // REQ-11.4: LLM response event with token count
    #[test]
    fn test_llm_response_event_with_token_count() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        debugger.emit_llm_response("Paris is the capital of France.", Some(8));

        let events = hook.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DebugEvent::LlmResponse {
                content,
                tokens_used,
            } => {
                assert!(content.contains("Paris"));
                assert_eq!(*tokens_used, Some(8));
            }
            _ => panic!("Expected LlmResponse event"),
        }
    }

    // REQ-11.4: finished event records total steps
    #[test]
    fn test_finished_event_records_total_steps() {
        let hook = RecordingHook::new();
        let mut debugger = AgentDebugger::new();
        debugger.add_hook(hook.clone());

        debugger.emit_finished("The answer is 42.", 3);

        let events = hook.events();
        match &events[0] {
            DebugEvent::Finished {
                output,
                total_steps,
            } => {
                assert!(output.contains("42"));
                assert_eq!(*total_steps, 3);
            }
            _ => panic!("Expected Finished event"),
        }
    }
}
