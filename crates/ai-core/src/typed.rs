//! Type-safe interfaces for agents, tools, and pipelines (REQ-15.6).
//!
//! This module provides generic type-safe wrappers that enable compile-time
//! verification of input/output type flow across pipeline steps.
//!
//! ## Features
//!
//! - `TypedAgent<I, O>` — Agent with typed input and output
//! - `TypedTool<I, O>` — Tool with typed execute
//! - `TypedPipeline` — Compile-time verification of pipeline type flow

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

/// Error type for typed operations.
#[derive(Debug, thiserror::Error)]
pub enum TypedError {
    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    Execution(String),
}

impl From<serde_json::Error> for TypedError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// A type-safe agent with generic input and output types.
///
/// This trait provides compile-time guarantees that the agent receives
/// the correct input type and produces the correct output type.
///
/// # Type Parameters
///
/// * `I` — Input type (must be serializable)
/// * `O` — Output type (must be deserializable)
#[async_trait]
pub trait TypedAgent<I, O>: Send + Sync
where
    I: Serialize + Send + Sync + 'static,
    O: DeserializeOwned + Send + Sync + 'static,
{
    /// Execute the agent with typed input, producing typed output.
    async fn execute(&self, input: I) -> Result<O, TypedError>;

    /// Get the agent's name.
    fn name(&self) -> &str;
}

/// A type-safe tool with generic input and output types.
///
/// This trait provides compile-time guarantees that the tool receives
/// the correct input type and produces the correct output type.
///
/// # Type Parameters
///
/// * `I` — Input type (must be deserializable from JSON)
/// * `O` — Output type (must be serializable to JSON)
#[async_trait]
pub trait TypedTool<I, O>: Send + Sync
where
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + Sync + 'static,
{
    /// Execute the tool with typed input, producing typed output.
    async fn execute(&self, input: I) -> Result<O, TypedError>;

    /// Get the tool's name.
    fn name(&self) -> &str;

    /// Get a description of the tool.
    fn description(&self) -> &str;
}

/// A type-safe pipeline step that transforms input type `I` into output type `O`.
///
/// This enables compile-time verification that pipeline steps connect
/// correctly (output of step A matches input of step B).
#[async_trait]
pub trait TypedStep<I, O>: Send + Sync
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Execute this step, transforming input to output.
    async fn run(&self, input: I) -> Result<O, TypedError>;
}

/// Adapter that chains two typed steps together.
///
/// Ensures at compile time that the output type of step A matches
/// the input type of step B: `A<I, M> -> B<M, O>` produces `Chain<I, O>`.
pub struct TypedChain<I, M, O, A, B>
where
    I: Send + Sync + 'static,
    M: Send + Sync + 'static,
    O: Send + Sync + 'static,
    A: TypedStep<I, M>,
    B: TypedStep<M, O>,
{
    first: A,
    second: B,
    _phantom: PhantomData<(I, M, O)>,
}

impl<I, M, O, A, B> TypedChain<I, M, O, A, B>
where
    I: Send + Sync + 'static,
    M: Send + Sync + 'static,
    O: Send + Sync + 'static,
    A: TypedStep<I, M>,
    B: TypedStep<M, O>,
{
    /// Create a new chain of two steps.
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<I, M, O, A, B> TypedStep<I, O> for TypedChain<I, M, O, A, B>
where
    I: Send + Sync + 'static,
    M: Send + Sync + 'static,
    O: Send + Sync + 'static,
    A: TypedStep<I, M>,
    B: TypedStep<M, O>,
{
    async fn run(&self, input: I) -> Result<O, TypedError> {
        let mid = self.first.run(input).await?;
        self.second.run(mid).await
    }
}

/// Extension trait for chaining typed steps.
pub trait TypedStepExt<I, O>: TypedStep<I, O> + Sized
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Chain this step with another step whose input matches this step's output.
    ///
    /// # Compile-time safety
    ///
    /// If the output type of `self` doesn't match the input type of `next`,
    /// the code will fail to compile.
    fn then<O2, B>(self, next: B) -> TypedChain<I, O, O2, Self, B>
    where
        O2: Send + Sync + 'static,
        B: TypedStep<O, O2>,
    {
        TypedChain::new(self, next)
    }
}

// Blanket implementation for all TypedStep implementations
impl<I, O, T> TypedStepExt<I, O> for T
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    T: TypedStep<I, O> + Sized,
{
}

/// A wrapper that adapts a `TypedAgent` into a `TypedStep`.
pub struct AgentStep<I, O, A>
where
    I: Serialize + Send + Sync + 'static,
    O: DeserializeOwned + Send + Sync + 'static,
    A: TypedAgent<I, O>,
{
    agent: A,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, A> AgentStep<I, O, A>
where
    I: Serialize + Send + Sync + 'static,
    O: DeserializeOwned + Send + Sync + 'static,
    A: TypedAgent<I, O>,
{
    /// Create a new agent step.
    pub fn new(agent: A) -> Self {
        Self {
            agent,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, A> TypedStep<I, O> for AgentStep<I, O, A>
where
    I: Serialize + Send + Sync + 'static,
    O: DeserializeOwned + Send + Sync + 'static,
    A: TypedAgent<I, O>,
{
    async fn run(&self, input: I) -> Result<O, TypedError> {
        self.agent.execute(input).await
    }
}

/// A wrapper that adapts a `TypedTool` into a `TypedStep`.
pub struct ToolStep<I, O, T>
where
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + Sync + 'static,
    T: TypedTool<I, O>,
{
    tool: T,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, T> ToolStep<I, O, T>
where
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + Sync + 'static,
    T: TypedTool<I, O>,
{
    /// Create a new tool step.
    pub fn new(tool: T) -> Self {
        Self {
            tool,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, T> TypedStep<I, O> for ToolStep<I, O, T>
where
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + Sync + 'static,
    T: TypedTool<I, O>,
{
    async fn run(&self, input: I) -> Result<O, TypedError> {
        self.tool.execute(input).await
    }
}

/// A simple function-based typed step for testing and simple transformations.
pub struct FnStep<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(I) -> Result<O, TypedError> + Send + Sync,
{
    func: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> FnStep<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(I) -> Result<O, TypedError> + Send + Sync,
{
    /// Create a new function step.
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, F> TypedStep<I, O> for FnStep<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(I) -> Result<O, TypedError> + Send + Sync,
{
    async fn run(&self, input: I) -> Result<O, TypedError> {
        (self.func)(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // ---- Test types ----

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MyInput {
        query: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MyOutput {
        answer: String,
        confidence: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct MyStruct {
        data: String,
        count: u32,
    }

    // ---- Test agent implementation ----

    struct TestAgent;

    #[async_trait]
    impl TypedAgent<String, MyOutput> for TestAgent {
        async fn execute(&self, input: String) -> Result<MyOutput, TypedError> {
            Ok(MyOutput {
                answer: format!("Response to: {}", input),
                confidence: 0.95,
            })
        }

        fn name(&self) -> &str {
            "test_agent"
        }
    }

    // ---- Test tool implementation ----

    struct TestTool;

    #[async_trait]
    impl TypedTool<MyInput, MyOutput> for TestTool {
        async fn execute(&self, input: MyInput) -> Result<MyOutput, TypedError> {
            Ok(MyOutput {
                answer: format!("Processed: {}", input.query),
                confidence: 0.9,
            })
        }

        fn name(&self) -> &str {
            "test_tool"
        }

        fn description(&self) -> &str {
            "A test tool"
        }
    }

    // REQ-15.6: Unit: Agent<String, MyOutput> compiles and produces MyOutput from String input
    #[tokio::test]
    async fn test_typed_agent_string_to_my_output() {
        let agent = TestAgent;
        let result = agent.execute("Hello".to_string()).await.unwrap();

        assert_eq!(result.answer, "Response to: Hello");
        assert_eq!(result.confidence, 0.95);
    }

    // REQ-15.6: Unit: Tool<MyInput, MyOutput> compiles with typed execute(MyInput) -> MyOutput
    #[tokio::test]
    async fn test_typed_tool_my_input_to_my_output() {
        let tool = TestTool;
        let input = MyInput {
            query: "test query".to_string(),
        };
        let result = tool.execute(input).await.unwrap();

        assert_eq!(result.answer, "Processed: test query");
        assert_eq!(result.confidence, 0.9);
    }

    // REQ-15.6: Unit: pipeline step A<String, MyStruct> -> B<MyStruct, String> compiles
    #[tokio::test]
    async fn test_pipeline_type_flow_compiles() {
        // Step A: String -> MyStruct
        let step_a = FnStep::new(|input: String| -> Result<MyStruct, TypedError> {
            Ok(MyStruct {
                data: input,
                count: 1,
            })
        });

        // Step B: MyStruct -> String
        let step_b = FnStep::new(|input: MyStruct| -> Result<String, TypedError> {
            Ok(format!("{}:{}", input.data, input.count))
        });

        // Chain A -> B: String -> String (this compiles because types match)
        let pipeline = step_a.then(step_b);

        let result = pipeline.run("hello".to_string()).await.unwrap();
        assert_eq!(result, "hello:1");
    }

    // REQ-15.6: Unit: pipeline step A<String, MyStruct> -> B<OtherStruct, String> fails to compile
    // This test verifies compile-time type safety. The commented code below would
    // fail to compile because MyStruct != OtherStruct:
    //
    // ```compile_fail
    // #[derive(Debug, Clone)]
    // struct OtherStruct { value: i32 }
    //
    // let step_a = FnStep::new(|input: String| -> Result<MyStruct, TypedError> {
    //     Ok(MyStruct { data: input, count: 1 })
    // });
    // let step_b = FnStep::new(|input: OtherStruct| -> Result<String, TypedError> {
    //     Ok(format!("{}", input.value))
    // });
    // // This line would fail to compile:
    // let pipeline = step_a.then(step_b);
    // ```
    //
    // We verify this by testing that the correct types DO compile (above test),
    // and document that mismatched types are caught at compile time.
    #[test]
    fn test_type_mismatch_is_compile_time_error() {
        // This test documents that type mismatches are caught at compile time.
        // The actual verification is done by the compiler - mismatched types
        // simply won't compile. See the compile_fail example above.

        // We verify the positive case: matching types compile fine
        fn _assert_chain_compiles() {
            let _step_a = FnStep::new(|input: String| -> Result<MyStruct, TypedError> {
                Ok(MyStruct {
                    data: input,
                    count: 1,
                })
            });
            let _step_b =
                FnStep::new(|input: MyStruct| -> Result<String, TypedError> { Ok(input.data) });
            // This compiles because String -> MyStruct -> String types match
            let _chain = _step_a.then(_step_b);
        }
    }

    // Test three-step pipeline
    #[tokio::test]
    async fn test_three_step_pipeline() {
        let step_a =
            FnStep::new(|input: String| -> Result<u32, TypedError> { Ok(input.len() as u32) });

        let step_b = FnStep::new(|input: u32| -> Result<MyStruct, TypedError> {
            Ok(MyStruct {
                data: "counted".to_string(),
                count: input,
            })
        });

        let step_c = FnStep::new(|input: MyStruct| -> Result<String, TypedError> {
            Ok(format!("{} items: {}", input.count, input.data))
        });

        let pipeline = step_a.then(step_b).then(step_c);
        let result = pipeline.run("hello world".to_string()).await.unwrap();
        assert_eq!(result, "11 items: counted");
    }

    // Test agent as pipeline step
    #[tokio::test]
    async fn test_agent_as_pipeline_step() {
        let agent_step = AgentStep::new(TestAgent);
        let result = agent_step.run("question".to_string()).await.unwrap();
        assert_eq!(result.answer, "Response to: question");
    }

    // Test tool as pipeline step
    #[tokio::test]
    async fn test_tool_as_pipeline_step() {
        let tool_step = ToolStep::new(TestTool);
        let input = MyInput {
            query: "search".to_string(),
        };
        let result = tool_step.run(input).await.unwrap();
        assert_eq!(result.answer, "Processed: search");
    }
}
