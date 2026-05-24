// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Procedural macros for ergonomic tool and agent definition.
//!
//! ## `#[tool]` — Turn an `async fn` into a `Tool` implementation
//!
//! Apply `#[tool(name = "...", description = "...")]` to an `async fn` that
//! takes `serde_json::Value` and returns `Result<ai_core::tool::ToolOutput, ai_core::tool::ToolError>`.
//! The macro generates a struct and a `Tool` impl so the function can be added to
//! an agent's tool registry.
//!
//! ### Example
//!
//! ```rust,ignore
//! use ai_tools_macros::tool;
//! use serde_json::Value;
//! use ai_core::tool::{ToolOutput, ToolError};
//!
//! #[tool(name = "greet", description = "Greet a user by name")]
//! async fn greet_user(input: Value) -> Result<ToolOutput, ToolError> {
//!     let name = input["name"].as_str().unwrap_or("world");
//!     Ok(ToolOutput::success(format!("Hello, {name}!")))
//! }
//!
//! // Generates: `pub struct GreetUserTool` implementing `ai_core::tool::Tool`.
//! // Instantiate with `GreetUserTool::new()`.
//! ```

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ItemFn, LitStr, Token,
};

// ─── Attribute argument parsing ──────────────────────────────────────────────

/// Parsed arguments from `#[tool(name = "...", description = "...")]`.
struct ToolArgs {
    name: Option<String>,
    description: Option<String>,
    input_schema: Option<String>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut input_schema = None;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: LitStr = input.parse()?;

            match key.to_string().as_str() {
                "name" => name = Some(value.value()),
                "description" => description = Some(value.value()),
                "input_schema" => input_schema = Some(value.value()),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[tool] key `{other}`. Valid keys: name, description, input_schema"),
                    ))
                }
            }

            // Consume optional trailing comma.
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(ToolArgs {
            name,
            description,
            input_schema,
        })
    }
}

// ─── Main macro ──────────────────────────────────────────────────────────────

/// Derive a [`Tool`](ai_core::tool::Tool) implementation from an `async fn`.
///
/// # Arguments
///
/// | Key | Required | Default | Description |
/// |-----|----------|---------|-------------|
/// | `name` | No | snake_cased function name | Tool name shown to the LLM |
/// | `description` | **Yes** | — | Human-readable description |
/// | `input_schema` | No | `{}` (any object) | JSON Schema string for the input |
///
/// # Generated code
///
/// Given `#[tool(name = "greet", description = "...")]` on `async fn greet_user(...)`,
/// the macro generates:
/// - `pub struct GreetUserTool { descriptor: ai_core::tool::ToolDescriptor }`
/// - `impl GreetUserTool { pub fn new() -> Self { ... } }`
/// - `impl ai_core::tool::Tool for GreetUserTool { ... }`
///
/// The original `async fn` is kept private and called by the `execute` impl.
#[proc_macro_attribute]
pub fn tool(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attrs as ToolArgs);
    let func = parse_macro_input!(item as ItemFn);

    match expand_tool(args, func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_tool(args: ToolArgs, func: ItemFn) -> syn::Result<TokenStream2> {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();

    // Derive tool name: explicit arg or snake_case fn name.
    let tool_name = args.name.unwrap_or_else(|| func_name_str.clone());

    // Description is required.
    let description = args.description.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "#[tool] requires `description = \"...\"` argument",
        )
    })?;

    // Input schema: parse JSON string or fall back to accept-any object.
    let input_schema_json = args
        .input_schema
        .unwrap_or_else(|| r#"{"type":"object"}"#.to_string());

    // Build a PascalCase struct name from the function name.
    let struct_name_str = to_pascal_case(&func_name_str);
    let struct_ident = syn::Ident::new(&struct_name_str, func_name.span());

    let expanded = quote! {
        // Keep the original function.
        #func

        /// Auto-generated Tool wrapper for `#func_name`.
        #[derive(Clone)]
        pub struct #struct_ident {
            descriptor: ai_core::tool::ToolDescriptor,
        }

        impl #struct_ident {
            /// Create a new instance of this tool.
            pub fn new() -> Self {
                let schema: serde_json::Value =
                    serde_json::from_str(#input_schema_json)
                        .expect(concat!("invalid input_schema JSON in #[tool] on ", #func_name_str));

                Self {
                    descriptor: ai_core::tool::ToolDescriptor::new(
                        #tool_name,
                        #description,
                        schema,
                    ),
                }
            }
        }

        impl Default for #struct_ident {
            fn default() -> Self {
                Self::new()
            }
        }

        #[async_trait::async_trait]
        impl ai_core::tool::Tool for #struct_ident {
            fn descriptor(&self) -> ai_core::tool::ToolDescriptor {
                self.descriptor.clone()
            }

            async fn execute(
                &self,
                input: serde_json::Value,
            ) -> Result<ai_core::tool::ToolOutput, ai_core::error::ToolError> {
                #func_name(input).await
            }
        }
    };

    Ok(expanded)
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Convert `snake_case` to `PascalCase`, appending "Tool" suffix.
///
/// e.g. `greet_user` → `GreetUserTool`
fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut cap_next = true;
    for ch in s.chars() {
        if ch == '_' {
            cap_next = true;
        } else if cap_next {
            result.extend(ch.to_uppercase());
            cap_next = false;
        } else {
            result.push(ch);
        }
    }
    result.push_str("Tool");
    result
}
