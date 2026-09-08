//! Tool implementations behind the [`crate::agent::catalog::ToolCatalog`].
//!
//! [`builtins`] holds the in-tree tools shared by the MCP server and the
//! agent executor. User-defined `AgentTool`s are merged in a later task;
//! the catalog's `dispatch`/`specs_for` are written so that addition is
//! additive.

pub mod builtins;
pub mod custom;
pub mod ops;
pub mod web;
