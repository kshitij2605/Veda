//! User-defined tools, dispatched.
//!
//! `AgentTool { id, name, description, kind, config }` has been storable with
//! full CRUD at `/v1/agent-tools` since it was added, but `catalog()`,
//! `specs_for()` and `dispatch_inner()` were all builtin-only — so a stored
//! tool was never advertised to the model and never ran. The struct's own
//! comment said `kind` + opaque `config` were "interpreted by the tool
//! catalog in a later task": there were no semantics to implement, only to
//! choose.
//!
//! **`kind: "search_preset"`** — `config` pins a collection plus search
//! parameters; the caller supplies only the query text. Dispatch runs through
//! the existing `search` builtin, so tenant scoping, RLS, retrieval
//! guardrails and field sensitivity are inherited rather than reimplemented.
//!
//! Deliberately **not** an HTTP-callout design. That is an SSRF surface, and
//! this repo already treats internet reach as dangerous — `web_search` and
//! `fetch_url` are `OPT_IN_BUILTINS`, granted per agent. A callout kind would
//! need a host allowlist plus the same per-agent opt-in before it is safe.

use serde_json::{Value, json};

use super::builtins;
use crate::agent::catalog::{ToolContext, ToolOutcome, ToolSpec};
use crate::agent::model::AgentTool;
use crate::agent::store;
use crate::tenant;

/// Preset keys copied verbatim from `config` into the `search` call. The
/// collection is handled separately because the caller must not override it.
const PRESET_KEYS: [&str; 5] = ["mode", "field", "filter", "limit", "fields"];

/// Load one custom tool for this tenant. `None` when the id names nothing —
/// the caller turns that into the existing "unknown tool" not-found, whose
/// exact wording MCP matches on.
pub async fn load(ctx: &ToolContext<'_>, tool_id: &str) -> Option<AgentTool> {
    let scoped = tenant::scope(ctx.tenant, tool_id);
    store::get_tool(&ctx.state.inner.saved_objects, &scoped).await
}

/// Every custom tool visible to this tenant, with ids unscoped so they match
/// what an agent whitelist and `dispatch` use.
pub async fn list_for_tenant(ctx: &ToolContext<'_>) -> Vec<AgentTool> {
    let mut tools = store::list_tools(&ctx.state.inner.saved_objects).await;
    tools.retain(|t| tenant::strip_prefix(ctx.tenant, &t.id).is_some());
    for t in &mut tools {
        if let Some(name) = tenant::strip_prefix(ctx.tenant, &t.id) {
            t.id = name.to_owned();
        }
    }
    tools
}

/// The collection a preset pins, if any.
///
/// `ToolCatalog::dispatch` needs this *before* dispatching: its retrieval
/// guard and field-sensitivity lookups key off the collection, and they read
/// it from the call's args. A preset's args carry only `query`, so without
/// this the guards would resolve against `None` and silently weaken — the
/// tool would run with less enforcement than the builtin it wraps.
pub fn pinned_collection(tool: &AgentTool) -> Option<String> {
    tool.config
        .get("collection")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// Advertise a stored tool to the model. Without a spec the tool would
/// dispatch correctly and never be called, because the LLM would not know it
/// exists.
pub fn spec_for(tool: &AgentTool) -> ToolSpec {
    let described = if tool.description.trim().is_empty() {
        match pinned_collection(tool) {
            Some(c) => format!("Saved search preset over the '{c}' collection."),
            None => "Saved search preset.".to_owned(),
        }
    } else {
        tool.description.clone()
    };
    ToolSpec {
        id: tool.id.clone(),
        name: tool.name.clone(),
        description: format!(
            "{described} The collection and search parameters are fixed by the preset; \
             supply only the query text."
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query text. The collection, mode, \
                                    field, filter and limit are pinned by the preset."
                }
            },
            "required": ["query"]
        }),
        kind: "custom".to_owned(),
        // A preset can only ever run a read-only `search`.
        read_only: true,
        destructive: false,
        title: Some(tool.name.clone()),
    }
}

/// Route a stored tool by its `kind`.
pub async fn dispatch(ctx: &ToolContext<'_>, tool: &AgentTool, args: Value) -> ToolOutcome {
    match tool.kind.as_str() {
        "search_preset" => search_preset(ctx, tool, args).await,
        // An unknown kind is a stored-configuration error the operator can
        // fix, not a missing tool and not an internal fault.
        other => ToolOutcome::invalid_args(format!(
            "custom tool '{}' has kind '{other}', which this server cannot run; \
             the supported kind is 'search_preset'",
            tool.name
        )),
    }
}

async fn search_preset(ctx: &ToolContext<'_>, tool: &AgentTool, args: Value) -> ToolOutcome {
    let Some(collection) = pinned_collection(tool) else {
        return ToolOutcome::invalid_args(format!(
            "custom tool '{}' is a search_preset but its config pins no `collection`",
            tool.name
        ));
    };
    let Some(query) = args.get("query").and_then(|v| v.as_str()) else {
        return ToolOutcome::invalid_args(
            "this tool needs a `query` string: the preset fixes the collection and search \
             parameters, and the caller supplies the text",
        );
    };

    // Build the builtin call. The pinned values are written LAST-ish here in
    // the sense that the caller supplies nothing but `query` — a preset that
    // could be overridden per call would not be a preset, and letting the
    // model pick the collection is exactly the scoping hole this design
    // avoids.
    let mut search_args = serde_json::Map::new();
    search_args.insert("collection".to_owned(), json!(collection));
    search_args.insert("query".to_owned(), json!(query));
    for key in PRESET_KEYS {
        if let Some(v) = tool.config.get(key) {
            search_args.insert((*key).to_owned(), v.clone());
        }
    }

    builtins::search(ctx, Value::Object(search_args)).await
}
