# RAG Eval Auto-Generation v2 + Dual-Mode Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a dedicated, semaphore-limited answer-generation LLM; make eval-set generation diverse across documents and content types with equal easy/medium/hard proportions; add an opt-in per-collection auto-trigger that regenerates the testset and runs a tuning campaign every time a collection grows; give Niriksha a corpus-aware testset-size control; and add a performance-only tuning mode alongside the existing quality-weighted one.

**Architecture:** Extends the already-shipped RAG eval/tuning subsystem (`crates/veda-gateway/src/rest/rag_eval.rs`, `crates/veda-gateway/src/tuning/`) — no new storage subsystem (auto-optimization config persists via the same `SavedObjectStore` campaigns/eval-sets already use); a new dedicated `LlmConfig` instance + `tokio::sync::Semaphore` bound the production answer-gen endpoint; the auto-trigger is a detached background task spawned from the shared `ingest_common::ingest_markdown` tail.

**Tech Stack:** Rust (axum, tokio, serde) for the gateway; TypeScript/React (Vite) for Niriksha.

## Global Constraints

- `ANSWER_GEN_BASE_URL` / `ANSWER_GEN_API_KEY` / `ANSWER_GEN_MODEL` are a **dedicated** LLM config, separate from any other LLM config in the gateway. Live values (production, shared infra): `ANSWER_GEN_BASE_URL=http://llm-host.internal.example:9000/v1`, `ANSWER_GEN_API_KEY=tensorflow`, `ANSWER_GEN_MODEL=Qwen/Qwen3.5-122B-A10B`. Confirmed `max_model_len: 128000`.
- Concurrency to the answer-gen endpoint is bounded by a fixed `tokio::sync::Semaphore(5)` — never configurable in this iteration (YAGNI).
- Auto-optimization is opt-in per collection, **default `false`**.
- When enabled + answer-gen configured: fully hands-off — generation chains directly into starting a tuning campaign, no manual step.
- Re-triggers on **every** ingestion call that adds data (not just the first); each trigger **regenerates the testset from scratch** at the newly-recommended size.
- Auto-trigger hooks only the document ingestion paths (`process-document(s)`, `ingest-url` — both funnel through `ingest_common::ingest_markdown`), not `/insert`/`/embed-insert`.
- Performance-only tuning mode keeps a quality guard: minimizes p50 latency only among trials whose recall clears a floor derived from the live collection's own current (tier1-default) recall — never a fixed magic number.
- Real-data verification (final task) uses the already-ingested `rag_test` collection on gpu-host-c (184 chunks / 18 documents from `/home/mac/25gitlab/rag_anything/bulk_module/tmp`).

---

### Task 1: Dedicated answer-gen LLM config + semaphore wiring

**Files:**
- Modify: `crates/veda-core/src/llm_client.rs` (add `LlmConfig::from_env_answer_gen()`)
- Modify: `crates/veda-gateway/src/rest/state.rs` (add `answer_gen_config` + `answer_gen_semaphore` fields, `load_answer_gen_from_env()` method)
- Modify: `crates/veda-gateway/src/server.rs` (call the new loader at startup)
- Test: `crates/veda-core/src/llm_client.rs` (inline `#[cfg(test)]`), `crates/veda-gateway/src/rest/state.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Consumes: `LlmConfig` (existing struct: `{ endpoint: String, api_key: Option<String>, model: String, timeout: Duration }`), `normalize_endpoint(&str) -> String` (existing private fn in `llm_client.rs`).
- Produces: `LlmConfig::from_env_answer_gen() -> Option<LlmConfig>`; `AppStateInner.answer_gen_config: RwLock<Option<LlmConfig>>`; `AppStateInner.answer_gen_semaphore: Arc<tokio::sync::Semaphore>`; `AppState::load_answer_gen_from_env(&self) -> impl Future<Output = ()>` (async method, no return value).

- [ ] **Step 1: Write the failing test for `from_env_answer_gen`**

In `crates/veda-core/src/llm_client.rs`, inside the existing `#[cfg(test)] mod tests` block (find it near the bottom of the file), add:

```rust
#[test]
fn from_env_answer_gen_reads_dedicated_prefix() {
    // SAFETY: tests run single-threaded within this module via serial env var mutation;
    // mirrors the existing `from_env_reads_...`-style tests in this same block.
    unsafe {
        std::env::remove_var("VEDA_LLM_ENDPOINT");
        std::env::set_var("ANSWER_GEN_BASE_URL", "http://llm-host.internal.example:9000/v1");
        std::env::set_var("ANSWER_GEN_API_KEY", "tensorflow");
        std::env::set_var("ANSWER_GEN_MODEL", "Qwen/Qwen3.5-122B-A10B");
    }
    let cfg = LlmConfig::from_env_answer_gen().expect("answer-gen config should be present");
    assert_eq!(cfg.endpoint, "http://llm-host.internal.example:9000/v1/chat/completions");
    assert_eq!(cfg.model, "Qwen/Qwen3.5-122B-A10B");
    assert_eq!(cfg.api_key.as_deref(), Some("tensorflow"));
    unsafe {
        std::env::remove_var("ANSWER_GEN_BASE_URL");
        std::env::remove_var("ANSWER_GEN_API_KEY");
        std::env::remove_var("ANSWER_GEN_MODEL");
    }
}

#[test]
fn from_env_answer_gen_none_when_unset() {
    unsafe {
        std::env::remove_var("ANSWER_GEN_BASE_URL");
    }
    assert!(LlmConfig::from_env_answer_gen().is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p veda-core from_env_answer_gen`
Expected: FAIL with "no function or associated item named `from_env_answer_gen` found"

- [ ] **Step 3: Implement `from_env_answer_gen`**

In `crates/veda-core/src/llm_client.rs`, inside `impl LlmConfig` (right after the existing `from_env` method), add:

```rust
    /// Read a DEDICATED answer-generation LLM config from `ANSWER_GEN_*`
    /// env vars — intentionally separate from `VEDA_LLM_*` (used by
    /// agents/triage/faithfulness-judging elsewhere), since eval-set
    /// generation may run against different, production-shared infra
    /// with its own concurrency budget (see `AppStateInner.answer_gen_semaphore`).
    /// Returns `None` if `ANSWER_GEN_BASE_URL` is unset — eval-set
    /// generation and the auto-optimization trigger both no-op cleanly.
    pub fn from_env_answer_gen() -> Option<Self> {
        let endpoint = std::env::var("ANSWER_GEN_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|e| normalize_endpoint(&e))?;
        let model = std::env::var("ANSWER_GEN_MODEL").unwrap_or_else(|_| "default".to_owned());
        let api_key = std::env::var("ANSWER_GEN_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        let timeout = std::env::var("ANSWER_GEN_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));
        Some(Self {
            endpoint,
            api_key,
            model,
            timeout,
        })
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p veda-core from_env_answer_gen`
Expected: PASS (2 tests)

- [ ] **Step 5: Add `answer_gen_config` + `answer_gen_semaphore` to `AppStateInner`**

In `crates/veda-gateway/src/rest/state.rs`, find the line `pub llm_config: RwLock<Option<LlmConfig>>,` (there are 3 construction sites initializing it to `RwLock::new(None)` — find all 3 via `grep -n llm_config crates/veda-gateway/src/rest/state.rs`). Add a sibling field declaration immediately after `pub llm_config: RwLock<Option<LlmConfig>>,`:

```rust
    /// Dedicated answer-generation LLM — separate from `llm_config`
    /// (used by agents/triage). Populated from `ANSWER_GEN_*` env vars.
    pub answer_gen_config: RwLock<Option<LlmConfig>>,
    /// Bounds concurrent calls to the answer-gen endpoint. Fixed at 5
    /// permits (not configurable in v1) — this is shared production
    /// infra, not a local test double.
    pub answer_gen_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
```

At EACH of the 3 struct-literal construction sites that currently have `llm_config: RwLock::new(None),`, add immediately after:

```rust
                answer_gen_config: RwLock::new(None),
                answer_gen_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(5)),
```

(Match the existing indentation at each site exactly — these are struct literals inside different constructor functions, so indentation levels may differ; use the surrounding `llm_config: RwLock::new(None),` line's own indentation as the reference for each site.)

- [ ] **Step 6: Add `load_answer_gen_from_env`**

In `crates/veda-gateway/src/rest/state.rs`, immediately after the existing `pub async fn load_llm_from_env(&self)` method body's closing brace, add:

```rust

    /// Load the dedicated answer-generation LLM endpoint from
    /// `ANSWER_GEN_*` env vars. No-op when unset — eval-set generation's
    /// `generate_one_qa` then returns `None` per-case and
    /// `generate_eval_set` degrades to an empty-cases 503, exactly
    /// mirroring `load_llm_from_env`'s no-op contract.
    pub async fn load_answer_gen_from_env(&self) {
        let Some(cfg) = LlmConfig::from_env_answer_gen() else {
            return;
        };
        let endpoint = cfg.endpoint.clone();
        let model = cfg.model.clone();
        let mut guard = self.inner.answer_gen_config.write().await;
        *guard = Some(cfg);
        tracing::info!(model = %model, endpoint = %endpoint, "registered answer-gen LLM endpoint from environment");
    }
```

- [ ] **Step 7: Wire the loader at server startup**

In `crates/veda-gateway/src/server.rs`, find `app_state.load_llm_from_env().await;` (around line 39) and add immediately after:

```rust
    app_state.load_answer_gen_from_env().await;
```

- [ ] **Step 8: Build and run the full state.rs/server.rs test suite**

Run: `cargo build -p veda-gateway && cargo test -p veda-gateway --lib rest::state:: `
Expected: builds clean, all existing state tests still pass (no behavior change for existing fields)

- [ ] **Step 9: Commit**

```bash
git add crates/veda-core/src/llm_client.rs crates/veda-gateway/src/rest/state.rs crates/veda-gateway/src/server.rs
git commit -m "feat(rag-eval): dedicated answer-gen LLM config + concurrency semaphore"
```

---

### Task 2: Diverse, document-and-type-aware, equal-thirds chunk sampling

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs` (extend `SourceChunk`, replace `load_source_chunks`'s flat sampling with grouped/weighted/tagged sampling, relabel difficulty, thread the semaphore + dedicated config into `generate_one_qa`)
- Test: `crates/veda-gateway/src/rest/rag_eval.rs` (inline `#[cfg(test)] mod generation_tests`)

**Interfaces:**
- Consumes: `Record::get(&self, field: &str) -> Option<&FieldValue>` (existing, `veda-core`); `AppStateInner.answer_gen_config`/`answer_gen_semaphore` (Task 1).
- Produces: `SourceChunk { chunk_id: String, text: String, parent_doc_id: Option<String> }` (extended); `fn sample_diverse_chunks(chunks: Vec<SourceChunk>, target_count: usize) -> Vec<SourceChunk>`; `fn document_groups(chunks: &[SourceChunk]) -> Vec<DocumentGroup>` (reused by Task 3); `struct DocumentGroup { key: String, chunk_ids: Vec<String>, total_len: usize }`; difficulty labels become `"easy"`/`"medium"`/`"hard"`.

- [ ] **Step 1: Write the failing tests for document-weighted, type-diverse sampling**

In `crates/veda-gateway/src/rest/rag_eval.rs`, inside the existing `#[cfg(test)] mod generation_tests` block, add:

```rust
    fn chunk_with_doc(id: &str, text: &str, doc: &str) -> SourceChunk {
        SourceChunk {
            chunk_id: id.to_owned(),
            text: text.to_owned(),
            parent_doc_id: Some(doc.to_owned()),
        }
    }

    #[test]
    fn document_groups_weighs_by_total_text_length() {
        let chunks = vec![
            chunk_with_doc("c1", "short", "docA"),
            chunk_with_doc("c2", "a much longer piece of text than the others here", "docB"),
            chunk_with_doc("c3", "also short", "docA"),
        ];
        let groups = document_groups(&chunks);
        assert_eq!(groups.len(), 2, "two distinct parent_doc_ids -> two groups");
        let doc_b = groups.iter().find(|g| g.key == "docB").unwrap();
        let doc_a = groups.iter().find(|g| g.key == "docA").unwrap();
        assert!(
            doc_b.total_len > doc_a.total_len,
            "docB's single long chunk should outweigh docA's two short ones"
        );
    }

    #[test]
    fn document_groups_treats_missing_parent_doc_id_as_its_own_singleton_group() {
        let chunks = vec![SourceChunk {
            chunk_id: "orphan1".to_owned(),
            text: "no parent doc id here".to_owned(),
            parent_doc_id: None,
        }];
        let groups = document_groups(&chunks);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].key, "orphan1", "orphan chunks key their own singleton group by chunk id");
    }

    #[test]
    fn sample_diverse_chunks_allocates_proportionally_to_document_length() {
        // docA: 1 short chunk. docB: 3 long chunks (much more total text).
        // A target of 8 should give docB noticeably more than docA, not an even 4/4 split.
        let long = "x".repeat(500);
        let chunks = vec![
            chunk_with_doc("a1", "short", "docA"),
            chunk_with_doc("b1", &long, "docB"),
            chunk_with_doc("b2", &long, "docB"),
            chunk_with_doc("b3", &long, "docB"),
        ];
        let sampled = sample_diverse_chunks(chunks, 8);
        // Only 4 distinct chunks exist; sampling target 8 should return all 4
        // (can't exceed available source chunks), but at least confirm both
        // documents are represented (diversity requirement).
        let doc_a_count = sampled.iter().filter(|c| c.parent_doc_id.as_deref() == Some("docA")).count();
        let doc_b_count = sampled.iter().filter(|c| c.parent_doc_id.as_deref() == Some("docB")).count();
        assert!(doc_a_count >= 1, "docA must be represented at least once");
        assert!(doc_b_count >= 1, "docB must be represented at least once");
    }

    #[test]
    fn sample_diverse_chunks_tags_and_surfaces_table_and_image_chunks() {
        let chunks = vec![
            chunk_with_doc("t1", "plain prose paragraph about a topic", "doc1"),
            chunk_with_doc("t2", "| col1 | col2 |\n|---|---|\n| a | b |", "doc1"),
            chunk_with_doc("t3", "see figure: ![caption](images/1.jpg)", "doc1"),
        ];
        let sampled = sample_diverse_chunks(chunks, 3);
        assert_eq!(sampled.len(), 3, "all 3 available chunks should be returned when target == available count");
        let ids: Vec<&str> = sampled.iter().map(|c| c.chunk_id.as_str()).collect();
        assert!(ids.contains(&"t2"), "the table chunk must not be silently dropped");
        assert!(ids.contains(&"t3"), "the image chunk must not be silently dropped");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p veda-gateway --lib rest::rag_eval::generation_tests:: document_groups`
Expected: FAIL with "cannot find function `document_groups`" (and similarly for `sample_diverse_chunks`)

- [ ] **Step 3: Extend `SourceChunk` and implement the grouping/sampling functions**

In `crates/veda-gateway/src/rest/rag_eval.rs`, find the existing struct:

```rust
pub struct SourceChunk {
    pub chunk_id: String,
    pub text: String,
}
```

Replace it with:

```rust
pub struct SourceChunk {
    pub chunk_id: String,
    pub text: String,
    /// The document this chunk was chunked from (the `parent_doc_id`
    /// field stamped by `ingest_common::ingest_markdown` — see the
    /// 2026-07-05 real-data E2E fix). `None` for chunks inserted outside
    /// the document pipeline (e.g. raw `/insert`), which then each form
    /// their own singleton "document" group in `document_groups`.
    pub parent_doc_id: Option<String>,
}
```

Immediately after the `SourceChunk` struct, add:

```rust
/// One document's worth of chunks, for proportional-by-length sampling.
/// `key` is the `parent_doc_id` when present, else the lone chunk's own
/// id (an orphan chunk is its own singleton "document").
pub struct DocumentGroup {
    pub key: String,
    pub chunk_ids: Vec<String>,
    pub total_len: usize,
}

/// Group chunks by `parent_doc_id` (orphans become singleton groups),
/// computing each group's total text length as its weight. Shared by
/// `sample_diverse_chunks` (Task 2) and the size-recommendation endpoint
/// (Task 3) so both use the identical weighting.
pub fn document_groups(chunks: &[SourceChunk]) -> Vec<DocumentGroup> {
    use std::collections::HashMap;
    let mut groups: HashMap<String, DocumentGroup> = HashMap::new();
    for c in chunks {
        let key = c.parent_doc_id.clone().unwrap_or_else(|| c.chunk_id.clone());
        let entry = groups.entry(key.clone()).or_insert_with(|| DocumentGroup {
            key,
            chunk_ids: Vec::new(),
            total_len: 0,
        });
        entry.chunk_ids.push(c.chunk_id.clone());
        entry.total_len += c.text.len();
    }
    let mut out: Vec<DocumentGroup> = groups.into_values().collect();
    out.sort_by(|a, b| a.key.cmp(&b.key)); // deterministic order for tests
    out
}

/// Cheap heuristic element-type tag for a chunk, used only to spread
/// generated questions across content types rather than only ever
/// drawing from the majority type (usually prose).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum ElementType {
    Table,
    Image,
    Text,
}

fn classify_element(text: &str) -> ElementType {
    if text.contains("![") && text.contains("](") {
        return ElementType::Image;
    }
    let pipe_lines = text
        .lines()
        .filter(|l| l.matches('|').count() >= 2)
        .count();
    if pipe_lines >= 2 {
        return ElementType::Table;
    }
    ElementType::Text
}

/// Largest-remainder apportionment of `target_count` across `weights`
/// (parallel to `groups`), guaranteeing the returned counts sum to
/// exactly `target_count.min(total chunks available)`.
fn apportion(weights: &[usize], target_count: usize) -> Vec<usize> {
    let total_weight: usize = weights.iter().sum();
    if total_weight == 0 || weights.is_empty() {
        return vec![0; weights.len()];
    }
    let mut counts = vec![0usize; weights.len()];
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(weights.len());
    let mut allocated = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        let exact = target_count as f64 * (w as f64 / total_weight as f64);
        let floor = exact.floor() as usize;
        counts[i] = floor;
        allocated += floor;
        remainders.push((i, exact - floor as f64));
    }
    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut remaining = target_count.saturating_sub(allocated);
    for (i, _) in remainders {
        if remaining == 0 {
            break;
        }
        counts[i] += 1;
        remaining -= 1;
    }
    counts
}

/// Diverse, corpus-proportional chunk sampler: groups chunks by document
/// (weighted by total text length, per the auto-generation design's
/// "a short doc gets fewer questions than a long one" requirement),
/// apportions `target_count` across groups via largest-remainder, then
/// within each group's allocation draws round-robin across whichever
/// element types (table/image/text) are actually present so generation
/// isn't dominated by whichever type happens to be most common.
pub fn sample_diverse_chunks(chunks: Vec<SourceChunk>, target_count: usize) -> Vec<SourceChunk> {
    if chunks.is_empty() || target_count == 0 {
        return Vec::new();
    }
    let by_id: std::collections::HashMap<String, SourceChunk> =
        chunks.into_iter().map(|c| (c.chunk_id.clone(), c)).collect();
    let groups = {
        let all: Vec<SourceChunk> = by_id.values().cloned().collect();
        document_groups(&all)
    };
    let weights: Vec<usize> = groups.iter().map(|g| g.total_len.max(1)).collect();
    let allocations = apportion(&weights, target_count);

    let mut out = Vec::new();
    for (group, alloc) in groups.iter().zip(allocations.iter()) {
        if *alloc == 0 {
            continue;
        }
        let mut group_chunks: Vec<&SourceChunk> = group
            .chunk_ids
            .iter()
            .filter_map(|id| by_id.get(id))
            .collect();
        // Stable order, then bucket by element type for round-robin diversity.
        group_chunks.sort_by(|a, b| a.chunk_id.cmp(&b.chunk_id));
        let mut by_type: std::collections::HashMap<ElementType, Vec<&SourceChunk>> =
            std::collections::HashMap::new();
        for c in &group_chunks {
            by_type.entry(classify_element(&c.text)).or_default().push(c);
        }
        let mut type_order: Vec<ElementType> = by_type.keys().copied().collect();
        type_order.sort_by_key(|t| format!("{t:?}"));
        let mut type_idx = 0usize;
        let mut cursors: std::collections::HashMap<ElementType, usize> =
            std::collections::HashMap::new();
        let mut picked = 0usize;
        while picked < *alloc && !type_order.is_empty() {
            let ty = type_order[type_idx % type_order.len()];
            let bucket = by_type.get(&ty).unwrap();
            let cursor = cursors.entry(ty).or_insert(0);
            if *cursor < bucket.len() {
                out.push(bucket[*cursor].clone());
                *cursor += 1;
                picked += 1;
            }
            type_idx += 1;
            if type_order.iter().all(|t| {
                *cursors.get(t).unwrap_or(&0) >= by_type.get(t).map(|b| b.len()).unwrap_or(0)
            }) {
                break; // every bucket in this group exhausted
            }
        }
    }
    out
}
```

Note: `SourceChunk` needs `Clone` for the sampler above — find its derive attribute (`#[derive(...)]` immediately above `pub struct SourceChunk`) and add `Clone` if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p veda-gateway --lib rest::rag_eval::generation_tests::`
Expected: PASS (all tests in this module, including the 4 new ones)

- [ ] **Step 5: Wire `sample_diverse_chunks` into `load_source_chunks` and stamp `parent_doc_id`**

In `crates/veda-gateway/src/rest/rag_eval.rs`, find `load_source_chunks`'s two `out.push(SourceChunk { chunk_id: ..., text })` call sites (one under the growing-segment loop, one under the sealed-segments loop). Change BOTH to also read `parent_doc_id`:

```rust
                    out.push(SourceChunk {
                        chunk_id: record.id().to_string(),
                        text,
                        parent_doc_id: record
                            .get("parent_doc_id")
                            .and_then(|v| match v {
                                veda_core::record::FieldValue::String(s) => Some(s.clone()),
                                _ => None,
                            }),
                    });
```

(Match this shape at both call sites — the field list is identical, only the surrounding loop differs.)

In `generate_eval_set`, find:

```rust
    let mix = default_difficulty_mix();
    let cases = generate_cases(&chunks, &mix, req.num_cases, move |chunk, difficulty| {
```

Change to sample diverse chunks BEFORE generating cases:

```rust
    let sampled = sample_diverse_chunks(chunks, req.num_cases);
    let mix = default_difficulty_mix();
    let cases = generate_cases(&sampled, &mix, req.num_cases, move |chunk, difficulty| {
```

- [ ] **Step 6: Relabel the difficulty mix to easy/medium/hard**

In `crates/veda-gateway/src/rest/rag_eval.rs`, find:

```rust
fn default_difficulty_mix() -> Vec<(String, f64)> {
    vec![
        ("simple".to_owned(), 0.4),
        ("reasoning".to_owned(), 0.4),
        ("multi_context".to_owned(), 0.2),
    ]
}
```

Replace with:

```rust
/// Equal-weighted easy/medium/hard split. `generate_cases`'s sampling
/// loop round-robins by index (not by weight), so three equal-weight
/// labels genuinely produce ~equal thirds in the output — this was
/// already true before this rename (the previous 40/40/20 weights were
/// silently ignored by the round-robin), just mislabeled as RAGAS-style
/// simple/reasoning/multi_context instead of the easy/medium/hard the
/// product actually wants.
fn default_difficulty_mix() -> Vec<(String, f64)> {
    vec![
        ("easy".to_owned(), 1.0),
        ("medium".to_owned(), 1.0),
        ("hard".to_owned(), 1.0),
    ]
}
```

Update the prompt in `generate_one_qa` (see Step 7 below) to use these new labels — check it doesn't hardcode `"simple"`/`"reasoning"`/`"multi_context"` anywhere else in the file via `grep -n 'simple\|multi_context' crates/veda-gateway/src/rest/rag_eval.rs`; if any other reference exists outside tests, update it to the new labels too.

- [ ] **Step 7: Thread the answer-gen config + semaphore into `generate_one_qa`**

In `crates/veda-gateway/src/rest/rag_eval.rs`, find:

```rust
async fn generate_one_qa(
    llm_config: &Option<veda_core::llm_client::LlmConfig>,
    chunk_text: &str,
    difficulty: &str,
) -> Option<GeneratedQa> {
    let config = llm_config.as_ref()?;
    let client = veda_core::llm_client::LlmClient::new(config.clone());
```

Replace with:

```rust
async fn generate_one_qa(
    llm_config: &Option<veda_core::llm_client::LlmConfig>,
    semaphore: &std::sync::Arc<tokio::sync::Semaphore>,
    chunk_text: &str,
    difficulty: &str,
) -> Option<GeneratedQa> {
    let config = llm_config.as_ref()?;
    // Bound concurrency against the (shared, production) answer-gen
    // endpoint — never more than `semaphore`'s permit count in flight
    // regardless of how many chunks are being processed at once.
    let _permit = semaphore.acquire().await.ok()?;
    let client = veda_core::llm_client::LlmClient::new(config.clone());
```

In `generate_eval_set`, find:

```rust
    let llm_config = state.inner.llm_config.read().await.clone();
    if llm_config.is_none() {
        return Err(service_unavailable(
            "VEDA_LLM_ENDPOINT not configured. Set the env var to enable eval-set generation.",
        ));
    }
```

Replace with:

```rust
    let llm_config = state.inner.answer_gen_config.read().await.clone();
    if llm_config.is_none() {
        return Err(service_unavailable(
            "ANSWER_GEN_BASE_URL not configured. Set the env var to enable eval-set generation.",
        ));
    }
    let semaphore = state.inner.answer_gen_semaphore.clone();
```

And find the closure that calls `generate_one_qa`:

```rust
    let cases = generate_cases(&sampled, &mix, req.num_cases, move |chunk, difficulty| {
        let llm_config = llm_config.clone();
        let text = chunk.text.clone();
        let difficulty = difficulty.to_owned();
        async move { generate_one_qa(&llm_config, &text, &difficulty).await }
    })
    .await;
```

Replace with:

```rust
    let cases = generate_cases(&sampled, &mix, req.num_cases, move |chunk, difficulty| {
        let llm_config = llm_config.clone();
        let semaphore = semaphore.clone();
        let text = chunk.text.clone();
        let difficulty = difficulty.to_owned();
        async move { generate_one_qa(&llm_config, &semaphore, &text, &difficulty).await }
    })
    .await;
```

- [ ] **Step 8: Fix the existing `generate_one_qa` unit tests' call signature**

Search `crates/veda-gateway/src/rest/rag_eval.rs` for any direct test calls to `generate_one_qa(` (there may be none if it's only exercised indirectly through `generate_cases`'s injected closure in tests — confirm via `grep -n 'generate_one_qa(' crates/veda-gateway/src/rest/rag_eval.rs`). If any direct test call exists, update it to pass a `&std::sync::Arc::new(tokio::sync::Semaphore::new(5))` as the new second argument.

- [ ] **Step 9: Run the full rag_eval test suite**

Run: `cargo test -p veda-gateway --lib rest::rag_eval::`
Expected: PASS (every existing + new test in this module)

- [ ] **Step 10: Run clippy + fmt**

Run: `cargo clippy -p veda-gateway --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean (fix any warnings/formatting before continuing)

- [ ] **Step 11: Commit**

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs
git commit -m "feat(rag-eval): diverse document/element-type-weighted chunk sampling, easy/medium/hard difficulty"
```

---

### Task 3: Eval-set-size-recommendation endpoint

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs` (new handler + response struct, reusing Task 2's `document_groups`)
- Modify: `crates/veda-gateway/src/rest/mod.rs` (route registration)
- Test: `crates/veda-gateway/tests/rag_eval_test.rs` (new integration test — check this file exists via `ls crates/veda-gateway/tests/ | grep rag_eval`; if a differently-named existing eval-set integration test file exists instead, add the test there and note the actual filename in the commit)

**Interfaces:**
- Consumes: `document_groups`, `load_source_chunks` (Task 2, same file).
- Produces: `GET /v1/collections/{name}/eval-set-size-recommendation` → `{ recommended: usize, max: usize, chunk_count: usize, document_count: usize }`.

- [ ] **Step 1: Write the failing integration test**

Find the existing eval-set integration test file: run `ls crates/veda-gateway/tests/ | grep -i eval`. Add this test to that file (or create `crates/veda-gateway/tests/rag_eval_recommendation_test.rs` mirroring that file's `start_server`/`Client` boilerplate if none exists):

```rust
#[tokio::test]
async fn eval_set_size_recommendation_reflects_real_corpus_shape() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "reco_test",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "content", "type": "string" },
                { "name": "parent_doc_id", "type": "string", "filterable": true }
            ],
            "seed_records": [
                { "v": [1.0, 0.0], "content": "short doc one chunk", "parent_doc_id": "docA" },
                { "v": [0.0, 1.0], "content": "a considerably longer chunk of text for document two", "parent_doc_id": "docB" },
                { "v": [1.0, 1.0], "content": "another long chunk continuing document two's content here", "parent_doc_id": "docB" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    let resp = c
        .get(format!("{base}/v1/collections/reco_test/eval-set-size-recommendation"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "recommendation: {}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["chunk_count"], json!(3));
    assert_eq!(body["document_count"], json!(2));
    assert_eq!(body["max"], json!(3), "max must never exceed the collection's chunk count");
    assert!(
        body["recommended"].as_u64().unwrap() <= 3,
        "recommended must not exceed the chunk count either; got {body}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p veda-gateway --test <the_file_name_from_step_1> eval_set_size_recommendation`
Expected: FAIL with 404 (route not registered yet)

- [ ] **Step 3: Implement the handler**

In `crates/veda-gateway/src/rest/rag_eval.rs`, add (near `generate_eval_set`, after it):

```rust
#[derive(Serialize)]
pub struct EvalSetSizeRecommendation {
    pub recommended: usize,
    pub max: usize,
    pub chunk_count: usize,
    pub document_count: usize,
}

/// Recommend a testset size for `collection`: roughly 2 questions per
/// document (the product's stated "10 docs -> 20 questions" default),
/// but length-adjusted the same way `sample_diverse_chunks` weights its
/// allocation, so a corpus of mostly-short documents doesn't recommend
/// an unreasonably large set. `max` is always the collection's own
/// chunk count — Niriksha enforces this as a hard ceiling on the
/// user-editable size input.
pub async fn eval_set_size_recommendation(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
) -> Result<Json<EvalSetSizeRecommendation>, GatewayError> {
    let scoped = tenant::scope(&tenant, &collection);
    let chunks = load_source_chunks(&state, &scoped).await?;
    let groups = document_groups(&chunks);
    let document_count = groups.len();
    let chunk_count = chunks.len();
    // Base recommendation: 2 questions/document, floor 4 (a corpus of 1-2
    // tiny docs should still get a usable eval set), ceiling at chunk_count.
    let base = (document_count * 2).max(4);
    let recommended = base.min(chunk_count);
    Ok(Json(EvalSetSizeRecommendation {
        recommended,
        max: chunk_count,
        chunk_count,
        document_count,
    }))
}
```

Add `use crate::tenant;` and `use crate::tenant::Tenant;` at the top of the file if not already imported (check via `grep -n '^use crate::tenant' crates/veda-gateway/src/rest/rag_eval.rs` first — `generate_eval_set` already uses `Tenant`/`tenant::scope`, so these are almost certainly already imported; only add if missing).

- [ ] **Step 4: Register the route**

In `crates/veda-gateway/src/rest/mod.rs`, find the existing route registration:

```rust
        .route(
            "/v1/collections/{name}/eval-sets/generate",
            post(rag_eval::generate_eval_set),
        )
```

Add immediately after:

```rust
        .route(
            "/v1/collections/{name}/eval-set-size-recommendation",
            get(rag_eval::eval_set_size_recommendation),
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p veda-gateway --test <the_file_name_from_step_1> eval_set_size_recommendation`
Expected: PASS

- [ ] **Step 6: Run clippy + fmt**

Run: `cargo clippy -p veda-gateway --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs crates/veda-gateway/src/rest/mod.rs crates/veda-gateway/tests/
git commit -m "feat(rag-eval): eval-set-size-recommendation endpoint"
```

---

### Task 4: Auto-optimization config store + REST

**Files:**
- Create: `crates/veda-gateway/src/rest/auto_optimization.rs`
- Modify: `crates/veda-gateway/src/rest/mod.rs` (module declaration + routes)

**Interfaces:**
- Consumes: `SavedObjectStore::{put, get}` (existing, `crates/veda-gateway/src/rest/saved_objects.rs`); `TuningMode` (Task 6 — this task defines a placeholder-free `Quality`/`Performance` enum locally first since Task 6 needs it too; Task 6 will re-home shared logic if needed, but the enum itself is defined ONCE here and imported by Task 6, since config lives here).
- Produces: `AutoOptimizationConfig { collection: String, enabled: bool, target_questions: Option<usize>, mode: TuningMode }`; `pub async fn get_auto_optimization_config(store: &SavedObjectStore, scoped_collection: &str) -> Option<AutoOptimizationConfig>`; `pub async fn save_auto_optimization_config(store: &SavedObjectStore, cfg: &AutoOptimizationConfig)`; REST `GET/PUT /v1/collections/{name}/auto-optimization`.

- [ ] **Step 1: Write the failing test**

Create `crates/veda-gateway/src/rest/auto_optimization.rs` with just the test module first:

```rust
//! Opt-in, per-collection auto-optimization configuration: when enabled
//! (default `false`) and the dedicated answer-gen LLM is configured,
//! `ingest_common::ingest_markdown` (Task 5) automatically regenerates
//! this collection's eval set and starts a tuning campaign after every
//! ingestion that adds data. Persists via the same `SavedObjectStore`
//! campaigns/eval-sets already use — no new storage subsystem.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::rest::saved_objects::{SavedObject, SavedObjectStore};
use crate::tenant::{self, Tenant};

pub const TYPE_AUTO_OPTIMIZATION_CONFIG: &str = "auto_optimization_config";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TuningMode {
    /// Multi-objective (recall, faithfulness, latency) Pareto search —
    /// today's existing behavior.
    Quality,
    /// Same evaluation loop, but reports the single lowest-latency trial
    /// among those clearing a recall floor (see `tuning::campaign`).
    Performance,
}

impl Default for TuningMode {
    fn default() -> Self {
        Self::Quality
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoOptimizationConfig {
    pub collection: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_questions: Option<usize>,
    #[serde(default)]
    pub mode: TuningMode,
}

pub async fn get_auto_optimization_config(
    store: &SavedObjectStore,
    scoped_collection: &str,
) -> Option<AutoOptimizationConfig> {
    let obj = store.get(scoped_collection).await?;
    if obj.kind != TYPE_AUTO_OPTIMIZATION_CONFIG {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

pub async fn save_auto_optimization_config(store: &SavedObjectStore, cfg: &AutoOptimizationConfig) {
    let obj = SavedObject {
        id: cfg.collection.clone(),
        kind: TYPE_AUTO_OPTIMIZATION_CONFIG.to_owned(),
        name: format!("auto-optimization config for {}", cfg.collection),
        body: serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    };
    store.put(obj).await;
}

pub async fn get_auto_optimization(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
) -> Json<AutoOptimizationConfig> {
    let scoped = tenant::scope(&tenant, &collection);
    let cfg = get_auto_optimization_config(&state.inner.saved_objects, &scoped)
        .await
        .unwrap_or(AutoOptimizationConfig {
            collection: collection.clone(),
            enabled: false,
            target_questions: None,
            mode: TuningMode::default(),
        });
    Json(cfg)
}

#[derive(Deserialize)]
pub struct SetAutoOptimizationRequest {
    pub enabled: bool,
    #[serde(default)]
    pub target_questions: Option<usize>,
    #[serde(default)]
    pub mode: TuningMode,
}

pub async fn set_auto_optimization(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<SetAutoOptimizationRequest>,
) -> Result<Json<AutoOptimizationConfig>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &collection);
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{collection}' not found"),
                }),
            ));
        }
    }
    let cfg = AutoOptimizationConfig {
        collection: scoped.clone(),
        enabled: req.enabled,
        target_questions: req.target_questions,
        mode: req.mode,
    };
    save_auto_optimization_config(&state.inner.saved_objects, &cfg).await;
    Ok(Json(AutoOptimizationConfig {
        collection: collection.clone(),
        ..cfg
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::state::AppState;

    fn test_state() -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        AppState::new(tmp.path())
    }

    #[tokio::test]
    async fn save_and_get_round_trips() {
        let state = test_state();
        let cfg = AutoOptimizationConfig {
            collection: "default__mycol".to_owned(),
            enabled: true,
            target_questions: Some(30),
            mode: TuningMode::Performance,
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg).await;
        let loaded = get_auto_optimization_config(&state.inner.saved_objects, "default__mycol")
            .await
            .expect("config should round-trip");
        assert_eq!(loaded, cfg);
    }

    #[tokio::test]
    async fn get_returns_disabled_default_when_never_set() {
        let state = test_state();
        let loaded = get_auto_optimization_config(&state.inner.saved_objects, "default__never_configured").await;
        assert!(loaded.is_none(), "no config saved yet -> None, callers default to disabled");
    }

    #[tokio::test]
    async fn saving_the_same_collection_twice_overwrites_not_duplicates() {
        let state = test_state();
        let cfg1 = AutoOptimizationConfig {
            collection: "default__mycol".to_owned(),
            enabled: false,
            target_questions: None,
            mode: TuningMode::Quality,
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg1).await;
        let cfg2 = AutoOptimizationConfig {
            enabled: true,
            ..cfg1
        };
        save_auto_optimization_config(&state.inner.saved_objects, &cfg2).await;
        let loaded = get_auto_optimization_config(&state.inner.saved_objects, "default__mycol")
            .await
            .unwrap();
        assert!(loaded.enabled, "the second save must overwrite the first, not create a duplicate");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p veda-gateway --lib rest::auto_optimization`
Expected: FAIL with "module `auto_optimization` not found" (not yet declared in `mod.rs`/`lib.rs`)

- [ ] **Step 3: Declare the module**

In `crates/veda-gateway/src/rest/mod.rs`, find the block of `pub mod` / `mod` declarations near the top of the file (search for `mod rag_eval;` or similar — the file declares one line per rest submodule). Add a new line alphabetically near the top:

```rust
pub mod auto_optimization;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p veda-gateway --lib rest::auto_optimization`
Expected: PASS (3 tests)

- [ ] **Step 5: Register the REST routes**

In `crates/veda-gateway/src/rest/mod.rs`, near the eval-set-size-recommendation route added in Task 3, add:

```rust
        .route(
            "/v1/collections/{name}/auto-optimization",
            get(auto_optimization::get_auto_optimization).put(auto_optimization::set_auto_optimization),
        )
```

- [ ] **Step 6: Write and pass an HTTP-level integration test**

In whichever integration test file Task 3 used (or a new one), add:

```rust
#[tokio::test]
async fn auto_optimization_config_defaults_to_disabled_then_can_be_enabled() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "autoopt_test",
            "shard_count": 1,
            "fields": [{ "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }]
        }))
        .send()
        .await
        .unwrap();

    let resp = c
        .get(format!("{base}/v1/collections/autoopt_test/auto-optimization"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"], json!(false), "default must be disabled");

    let resp = c
        .put(format!("{base}/v1/collections/autoopt_test/auto-optimization"))
        .json(&json!({ "enabled": true, "target_questions": 12, "mode": "Performance" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let resp = c
        .get(format!("{base}/v1/collections/autoopt_test/auto-optimization"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["enabled"], json!(true));
    assert_eq!(body["target_questions"], json!(12));
    assert_eq!(body["mode"], json!("Performance"));
}
```

Run: `cargo test -p veda-gateway --test <file> auto_optimization_config_defaults`
Expected: PASS

- [ ] **Step 7: Run clippy + fmt**

Run: `cargo clippy -p veda-gateway --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

- [ ] **Step 8: Commit**

```bash
git add crates/veda-gateway/src/rest/auto_optimization.rs crates/veda-gateway/src/rest/mod.rs crates/veda-gateway/tests/
git commit -m "feat(auto-optimization): per-collection opt-in config store + REST"
```

---

### Task 5: Auto-trigger on ingestion

**Files:**
- Modify: `crates/veda-gateway/src/rest/auto_optimization.rs` (add the trigger function)
- Modify: `crates/veda-gateway/src/rest/ingest_common.rs` (spawn it from `ingest_markdown`'s tail)

**Interfaces:**
- Consumes: `get_auto_optimization_config` (Task 4); `eval_set_size_recommendation`'s underlying logic — reuse `document_groups`/`load_source_chunks` directly rather than an HTTP round-trip (Task 2/3); `generate_eval_set`, `run_and_persist_campaign` (existing).
- Produces: `pub async fn maybe_trigger_auto_pipeline(state: AppState, tenant: crate::tenant::TenantId, scoped_collection: String)` (spawned via `tokio::spawn`, never awaited by the caller — fire-and-forget).

- [ ] **Step 1: Write the failing test**

In `crates/veda-gateway/src/rest/auto_optimization.rs`'s `#[cfg(test)] mod tests`, add:

```rust
    async fn seed_collection_with_records(state: &AppState, name: &str, n: usize) -> String {
        use crate::tenant::TenantId;
        let tenant = TenantId::default_tenant();
        let scoped = tenant::scope(&tenant, name);
        let schema = veda_core::schema::CollectionSchema::builder(scoped.clone())
            .add_field(veda_core::schema::FieldDefinition::new(
                "content",
                veda_core::types::FieldType::String,
            ))
            .shard_count(1)
            .replication_factor(1)
            .build()
            .unwrap();
        {
            let mut schemas = state.inner.schemas.write().await;
            schemas.insert(scoped.clone(), schema.clone());
        }
        let mut managers = std::collections::HashMap::new();
        let tmp = tempfile::tempdir().unwrap();
        let mut sm = veda_storage::segment_manager::SegmentManager::new(schema, 0, tmp.path(), 512 * 1024 * 1024).unwrap();
        for i in 0..n {
            let mut record = veda_core::record::Record::new();
            record.set(
                "content",
                veda_core::record::FieldValue::String(format!("chunk number {i}")),
            );
            sm.insert(record).unwrap();
        }
        managers.insert(0u32, sm);
        state.inner.segment_managers.write().await.insert(scoped.clone(), managers);
        scoped
    }

    #[tokio::test]
    async fn maybe_trigger_auto_pipeline_is_a_noop_when_disabled() {
        let state = test_state();
        let scoped = seed_collection_with_records(&state, "trig_off", 3).await;
        let tenant = crate::tenant::TenantId::default_tenant();

        // No auto-optimization config saved at all -> disabled by default.
        maybe_trigger_auto_pipeline(state.clone(), tenant, scoped.clone()).await;

        let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
        assert!(
            sets.is_empty(),
            "disabled auto-optimization must not generate any eval set"
        );
    }

    #[tokio::test]
    async fn maybe_trigger_auto_pipeline_is_a_noop_without_an_answer_gen_endpoint() {
        let state = test_state();
        let scoped = seed_collection_with_records(&state, "trig_no_llm", 3).await;
        let tenant = crate::tenant::TenantId::default_tenant();
        save_auto_optimization_config(
            &state.inner.saved_objects,
            &AutoOptimizationConfig {
                collection: scoped.clone(),
                enabled: true,
                target_questions: None,
                mode: TuningMode::Quality,
            },
        )
        .await;

        // Enabled, but no ANSWER_GEN_* configured on this test AppState -> still a no-op.
        maybe_trigger_auto_pipeline(state.clone(), tenant, scoped.clone()).await;

        let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
        assert!(
            sets.is_empty(),
            "enabled-but-no-LLM must not fail loudly nor generate anything"
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p veda-gateway --lib rest::auto_optimization::tests::maybe_trigger`
Expected: FAIL with "cannot find function `maybe_trigger_auto_pipeline`" (and possibly "cannot find function `list_sets_pub`" — check via `grep -n 'pub.*fn list_sets' crates/veda-gateway/src/rest/rag_eval.rs`; if the existing function is named differently or not `pub`, use its actual name/visibility instead of `list_sets_pub` in the test above)

- [ ] **Step 3: Implement `maybe_trigger_auto_pipeline`**

In `crates/veda-gateway/src/rest/auto_optimization.rs`, add (before the `#[cfg(test)]` module):

```rust
/// Called from `ingest_common::ingest_markdown`'s tail after every
/// successful document ingestion. Fire-and-forget: the caller spawns
/// this via `tokio::spawn` and never awaits it, so it can never block or
/// fail the ingest HTTP response. Best-effort throughout — any failure
/// (disabled, no LLM, generation produced zero cases, campaign failed)
/// is logged and this simply returns.
pub async fn maybe_trigger_auto_pipeline(
    state: AppState,
    tenant: crate::tenant::TenantId,
    scoped_collection: String,
) {
    let Some(cfg) = get_auto_optimization_config(&state.inner.saved_objects, &scoped_collection).await else {
        return;
    };
    if !cfg.enabled {
        return;
    }
    if state.inner.answer_gen_config.read().await.is_none() {
        tracing::warn!(
            collection = %scoped_collection,
            "auto-optimization enabled but ANSWER_GEN_BASE_URL not configured; skipping"
        );
        return;
    }

    let display_name = tenant::display_name(&scoped_collection).to_owned();

    // Always regenerate fresh, sized to the NOW-grown corpus (per design:
    // never incrementally append — diversity/difficulty balance has to be
    // recomputed against the whole corpus each time).
    let target_questions = match cfg.target_questions {
        Some(n) => n,
        None => {
            match super::rag_eval::load_source_chunks(&state, &scoped_collection).await {
                Ok(chunks) => {
                    let groups = super::rag_eval::document_groups(&chunks);
                    (groups.len() * 2).max(4).min(chunks.len().max(1))
                }
                Err(_) => return,
            }
        }
    };

    let req = super::rag_eval::GenerateEvalSetRequest {
        name: Some(format!("auto-eval-{display_name}")),
        num_cases: target_questions,
    };
    // A DETERMINISTIC id (not the usual fresh `eval-{uuid}`) — this is
    // what makes "re-trigger every time it grows" a genuine regeneration
    // rather than an ever-accumulating pile of orphaned auto-generated
    // sets from every past ingestion. `SavedObjectStore.put` upserts by
    // id, so saving under this same id each time overwrites the prior
    // auto-generated set for this collection.
    let preferred_id = format!("auto-eval-{display_name}");
    let generated = match super::rag_eval::generate_eval_set_for_collection(
        &state,
        &tenant,
        &display_name,
        req,
        Some(preferred_id),
    )
    .await
    {
        Ok(set) => set,
        Err((status, err)) => {
            tracing::warn!(
                collection = %scoped_collection,
                status = %status,
                error = %err.0.error,
                "auto-optimization: eval-set generation failed; skipping campaign"
            );
            return;
        }
    };
    if generated.cases.is_empty() {
        tracing::warn!(collection = %scoped_collection, "auto-optimization: generated eval set has 0 cases; skipping campaign");
        return;
    }

    let scoped_eval_set_id = tenant::scope(&tenant, &generated.id);
    let budget = crate::tuning::campaign::CampaignBudget {
        max_trials: 40,
        max_duration_secs: 3600,
    };
    let campaign = crate::rest::tuning::run_and_persist_campaign_with_mode(
        &state,
        &scoped_collection,
        &scoped_eval_set_id,
        budget,
        3,
        cfg.mode.clone(),
    )
    .await;
    match campaign {
        Ok(c) => tracing::info!(
            collection = %scoped_collection,
            campaign_id = %c.id,
            trial_count = c.trials.len(),
            "auto-optimization: campaign completed"
        ),
        Err(e) => tracing::warn!(collection = %scoped_collection, error = %e, "auto-optimization: campaign failed"),
    }
}
```

This references two functions that don't exist yet as named — `generate_eval_set_for_collection` (a thin, non-HTTP-handler wrapper around `generate_eval_set`'s body so it can be called from a background task without an axum `Path`/`Json` extraction) and `run_and_persist_campaign_with_mode` (Task 6). Do NOT implement these yet in this step — Step 4 adds `generate_eval_set_for_collection`; Task 6 adds `run_and_persist_campaign_with_mode`. This task's tests (Step 1) only exercise the disabled/no-LLM no-op paths, which return before either function is reached, so they can pass before Step 4/Task 6 land — but the crate won't COMPILE until Step 4 lands (same commit).

- [ ] **Step 4: Extract `generate_eval_set_for_collection` from `generate_eval_set`**

In `crates/veda-gateway/src/rest/rag_eval.rs`, find `pub async fn generate_eval_set` and refactor it to delegate to a new non-HTTP-handler function that takes a display name + tenant directly (so `auto_optimization.rs` can call it without constructing axum extractors):

```rust
pub async fn generate_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<GenerateEvalSetRequest>,
) -> Result<Json<EvalSet>, GatewayError> {
    // A human/UI-triggered generate call always gets a fresh random id
    // (today's existing behavior, unchanged) — only the auto-optimization
    // trigger (Task 5) asks for a deterministic, overwritable id.
    let set = generate_eval_set_for_collection(&state, &tenant, &collection, req, None).await?;
    Ok(Json(set))
}

/// The actual generation logic, callable both from the HTTP handler
/// above and from `auto_optimization::maybe_trigger_auto_pipeline`
/// (which has no axum request to extract `Path`/`Json` from).
/// `preferred_id`: when `Some`, the resulting `EvalSet.id` is this exact
/// value instead of a fresh `eval-{uuid}` — used by the auto-optimization
/// trigger so repeated regeneration overwrites the same object (via
/// `SavedObjectStore.put`'s upsert-by-id) rather than accumulating a new
/// orphaned set on every ingestion.
pub async fn generate_eval_set_for_collection(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    collection: &str,
    req: GenerateEvalSetRequest,
    preferred_id: Option<String>,
) -> Result<EvalSet, GatewayError> {
    let llm_config = state.inner.answer_gen_config.read().await.clone();
    if llm_config.is_none() {
        return Err(service_unavailable(
            "ANSWER_GEN_BASE_URL not configured. Set the env var to enable eval-set generation.",
        ));
    }
    let semaphore = state.inner.answer_gen_semaphore.clone();

    let scoped = tenant::scope(tenant, collection);
    let chunks = load_source_chunks(state, &scoped).await?;
    if chunks.is_empty() {
        return Err(bad_request(format!(
            "collection '{collection}' has no records to generate cases from"
        )));
    }

    let sampled = sample_diverse_chunks(chunks, req.num_cases);
    let mix = default_difficulty_mix();
    let cases = generate_cases(&sampled, &mix, req.num_cases, move |chunk, difficulty| {
        let llm_config = llm_config.clone();
        let semaphore = semaphore.clone();
        let text = chunk.text.clone();
        let difficulty = difficulty.to_owned();
        async move { generate_one_qa(&llm_config, &semaphore, &text, &difficulty).await }
    })
    .await;

    let set = EvalSet {
        id: preferred_id.unwrap_or_else(|| format!("eval-{}", uuid::Uuid::new_v4())),
        name: req
            .name
            .unwrap_or_else(|| format!("generated-{collection}")),
        agent_id: None,
        cases,
    };
    save_and_return_set(state, tenant, set).await.map(|Json(s)| s)
}
```

(`State(state): State<AppState>` in the original handler took `state` by value; the extracted function takes `&AppState` — `save_and_return_set`'s existing signature should be checked via `grep -n 'fn save_and_return_set' crates/veda-gateway/src/rest/rag_eval.rs` to confirm it already accepts `&AppState`/`&TenantId` — adjust the call above to match its actual signature if it differs.)

- [ ] **Step 5: Run the Task 5 tests (disabled/no-LLM paths) to verify they pass**

Run: `cargo test -p veda-gateway --lib rest::auto_optimization::tests::maybe_trigger`
Expected: PASS (2 tests) — Task 6 must land in the same overall change for the crate to compile (the enabled+configured path references `run_and_persist_campaign_with_mode`), so run this AFTER Task 6's Step 3 lands; if working strictly task-by-task, stub `run_and_persist_campaign_with_mode` as a thin wrapper calling the existing `run_and_persist_campaign` (ignoring `mode`) for now, and let Task 6 replace the stub — note in the Task 6 implementer brief that this stub already exists and needs its body upgraded, not a new function created.

- [ ] **Step 6: Spawn the trigger from `ingest_markdown`'s tail**

In `crates/veda-gateway/src/rest/ingest_common.rs`, find:

```rust
    rebuild_executors(state, collection_name)
        .await
        .map_err(|(status, json_err)| (status, json_err.error.clone()))?;

    Ok(IngestResult {
        chunks_inserted: chunks_out.len(),
        chunks: chunks_out,
    })
}
```

Replace with:

```rust
    rebuild_executors(state, collection_name)
        .await
        .map_err(|(status, json_err)| (status, json_err.error.clone()))?;

    // Fire-and-forget: never blocks or fails this ingest call. See
    // `auto_optimization::maybe_trigger_auto_pipeline`'s doc comment.
    {
        let state = state.clone();
        let collection_name = collection_name.to_owned();
        let tenant = source_meta.tenant.clone();
        tokio::spawn(async move {
            super::auto_optimization::maybe_trigger_auto_pipeline(state, tenant, collection_name).await;
        });
    }

    Ok(IngestResult {
        chunks_inserted: chunks_out.len(),
        chunks: chunks_out,
    })
}
```

This assumes `SourceMeta` gains a `tenant: crate::tenant::TenantId` field (it currently doesn't — check via `grep -n 'pub struct SourceMeta' -A 10 crates/veda-gateway/src/rest/ingest_common.rs`). Add it:

```rust
pub struct SourceMeta {
    pub source_file: Option<String>,
    pub source_url: Option<String>,
    pub title: Option<String>,
    pub expires_at: Option<i64>,
    /// The tenant this ingestion ran under — needed so the auto-
    /// optimization trigger (spawned after this function returns) can
    /// correctly scope its eval-set-generation and campaign calls.
    pub tenant: crate::tenant::TenantId,
}
```

Then find every call site that constructs a `SourceMeta { .. }` literal (both in `document.rs`'s OCR path and wherever `/ingest-url` builds one — search `grep -rn 'SourceMeta {' crates/veda-gateway/src/rest/`) and add `tenant: tenant.clone(),` to each literal (each call site already has a `tenant: TenantId` in scope from its own `Tenant(tenant)` extractor — confirm this via reading each call site before editing).

- [ ] **Step 7: Run the full ingest_common + auto_optimization test suites**

Run: `cargo test -p veda-gateway --lib rest::ingest_common:: rest::auto_optimization::`
Expected: PASS (existing ingest tests unaffected; new auto_optimization tests pass)

- [ ] **Step 8: Run clippy + fmt**

Run: `cargo clippy -p veda-gateway --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

- [ ] **Step 9: Commit**

```bash
git add crates/veda-gateway/src/rest/auto_optimization.rs crates/veda-gateway/src/rest/ingest_common.rs crates/veda-gateway/src/rest/rag_eval.rs crates/veda-gateway/src/rest/document.rs
git commit -m "feat(auto-optimization): fire-and-forget testset+campaign trigger on ingestion"
```

---

### Task 6: `TuningMode` — quality vs performance-only selection

**Files:**
- Modify: `crates/veda-gateway/src/tuning/campaign.rs` (mode-aware trial selection)
- Modify: `crates/veda-gateway/src/rest/tuning.rs` (`StartCampaignRequest.mode`, `run_and_persist_campaign_with_mode`)

**Interfaces:**
- Consumes: `TuningMode` (Task 4, `auto_optimization.rs`); `Trial`, `TrialObjectives`, `PipelineConfig::default()` (existing, `campaign.rs`/`config.rs`).
- Produces: `pub async fn run_campaign_with_mode<F, Fut>(collection, eval_set_id, budget, outer_bases, mode: TuningMode, evaluate_trial: F) -> TuningCampaign` (extends `run_campaign`'s existing signature with `mode`); `run_and_persist_campaign_with_mode` in `rest/tuning.rs` (replaces the Task 5 stub).

- [ ] **Step 1: Write the failing test for performance-mode selection**

In `crates/veda-gateway/src/tuning/campaign.rs`'s `#[cfg(test)] mod campaign_loop_tests` (or the module actually containing `run_campaign_produces_a_valid_pareto_front` — confirm the exact module name via `grep -n 'mod .*tests' crates/veda-gateway/src/tuning/campaign.rs`), add:

```rust
    #[tokio::test]
    async fn performance_mode_selects_single_fastest_trial_above_recall_floor() {
        // Three fixed "trials": one slow-but-perfect-recall (the baseline,
        // which also sets the recall floor), one fast-but-below-floor
        // (must be excluded), one fast-and-above-floor (the expected winner).
        let mut call_count = 0usize;
        let evaluate_trial = move |config: PipelineConfig| {
            call_count += 1;
            let is_default = config == PipelineConfig::default();
            async move {
                if is_default {
                    TrialObjectives { recall: 1.0, faithfulness: 0.9, latency_p50_ms: 20.0 }
                } else if config.tier1.ef_search == 32 {
                    TrialObjectives { recall: 0.4, faithfulness: 0.4, latency_p50_ms: 1.0 }
                } else {
                    TrialObjectives { recall: 0.95, faithfulness: 0.8, latency_p50_ms: 5.0 }
                }
            }
        };
        let mut fast_bad = PipelineConfig::default();
        fast_bad.tier1.ef_search = 32;
        let mut fast_good = PipelineConfig::default();
        fast_good.tier1.ef_search = 64;
        let outer_bases = vec![PipelineConfig::default(), fast_bad, fast_good];

        let campaign = run_campaign_with_mode(
            "mycol",
            "eval-1",
            CampaignBudget { max_trials: 20, max_duration_secs: 60 },
            outer_bases,
            crate::rest::auto_optimization::TuningMode::Performance,
            evaluate_trial,
        )
        .await;

        assert_eq!(
            campaign.pareto_front.len(),
            1,
            "performance mode reports exactly one winner, not a multi-trial front"
        );
        let winner = campaign
            .trials
            .iter()
            .find(|t| t.id == campaign.pareto_front[0])
            .unwrap();
        assert!(
            winner.objectives.recall >= 0.95,
            "the winner must clear the recall floor (the default config's own 1.0 recall); got {:?}",
            winner.objectives
        );
        assert_eq!(
            winner.objectives.latency_p50_ms, 5.0,
            "among floor-passing trials the winner must be the fastest"
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p veda-gateway --lib tuning::campaign::campaign_loop_tests::performance_mode`
Expected: FAIL with "cannot find function `run_campaign_with_mode`"

- [ ] **Step 3: Implement `run_campaign_with_mode` + performance selection**

In `crates/veda-gateway/src/tuning/campaign.rs`, find the existing `pub async fn run_campaign<F, Fut>(...)` — rename it to a private `run_campaign_inner` and change its final selection step to be mode-aware, then re-expose both a backward-compatible `run_campaign` (quality mode, for existing callers/tests) and the new `run_campaign_with_mode`:

```rust
pub async fn run_campaign<F, Fut>(
    collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    outer_bases: Vec<PipelineConfig>,
    evaluate_trial: F,
) -> TuningCampaign
where
    F: FnMut(PipelineConfig) -> Fut,
    Fut: std::future::Future<Output = TrialObjectives>,
{
    run_campaign_with_mode(
        collection,
        eval_set_id,
        budget,
        outer_bases,
        crate::rest::auto_optimization::TuningMode::Quality,
        evaluate_trial,
    )
    .await
}

pub async fn run_campaign_with_mode<F, Fut>(
    collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    outer_bases: Vec<PipelineConfig>,
    mode: crate::rest::auto_optimization::TuningMode,
    mut evaluate_trial: F,
) -> TuningCampaign
where
    F: FnMut(PipelineConfig) -> Fut,
    Fut: std::future::Future<Output = TrialObjectives>,
{
```

(The body below this signature is UNCHANGED from the existing `run_campaign` up through the line `let mut trials: Vec<Trial> = Vec::new();` and the entire `'outer: for base in &outer_bases { ... }` loop — copy it verbatim into `run_campaign_with_mode`.)

Then find the tail:

```rust
    let pareto_ids = mark_and_collect_global_pareto_front(&mut trials);
    let completed_at = now_ms();
```

Replace with:

```rust
    let selected_ids = match mode {
        crate::rest::auto_optimization::TuningMode::Quality => {
            mark_and_collect_global_pareto_front(&mut trials)
        }
        crate::rest::auto_optimization::TuningMode::Performance => {
            mark_and_collect_performance_winner(&mut trials)
        }
    };
    let completed_at = now_ms();
```

And change the final struct literal's `pareto_front: pareto_ids,` to `pareto_front: selected_ids,`.

Immediately after `mark_and_collect_global_pareto_front`'s existing function body, add:

```rust
/// Performance-only selection: among trials whose recall clears the
/// floor set by the DEFAULT `PipelineConfig`'s own recall (never a fixed
/// magic number — switching to this mode can never regress recall below
/// what the collection already achieves today), pick the single lowest-
/// `latency_p50_ms` trial. Returns an empty vec (no winner) if nothing
/// clears the floor — callers must handle an empty `pareto_front`
/// gracefully, exactly as they already do for a `Quality`-mode front
/// that happens to contain zero qualifying trials.
fn mark_and_collect_performance_winner(trials: &mut [Trial]) -> Vec<String> {
    for t in trials.iter_mut() {
        t.is_pareto_optimal = false;
    }
    let recall_floor = trials
        .iter()
        .find(|t| t.config == PipelineConfig::default())
        .map(|t| t.objectives.recall)
        .unwrap_or(0.0);
    let winner_idx = trials
        .iter()
        .enumerate()
        .filter(|(_, t)| t.objectives.recall >= recall_floor)
        .min_by(|(_, a), (_, b)| {
            a.objectives
                .latency_p50_ms
                .partial_cmp(&b.objectives.latency_p50_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i);
    match winner_idx {
        Some(i) => {
            trials[i].is_pareto_optimal = true;
            vec![trials[i].id.clone()]
        }
        None => Vec::new(),
    }
}
```

`PipelineConfig` needs `PartialEq` for the `==` comparison above — check its derive attribute via `grep -n '#\[derive' crates/veda-gateway/src/tuning/config.rs` right before `pub struct PipelineConfig`; add `PartialEq` if not already present (and to `Tier1Config`/`Tier2Config`/`Tier3Config` if the derive doesn't already cascade — Rust requires every field type to itself implement `PartialEq` for a struct's derive to work).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p veda-gateway --lib tuning::campaign::`
Expected: PASS (existing Pareto-front tests unaffected; new performance-mode test passes)

- [ ] **Step 5: Thread `mode` through the REST layer**

In `crates/veda-gateway/src/rest/tuning.rs`, find `StartCampaignRequest`:

```rust
#[derive(Deserialize)]
pub struct StartCampaignRequest {
    pub eval_set_id: String,
    #[serde(default = "default_max_trials")]
    pub max_trials: usize,
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    #[serde(default = "default_num_bases")]
    pub num_bases: usize,
}
```

Add a `mode` field:

```rust
#[derive(Deserialize)]
pub struct StartCampaignRequest {
    pub eval_set_id: String,
    #[serde(default = "default_max_trials")]
    pub max_trials: usize,
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    #[serde(default = "default_num_bases")]
    pub num_bases: usize,
    #[serde(default)]
    pub mode: crate::rest::auto_optimization::TuningMode,
}
```

In `start_campaign`, find the call `run_and_persist_campaign(&state, &scoped_collection, &scoped_eval_set_id, budget, req.num_bases)` and change it to pass `req.mode.clone()` through (see Step 6 below for the updated `run_and_persist_campaign_with_mode` signature — `start_campaign` should call THAT function now, not the old one).

- [ ] **Step 6: Replace the Task 5 stub with the real mode-aware wrapper**

In `crates/veda-gateway/src/rest/tuning.rs`, find `pub async fn run_and_persist_campaign` — rename its body's call from `run_campaign(...)` to `run_campaign_with_mode(..., mode, ...)`, and add a `mode` parameter, renaming the function to `run_and_persist_campaign_with_mode` (matching what Task 5 already calls). Keep a thin `run_and_persist_campaign` wrapper (no `mode` param, defaults to `Quality`) ONLY if any existing caller (check `grep -rn 'run_and_persist_campaign(' crates/veda-gateway/src/`) doesn't already pass a mode — update every call site to use `_with_mode` directly instead, since this is a small, fully-controlled call-site set (this function is called from `start_campaign`, the scheduler's `JobKind::TuningCampaign` dispatch, and now `auto_optimization.rs` — check `crates/veda-gateway/src/scheduler.rs`'s call site too and update it to pass `crate::rest::auto_optimization::TuningMode::Quality` explicitly, preserving today's behavior for scheduled campaigns).

```rust
pub async fn run_and_persist_campaign_with_mode(
    state: &AppState,
    scoped_collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    num_bases: usize,
    mode: crate::rest::auto_optimization::TuningMode,
) -> Result<TuningCampaign, String> {
    let (vector_field, text_field) = resolve_search_fields(state, scoped_collection).await?;
    let bases = sample_outer_bases(num_bases);
    let sibling_by_base = materialize_bases(state, scoped_collection, &bases).await?;
    let bases_for_lookup = bases.clone();
    let state_ref = state;
    let eval_set_id_owned = eval_set_id.to_owned();
    let sibling_by_base_ref = &sibling_by_base;
    let live_collection = scoped_collection.to_owned();
    let vector_field_owned = vector_field;
    let text_field_owned = text_field;
    let campaign = crate::tuning::campaign::run_campaign_with_mode(
        scoped_collection,
        eval_set_id,
        budget,
        bases,
        mode,
        move |config: PipelineConfig| {
            let eval_set_id = eval_set_id_owned.clone();
            let live_collection = live_collection.clone();
            let vector_field = vector_field_owned.clone();
            let text_field = text_field_owned.clone();
            let bases_for_lookup = bases_for_lookup.clone();
            async move {
                evaluate_one_trial(
                    state_ref,
                    &eval_set_id,
                    &config,
                    &live_collection,
                    &vector_field,
                    &text_field,
                    &bases_for_lookup,
                    sibling_by_base_ref,
                )
                .await
            }
        },
    )
    .await;

    save_campaign(&state.inner.saved_objects, &campaign).await;
    Ok(campaign)
}
```

Update `start_campaign`'s call site accordingly (pass `req.mode.clone()`), and `crates/veda-gateway/src/scheduler.rs`'s `JobKind::TuningCampaign` dispatch to call `run_and_persist_campaign_with_mode(..., crate::rest::auto_optimization::TuningMode::Quality)` (preserving today's scheduled-campaign behavior exactly).

Update `auto_optimization.rs`'s `maybe_trigger_auto_pipeline` (Task 5, Step 3) to call `crate::rest::tuning::run_and_persist_campaign_with_mode` (the real function name) instead of the placeholder `run_and_persist_campaign_with_mode` it already referenced — the name matches, so this should already compile once this step lands; just confirm the parameter order matches exactly (`state, scoped_collection, eval_set_id, budget, num_bases, mode` — Task 5's call site passes `mode` LAST as `cfg.mode.clone()`, matching this signature).

- [ ] **Step 7: Run the full tuning + auto_optimization test suites**

Run: `cargo test -p veda-gateway --lib tuning:: rest::tuning:: rest::auto_optimization:: rest::scheduler:: rest::ingest_common::`
Expected: PASS — this is the integration point where Tasks 4/5/6 all compile and pass together

- [ ] **Step 8: Run clippy + fmt**

Run: `cargo clippy -p veda-gateway --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: clean

- [ ] **Step 9: Run the FULL gateway test suite as a final safety net**

Run: `cargo test -p veda-gateway`
Expected: 0 failed (this task touches the most call sites of any task in this plan — `run_campaign`/`run_and_persist_campaign` had 3+ callers before this change)

- [ ] **Step 10: Commit**

```bash
git add crates/veda-gateway/src/tuning/campaign.rs crates/veda-gateway/src/tuning/config.rs crates/veda-gateway/src/rest/tuning.rs crates/veda-gateway/src/rest/auto_optimization.rs crates/veda-gateway/src/scheduler.rs
git commit -m "feat(tuning): dual mode — Quality (existing Pareto front) vs Performance (fastest above a recall floor)"
```

---

### Task 7: Niriksha — "Generate from documents" UI + size recommendation

**Files:**
- Modify: `niriksha/src/api/ragEval.ts` (add `generateEvalSet`, `getEvalSetSizeRecommendation`, `getAutoOptimizationConfig`, `setAutoOptimizationConfig`)
- Modify: `niriksha/src/pages/quality/EvalHarness.tsx` (add a "Generate from documents" button + modal alongside the existing manual `CreateSetModal`)

**Interfaces:**
- Consumes: `GET /v1/collections/{name}/eval-set-size-recommendation`, `POST /v1/collections/{name}/eval-sets/generate`, `GET/PUT /v1/collections/{name}/auto-optimization` (Tasks 3/4).
- Produces: `generateEvalSet(collection: string, req: {name?: string, num_cases: number}): Promise<AxiosResponse<EvalSet>>`; `getEvalSetSizeRecommendation(collection: string): Promise<AxiosResponse<{recommended: number, max: number, chunk_count: number, document_count: number}>>`.

- [ ] **Step 1: Add the API client functions**

In `niriksha/src/api/ragEval.ts`, find the existing block of exported functions (`listEvalSets`, `createEvalSet`, etc. — near the bottom of the file). Add:

```typescript
export interface EvalSetSizeRecommendation {
  recommended: number
  max: number
  chunk_count: number
  document_count: number
}

export interface GenerateEvalSetRequest {
  name?: string
  num_cases: number
}

export const getEvalSetSizeRecommendation = (collection: string) =>
  api.get<EvalSetSizeRecommendation>(
    `/collections/${encodeURIComponent(collection)}/eval-set-size-recommendation`,
  )

export const generateEvalSet = (collection: string, req: GenerateEvalSetRequest) =>
  api.post<EvalSet>(`/collections/${encodeURIComponent(collection)}/eval-sets/generate`, req)

export type TuningMode = 'Quality' | 'Performance'

export interface AutoOptimizationConfig {
  collection: string
  enabled: boolean
  target_questions?: number | null
  mode: TuningMode
}

export const getAutoOptimizationConfig = (collection: string) =>
  api.get<AutoOptimizationConfig>(`/collections/${encodeURIComponent(collection)}/auto-optimization`)

export const setAutoOptimizationConfig = (
  collection: string,
  req: { enabled: boolean; target_questions?: number | null; mode: TuningMode },
) => api.put<AutoOptimizationConfig>(`/collections/${encodeURIComponent(collection)}/auto-optimization`, req)
```

- [ ] **Step 2: Type-check**

Run: `cd niriksha && npx tsc --noEmit -p tsconfig.app.json`
Expected: PASS, exit 0

- [ ] **Step 3: Add a "Generate from documents" entry point + modal**

In `niriksha/src/pages/quality/EvalHarness.tsx`, find the existing `<Button ... onClick={() => setCreating(true)}>` (the "+ New" style button that opens `CreateSetModal`) near line 126-133. Add a sibling button immediately after it:

```tsx
          <Button
            variant="secondary"
            size="sm"
            leading={<Sparkles size={14} />}
            onClick={() => setGenerating(true)}
          >
            Generate from documents
          </Button>
```

(Check the exact import list at the top of the file for an existing icon import to reuse instead of `Sparkles` if one already fits better — e.g. if `Wand2`/`FlaskConical` is already imported for a similar purpose elsewhere in this file; only add a new lucide-react import if none of the already-imported icons fit.)

Add `const [generating, setGenerating] = useState(false)` alongside the existing `const [creating, setCreating] = useState(false)` state declaration (line 50).

Near the existing `<CreateSetModal ... />` render (search for where `creating` is used to conditionally render it), add:

```tsx
      {generating && (
        <GenerateSetModal
          onClose={() => setGenerating(false)}
          onCreated={(id) => {
            setActiveId(id)
            setGenerating(false)
            load(true)
          }}
        />
      )}
```

At the bottom of the file (after the existing `CreateSetModal` function definition), add:

```tsx
function GenerateSetModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [collection, setCollection] = useState('')
  const [numCases, setNumCases] = useState(20)
  const [recommendation, setRecommendation] = useState<EvalSetSizeRecommendation | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadRecommendation = async (name: string) => {
    if (!name.trim()) {
      setRecommendation(null)
      return
    }
    try {
      const { data } = await getEvalSetSizeRecommendation(name.trim())
      setRecommendation(data)
      setNumCases(data.recommended)
    } catch {
      setRecommendation(null)
    }
  }

  const submit = async () => {
    if (!collection.trim()) {
      setError('Enter a collection name.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const { data } = await generateEvalSet(collection.trim(), { num_cases: numCases })
      onCreated(data.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed. Is ANSWER_GEN_* configured?')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal open onClose={onClose}>
      <div className="p-4">
        <h2 className="mb-3 text-sm font-semibold">Generate an eval set from documents</h2>
        <label className="mb-1 block text-xs font-medium">Collection</label>
        <input
          className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
          value={collection}
          onChange={(e) => {
            setCollection(e.target.value)
            loadRecommendation(e.target.value)
          }}
          placeholder="e.g. rag_test"
        />
        <label className="mb-1 block text-xs font-medium">
          Number of questions
          {recommendation && (
            <span className="ml-1.5 font-normal text-[var(--color-text-muted)]">
              ({recommendation.recommended} recommended for this index's size —{' '}
              {recommendation.document_count} documents, {recommendation.chunk_count} chunks)
            </span>
          )}
        </label>
        <input
          type="number"
          className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
          value={numCases}
          min={1}
          max={recommendation?.max}
          onChange={(e) => {
            const n = Number(e.target.value)
            const capped = recommendation ? Math.min(n, recommendation.max) : n
            setNumCases(Math.max(1, capped))
          }}
        />
        {error && <div className="mb-3 text-xs text-red-500">{error}</div>}
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={submit} disabled={saving}>
            {saving ? 'Generating…' : 'Generate'}
          </Button>
        </div>
      </div>
    </Modal>
  )
}
```

Add the necessary imports at the top of `EvalHarness.tsx`:

```typescript
import {
  generateEvalSet,
  getEvalSetSizeRecommendation,
  type EvalSetSizeRecommendation,
} from '../../api/ragEval'
```

(Merge this into the existing `from '../../api/ragEval'` import line if one already exists in this file, rather than adding a duplicate import statement.)

- [ ] **Step 4: Type-check and build**

Run: `cd niriksha && npx tsc --noEmit -p tsconfig.app.json && npm run build`
Expected: both PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add niriksha/src/api/ragEval.ts niriksha/src/pages/quality/EvalHarness.tsx
git commit -m "feat(niriksha): generate-from-documents eval-set flow with a corpus-aware size recommendation"
```

---

### Task 8: Niriksha — auto-optimization toggle + tuning-mode selector

**Files:**
- Modify: `niriksha/src/pages/quality/campaigns-api.ts` (add `mode` to `StartCampaignRequest`)
- Modify: `niriksha/src/pages/quality/Campaigns.tsx` (mode selector on the start-campaign form)
- Modify: `niriksha/src/pages/quality/EvalHarness.tsx` (a small auto-optimization toggle on the `GenerateSetModal` from Task 7, so enabling it is discoverable right where a user just generated a set for a collection)

**Interfaces:**
- Consumes: `getAutoOptimizationConfig`/`setAutoOptimizationConfig` (Task 7's additions to `ragEval.ts`).

- [ ] **Step 1: Add `mode` to the campaigns API's request type**

In `niriksha/src/pages/quality/campaigns-api.ts`, find:

```typescript
export interface StartCampaignRequest {
  eval_set_id: string
  max_trials?: number
  max_duration_secs?: number
  num_bases?: number
}
```

Add:

```typescript
export interface StartCampaignRequest {
  eval_set_id: string
  max_trials?: number
  max_duration_secs?: number
  num_bases?: number
  mode?: 'Quality' | 'Performance'
}
```

- [ ] **Step 2: Add a mode selector to `StartCampaignForm`**

In `niriksha/src/pages/quality/Campaigns.tsx`, find `StartCampaignForm`'s body — specifically the `<label className="mb-2 block text-xs font-medium">Max trials</label>` block. Add, immediately before it:

```tsx
      <label className="mb-2 block text-xs font-medium">Optimization mode</label>
      <select
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={mode}
        onChange={(e) => setMode(e.target.value as 'Quality' | 'Performance')}
      >
        <option value="Quality">Quality (recall + faithfulness + latency, full Pareto front)</option>
        <option value="Performance">Performance (fastest, without dropping below today's recall)</option>
      </select>
```

Add `const [mode, setMode] = useState<'Quality' | 'Performance'>('Quality')` alongside the existing `const [evalSetId, setEvalSetId] = useState(evalSetIds[0] ?? '')` line in `StartCampaignForm`.

Update the `onSubmit` call at the bottom of the form:

```tsx
          onClick={() => onSubmit({ eval_set_id: evalSetId, max_trials: maxTrials, mode })}
```

- [ ] **Step 3: Type-check and build**

Run: `cd niriksha && npx tsc --noEmit -p tsconfig.app.json && npm run build`
Expected: both PASS

- [ ] **Step 4: Add an auto-optimization toggle to `GenerateSetModal`**

In `niriksha/src/pages/quality/EvalHarness.tsx`'s `GenerateSetModal` (from Task 7), add state and a toggle so a user can opt a collection into the hands-off pipeline right after generating its first testset:

```tsx
  const [autoOptimize, setAutoOptimize] = useState(false)
```

(add alongside the modal's other `useState` declarations), and in `submit`, after the successful `generateEvalSet` call, add:

```typescript
      if (autoOptimize) {
        await setAutoOptimizationConfig(collection.trim(), { enabled: true, mode: 'Quality' })
      }
```

(insert this line right before `onCreated(data.id)` inside the existing `try` block). Add the checkbox to the modal's JSX, right before the Cancel/Generate button row:

```tsx
        <label className="mb-3 flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={autoOptimize}
            onChange={(e) => setAutoOptimize(e.target.checked)}
          />
          Auto-regenerate this testset and re-tune every time this collection grows
        </label>
```

Add `setAutoOptimizationConfig` to the existing `ragEval` import in this file.

- [ ] **Step 5: Type-check and build**

Run: `cd niriksha && npx tsc --noEmit -p tsconfig.app.json && npm run build`
Expected: both PASS

- [ ] **Step 6: Commit**

```bash
git add niriksha/src/pages/quality/campaigns-api.ts niriksha/src/pages/quality/Campaigns.tsx niriksha/src/pages/quality/EvalHarness.tsx
git commit -m "feat(niriksha): Quality/Performance mode selector + a discoverable auto-optimize opt-in"
```

---

### Task 9: Real-data end-to-end verification

**Files:** none (verification only — run on gpu-host-c's live server against the real corpus)

- [ ] **Step 1: Relay every changed file to gpu-host-c**

From the Windows mirror, `scp` every file touched in Tasks 1-8 to the corresponding path under `/home/mac/25gitlab/project-mdb/` on gpu-host-c (mirroring this whole session's established relay pattern).

- [ ] **Step 2: Rebuild and restart veda-server on gpu-host-c with the real `ANSWER_GEN_*` credentials**

```bash
ssh gpu-host-c "cd /home/mac/25gitlab/project-mdb && /home/mac/.cargo/bin/cargo build --release -p veda-server"
```

Stop the current server, relaunch with the SAME env as before PLUS:

```
ANSWER_GEN_BASE_URL=http://llm-host.internal.example:9000/v1
ANSWER_GEN_API_KEY=tensorflow
ANSWER_GEN_MODEL=Qwen/Qwen3.5-122B-A10B
```

Verify: `curl --noproxy '*' http://localhost:6340/v1/admin/health` → `200`.

- [ ] **Step 3: Generate a diverse testset from the real corpus and inspect it**

```bash
curl --noproxy '*' http://localhost:6340/v1/collections/rag_test/eval-set-size-recommendation
```

Confirm `document_count` is close to 18 (the real ingested PPTX-derived PDFs) and `chunk_count` close to 184; `recommended` should be `document_count * 2` (capped at `chunk_count`).

```bash
curl --noproxy '*' -X POST http://localhost:6340/v1/collections/rag_test/eval-sets/generate \
  -H 'Content-Type: application/json' -d '{"num_cases": 20}'
```

Fetch the resulting set and manually inspect (via `python3 -m json.tool` or similar): confirm (a) `source.source_chunk_ids` values span at least 5 distinct real documents (cross-reference against the corpus's actual `parent_doc_id` values, e.g. via `GET /v1/collections/rag_test/records?limit=1000` and checking which `parent_doc_id`s the generated cases' chunk ids belong to); (b) `difficulty` values are within 1 of an equal three-way split across the 20 cases; (c) at least one generated question's source chunk contains `![` (image) or a `|...|` table pattern, if any such chunks exist in this corpus (spot-check via grepping the real records for these patterns first to know whether this corpus even has any — the PPTX-derived PDFs are known from this session's earlier OCR ingestion to contain `![](images/...)` references, so at least the image case should be checkable).

- [ ] **Step 4: Verify the semaphore doesn't wedge under load**

Fire the same generation request 3 times concurrently (`&` in bash, or `xargs -P3`) against `rag_test` (or a smaller test collection to keep it fast) and confirm all 3 complete successfully within a reasonable time (no deadlock, no more than 5 concurrent requests ever observed against the answer-gen endpoint — this can be eyeballed via the endpoint's own request log if accessible, or simply by confirming no timeout/error occurs and total wall-clock roughly matches `ceil(15/5) × per-request-time` rather than `15 × per-request-time` or hanging indefinitely).

- [ ] **Step 5: Verify the auto-trigger end-to-end on real data**

```bash
curl --noproxy '*' -X PUT http://localhost:6340/v1/collections/rag_test/auto-optimization \
  -H 'Content-Type: application/json' -d '{"enabled": true, "mode": "Quality"}'
```

Ingest ONE more real document from `/home/mac/25gitlab/rag_anything/bulk_module/tmp` (any PDF not yet ingested, or re-ingest one of the 18 with a distinct `source_file` name so it's treated as new) via `POST /v1/collections/rag_test/process-document`. Confirm the HTTP response returns promptly (not blocked waiting on the background pipeline). Poll `GET /v1/collections/rag_test/tuning-campaigns` a few seconds later and confirm a NEW campaign appears with a `default__auto-eval-rag_test`-derived eval_set_id and real (non-fallback) `recall`/`latency_p50_ms` values in its trials — proving the fire-and-forget trigger genuinely ran generation + a campaign without any manual step.

- [ ] **Step 6: Verify Performance mode against the real corpus**

```bash
curl --noproxy '*' -X POST http://localhost:6340/v1/collections/rag_test/tuning-campaigns \
  -H 'Content-Type: application/json' \
  -d '{"eval_set_id":"<a real eval set id from step 3>","max_trials":20,"num_bases":5,"mode":"Performance"}'
```

Confirm the response's `pareto_front` contains exactly ONE trial id, and that trial's `recall` is `>=` whatever recall the DEFAULT `PipelineConfig` achieves on this same eval set (cross-check by also running the same campaign with `"mode":"Quality"` and comparing the Quality-mode trial matching `tier1: {ef_search:96, fusion_mode:"Rrf", dense_weight:0.5, rerank:false, top_k:10}, tier2: {index_type:"Hnsw", ...}, tier3: {chunker_strategy:"MarkdownRecursive", ...}` — the default config — against the Performance winner's recall).

- [ ] **Step 7: Verify the Niriksha UI live via Playwright**

Navigate to `http://gpu-host-c.internal.example:5175/niriksha/eval-harness`, click "Generate from documents", type `rag_test` into the collection field, confirm the recommendation caption appears (~"36 recommended" or similar, matching Step 2's real numbers) and the number input is capped at the real chunk count, and confirm the "auto-regenerate" checkbox is present. Then navigate to Quality Lab → Campaigns for `rag_test` and confirm the mode selector (Quality/Performance) appears on the "+ New campaign" form and a Performance-mode campaign started from the UI completes and shows a single-trial front in `CampaignDetail`.

- [ ] **Step 8: Revert any live-state side effects from this verification pass**

If Step 5's auto-trigger produced an unwanted extra sibling/alias change on the LIVE `rag_test` alias (unlikely for a Quality-mode auto-campaign since campaigns don't apply trials automatically — only `apply_trial` swaps aliases — but confirm this explicitly: `GET /v1/collections/rag_test/records?limit=1` should still return the same 184+1 record count and the alias should be untouched), and set `enabled: false` back on `rag_test`'s auto-optimization config once verification is complete, so the live corpus doesn't keep re-triggering expensive campaigns on every future test ingestion in this shared environment.
