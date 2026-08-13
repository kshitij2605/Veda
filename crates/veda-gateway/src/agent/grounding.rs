//! Answer citations + faithfulness / answer-relevancy scoring for agent
//! runs — the foundation of Veda's RAG-quality signals.
//!
//! Two pure-ish pieces, both driven from the executor's step trace:
//!
//!   * [`extract_citations`] walks an [`AgentStep`] trace and pulls the
//!     retrieved chunks out of every `search`-shaped `ToolResult` (the
//!     `{ "hits": [{ "id", "distance", "fields" }] }` shape produced by the
//!     built-in `search` tool). It resolves each hit's `collection` from
//!     the paired `ToolCall` args and a short `snippet` from the first
//!     stringy `fields` entry, deduping by `chunk_id`.
//!
//!   * [`score_grounding`] rates how well the agent's answer is supported
//!     by those retrieved chunks. Primary path is an LLM judge (when a
//!     [`ChatModel`] is configured); the embedding-cosine fallback runs
//!     when no LLM is available or the judge's output can't be parsed. With
//!     no contexts at all it returns `no_context` and does not penalize.
//!
//! Both run **post-run** in `run_agent_turn`, off the executor hot path and
//! best-effort (a scoring failure never fails the turn).

use serde::{Deserialize, Serialize};

use veda_core::embedding_client::EmbeddingClient;
use veda_core::llm_client::{ChatMessage, ChatModel, CompletionOptions};

use super::model::AgentStep;

/// Maximum characters kept for a citation snippet.
const SNIPPET_CAP: usize = 240;

/// One retrieved chunk surfaced as a citation for an agent answer.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Citation {
    /// The retrieved record/chunk id (the search hit's `id`).
    pub chunk_id: String,
    /// Collection the chunk came from, resolved from the paired tool call's
    /// `collection` arg when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// A short text snippet from the hit's fields (first stringy field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Retrieval score (the hit's `distance`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// Faithfulness + answer-relevancy scores for one answer, 0.0–1.0.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Grounding {
    /// How well the answer's claims are supported by the retrieved
    /// contexts. `None` when not scorable (no contexts, or both paths
    /// failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faithfulness: Option<f64>,
    /// How relevant the answer is to the user's query. `None` when not
    /// scorable (the embedding fallback can only fill this if a query
    /// embedding is available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer_relevancy: Option<f64>,
    /// Which path produced the scores: `"llm_judge"`, `"embedding"`, or
    /// `"no_context"`.
    pub method: String,
}

impl Grounding {
    /// The neutral result for an answer that retrieved nothing: don't
    /// penalize a non-retrieval answer.
    fn no_context() -> Self {
        Self {
            faithfulness: None,
            answer_relevancy: None,
            method: "no_context".to_owned(),
        }
    }
}

/// Walk an [`AgentStep`] trace and extract one [`Citation`] per distinct
/// retrieved chunk. Search results are recognized by the built-in tool's
/// shape: a `ToolResult` whose `result` carries a `hits` array of
/// `{ id, distance, fields }`. The owning collection is resolved from the
/// paired `ToolCall` (matched by `tool_call_id`) `args.collection`.
///
/// Returns an empty vector when no retrieval happened. Deduped by
/// `chunk_id`, preserving first-seen order.
pub fn extract_citations(steps: &[AgentStep]) -> Vec<Citation> {
    // Map tool_call_id -> collection arg, for the paired ToolCall lookup.
    let mut collections: std::collections::HashMap<&str, String> = std::collections::HashMap::new();
    for step in steps {
        if let AgentStep::ToolCall {
            tool_call_id, args, ..
        } = step
            && let Some(c) = args.get("collection").and_then(|v| v.as_str())
        {
            collections.insert(tool_call_id.as_str(), c.to_owned());
        }
    }

    let mut out: Vec<Citation> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for step in steps {
        let AgentStep::ToolResult {
            tool_call_id,
            result,
            is_error,
            ..
        } = step
        else {
            continue;
        };
        if *is_error {
            continue;
        }
        let Some(hits) = result.get("hits").and_then(|h| h.as_array()) else {
            continue;
        };
        let collection = collections.get(tool_call_id.as_str()).cloned();
        for hit in hits {
            let Some(chunk_id) = hit.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            if !seen.insert(chunk_id.to_owned()) {
                continue;
            }
            let snippet = hit
                .get("fields")
                .and_then(|f| f.as_object())
                .and_then(first_stringy_field)
                .map(|s| truncate(&s, SNIPPET_CAP));
            let score = hit.get("distance").and_then(|v| v.as_f64());
            out.push(Citation {
                chunk_id: chunk_id.to_owned(),
                collection: collection.clone(),
                snippet,
                score,
            });
        }
    }
    out
}

/// First non-empty string value in a fields map (object iteration order),
/// skipping Veda's `_`-prefixed reserved fields.
///
/// The skip is load-bearing, not cosmetic. `fields` is a `serde_json::Map`
/// backed by a `BTreeMap`, so keys sort lexicographically and every reserved
/// key (`_guardrail_category`, `_guardrail_flagged`, …) sorts before any
/// ordinary lowercase field name. `_guardrail_category` is a *string*, so on
/// a PII-flagged hit — which the 2026-08-13 category policy deliberately
/// KEEPS — the "document text" picked here was the literal `"pii"`. That
/// became the citation snippet shown to the user and, worse, the context
/// string fed to the faithfulness judge, scoring the answer against the word
/// "pii" instead of against the retrieved passage.
fn first_stringy_field(map: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    map.iter()
        .filter(|(key, _)| !key.starts_with('_'))
        .find_map(|(_, v)| {
            v.as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
        })
}

/// Truncate `s` to at most `cap` characters (char-safe), appending an
/// ellipsis when truncated.
fn truncate(s: &str, cap: usize) -> String {
    if s.chars().count() <= cap {
        return s.to_owned();
    }
    let truncated: String = s.chars().take(cap).collect();
    format!("{truncated}…")
}

/// Cosine similarity of two equal-length vectors; 0.0 on length mismatch
/// or a zero-norm input.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Score how well `answer` is grounded in `contexts` (the citation
/// snippets) and how relevant it is to `query`.
///
/// Path selection:
///   * **No contexts** → [`Grounding::no_context`] (`None`/`None`).
///   * **LLM judge** (when `model` is `Some`) → a focused prompt asks the
///     model for two 0–1 scores; parse robustly, clamp, and on parse
///     failure fall through to the embedding path.
///   * **Embedding fallback** (no `model`, or judge unparseable; when
///     `embedder` is `Some`) → faithfulness ≈ mean over contexts of
///     cosine(answer, context); answer_relevancy ≈ cosine(answer, query).
///   * Neither path available → `None`/`None`, `method = "embedding"`
///     (nothing to score with) — still best-effort, never an error.
pub async fn score_grounding<M: ChatModel>(
    model: Option<&M>,
    embedder: Option<&EmbeddingClient>,
    query: &str,
    answer: &str,
    contexts: &[String],
) -> Grounding {
    if contexts.is_empty() {
        return Grounding::no_context();
    }

    // Primary: LLM judge.
    if let Some(model) = model
        && let Some(g) = judge_with_llm(model, query, answer, contexts).await
    {
        return g;
    }
    // Judge failed/unparseable → fall through to embedding fallback.

    // Fallback: embedding cosine.
    embedding_fallback(embedder, query, answer, contexts).await
}

/// The LLM-judge prompt. Asks for two 0–1 scores on their own lines in a
/// fixed `KEY: value` format so the response parses even amid prose.
fn build_judge_prompt(query: &str, answer: &str, contexts: &[String]) -> String {
    let mut ctx_block = String::new();
    for (i, c) in contexts.iter().enumerate() {
        ctx_block.push_str(&format!("[{}] {}\n", i + 1, c.trim()));
    }
    format!(
        "You are a strict RAG evaluation judge. Given a user QUESTION, an ANSWER, and the \
retrieved CONTEXTS the answer should be grounded in, rate two things on a 0.0–1.0 scale:\n\
- FAITHFULNESS: are the ANSWER's factual claims supported by the CONTEXTS? 1.0 = every \
claim is directly supported; 0.0 = the answer contradicts or is unsupported by the contexts \
(hallucination).\n\
- ANSWER_RELEVANCY: how well does the ANSWER address the QUESTION? 1.0 = fully on-topic and \
complete; 0.0 = off-topic.\n\n\
Respond with EXACTLY two lines and nothing else:\n\
FAITHFULNESS: <number 0.0-1.0>\n\
ANSWER_RELEVANCY: <number 0.0-1.0>\n\n\
QUESTION:\n{query}\n\n\
ANSWER:\n{answer}\n\n\
CONTEXTS:\n{ctx_block}"
    )
}

/// Run the judge and parse its two scores. Returns `None` when the model
/// errors, returns no content, or neither score can be parsed (so the
/// caller falls back to the embedding path).
async fn judge_with_llm<M: ChatModel>(
    model: &M,
    query: &str,
    answer: &str,
    contexts: &[String],
) -> Option<Grounding> {
    let prompt = build_judge_prompt(query, answer, contexts);
    let messages = vec![ChatMessage::text("user", prompt)];
    let opts = CompletionOptions {
        temperature: Some(0.0),
        max_tokens: Some(64),
        json_mode: false,
    };
    // Reuse the model's existing per-call timeout (bounded by the LLM
    // client). No tools — we just want the text scores back.
    let completion = model
        .complete_with_tools(&messages, &[], None, &opts)
        .await
        .ok()?;
    let text = completion.content?;
    let faithfulness = parse_labeled_score(&text, "FAITHFULNESS");
    let answer_relevancy = parse_labeled_score(&text, "ANSWER_RELEVANCY");
    // Require at least one parseable score, else fall back.
    if faithfulness.is_none() && answer_relevancy.is_none() {
        return None;
    }
    Some(Grounding {
        faithfulness,
        answer_relevancy,
        method: "llm_judge".to_owned(),
    })
}

/// Parse a `LABEL: <number>` score from prose, tolerant of surrounding
/// text. Returns the first number after the label, clamped to 0.0–1.0.
fn parse_labeled_score(text: &str, label: &str) -> Option<f64> {
    let upper = text.to_ascii_uppercase();
    let idx = upper.find(label)?;
    // Scan the tail after the label for the first numeric token.
    let tail = &text[idx + label.len()..];
    first_number(tail).map(|n| n.clamp(0.0, 1.0))
}

/// Extract the first floating-point number from `s` (e.g. `: 0.85` → 0.85).
fn first_number(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit()
            || (c == '.' && i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            let mut seen_dot = c == '.';
            i += 1;
            while i < bytes.len() {
                let d = bytes[i] as char;
                if d.is_ascii_digit() {
                    i += 1;
                } else if d == '.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            return s[start..i].parse::<f64>().ok();
        }
        i += 1;
    }
    None
}

/// Embedding fallback: faithfulness ≈ mean cosine(answer, context_i);
/// answer_relevancy ≈ cosine(answer, query). `None`/`None` (method
/// `"embedding"`) when no embedder is available or embedding fails.
async fn embedding_fallback(
    embedder: Option<&EmbeddingClient>,
    query: &str,
    answer: &str,
    contexts: &[String],
) -> Grounding {
    let none = || Grounding {
        faithfulness: None,
        answer_relevancy: None,
        method: "embedding".to_owned(),
    };
    let Some(embedder) = embedder else {
        return none();
    };
    let Ok(answer_vec) = embedder.embed_text(answer).await else {
        return none();
    };

    // Faithfulness: mean cosine over contexts.
    let mut sims: Vec<f64> = Vec::with_capacity(contexts.len());
    for c in contexts {
        if let Ok(cv) = embedder.embed_text(c).await {
            sims.push(cosine(&answer_vec, &cv) as f64);
        }
    }
    let faithfulness = if sims.is_empty() {
        None
    } else {
        Some((sims.iter().sum::<f64>() / sims.len() as f64).clamp(0.0, 1.0))
    };

    // Answer relevancy: cosine(answer, query) when a query is available.
    let answer_relevancy = if query.trim().is_empty() {
        None
    } else {
        match embedder.embed_text(query).await {
            Ok(qv) => Some((cosine(&answer_vec, &qv) as f64).clamp(0.0, 1.0)),
            Err(_) => None,
        }
    };

    Grounding {
        faithfulness,
        answer_relevancy,
        method: "embedding".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use veda_core::embedding::{EmbeddingModelConfig, Modality};
    use veda_core::error::Result as LlmResult;
    use veda_core::llm_client::{ToolCompletion, ToolDef};

    // ── extract_citations ────────────────────────────────────────────────

    fn search_result(hits: serde_json::Value) -> serde_json::Value {
        let arr = hits.as_array().cloned().unwrap_or_default();
        serde_json::json!({ "count": arr.len(), "hits": arr })
    }

    #[test]
    fn extract_citations_pulls_and_dedups() {
        let steps = vec![
            AgentStep::ToolCall {
                tool_call_id: "c1".to_owned(),
                tool_id: "search".to_owned(),
                args: serde_json::json!({ "collection": "docs", "query": "hnsw" }),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: search_result(serde_json::json!([
                    { "id": "id-1", "distance": 0.12, "fields": { "title": "HNSW graphs" } },
                    { "id": "id-2", "distance": 0.34, "fields": { "body": "vectors" } },
                    // Duplicate id-1 → deduped.
                    { "id": "id-1", "distance": 0.99, "fields": { "title": "again" } },
                ])),
                is_error: false,
                duration_ms: Some(2.0),
            },
        ];
        let cites = extract_citations(&steps);
        assert_eq!(cites.len(), 2, "deduped to two distinct chunk ids");
        assert_eq!(cites[0].chunk_id, "id-1");
        assert_eq!(cites[0].collection.as_deref(), Some("docs"));
        assert_eq!(cites[0].snippet.as_deref(), Some("HNSW graphs"));
        assert_eq!(cites[0].score, Some(0.12));
        assert_eq!(cites[1].chunk_id, "id-2");
        assert_eq!(cites[1].snippet.as_deref(), Some("vectors"));
    }

    /// A PII-flagged hit is deliberately KEPT (2026-08-13 category policy),
    /// and it arrives carrying the reserved verdict fields. Those sort first
    /// in the `BTreeMap` behind `fields`, so before this fix the snippet —
    /// and the faithfulness judge's context — was the literal string
    /// `"pii"` rather than the passage.
    #[test]
    fn extract_citations_skip_reserved_fields_for_the_snippet() {
        let steps = vec![
            AgentStep::ToolCall {
                tool_call_id: "c1".to_owned(),
                tool_id: "search".to_owned(),
                args: serde_json::json!({ "collection": "docs", "query": "contact" }),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: search_result(serde_json::json!([{
                    "id": "id-pii",
                    "distance": 0.2,
                    "fields": {
                        "_guardrail_category": "pii",
                        "_guardrail_flagged": true,
                        "_guardrail_score": 1.0,
                        "_indexed_at": 1_786_000_000i64,
                        "content": "route billing questions to eve@example.com",
                    }
                }])),
                is_error: false,
                duration_ms: None,
            },
        ];
        let cites = extract_citations(&steps);
        assert_eq!(cites.len(), 1);
        assert_eq!(
            cites[0].snippet.as_deref(),
            Some("route billing questions to eve@example.com"),
            "the snippet must be the document text, not a reserved field value"
        );
    }

    #[test]
    fn extract_citations_empty_without_search() {
        let steps = vec![
            AgentStep::Reasoning {
                text: "no retrieval here".to_owned(),
            },
            AgentStep::ToolCall {
                tool_call_id: "c1".to_owned(),
                tool_id: "list_collections".to_owned(),
                args: serde_json::json!({}),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: serde_json::json!({ "collections": ["docs"], "count": 1 }),
                is_error: false,
                duration_ms: None,
            },
        ];
        assert!(extract_citations(&steps).is_empty());
    }

    #[test]
    fn extract_citations_skips_error_results() {
        let steps = vec![
            AgentStep::ToolCall {
                tool_call_id: "c1".to_owned(),
                tool_id: "search".to_owned(),
                args: serde_json::json!({ "collection": "docs" }),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: serde_json::json!({ "error": "boom" }),
                is_error: true,
                duration_ms: None,
            },
        ];
        assert!(extract_citations(&steps).is_empty());
    }

    // ── parse helpers ─────────────────────────────────────────────────────

    #[test]
    fn parse_labeled_score_tolerates_prose_and_clamps() {
        let text = "Here are my scores.\nFAITHFULNESS: 0.85\nANSWER_RELEVANCY: 1.4 (very on topic)";
        assert_eq!(parse_labeled_score(text, "FAITHFULNESS"), Some(0.85));
        // 1.4 clamps to 1.0.
        assert_eq!(parse_labeled_score(text, "ANSWER_RELEVANCY"), Some(1.0));
        assert_eq!(parse_labeled_score(text, "MISSING"), None);
    }

    // ── LLM-judge path with a fake ChatModel ──────────────────────────────

    /// A scripted, network-free [`ChatModel`] returning queued content.
    struct FakeJudge {
        replies: RefCell<std::collections::VecDeque<LlmResult<ToolCompletion>>>,
    }
    impl FakeJudge {
        fn content(text: &str) -> Self {
            Self {
                replies: RefCell::new(
                    [Ok(ToolCompletion {
                        content: Some(text.to_owned()),
                        tool_calls: Vec::new(),
                        finish_reason: Some("stop".to_owned()),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    })]
                    .into_iter()
                    .collect(),
                ),
            }
        }
    }
    impl ChatModel for FakeJudge {
        async fn complete_with_tools(
            &self,
            _m: &[ChatMessage],
            _t: &[ToolDef],
            _tc: Option<&str>,
            _o: &CompletionOptions,
        ) -> LlmResult<ToolCompletion> {
            self.replies.borrow_mut().pop_front().unwrap_or(Err(
                veda_core::error::VedaError::Query("drained".to_owned()),
            ))
        }
    }

    fn mock_embedder() -> EmbeddingClient {
        EmbeddingClient::new(EmbeddingModelConfig {
            name: "mock".to_owned(),
            endpoint: "http://mock:8000".to_owned(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Text],
            output_dimension: 64,
            max_input_size_bytes: 1024 * 1024,
            timeout_seconds: 5,
            batch_size: 1,
        })
    }

    #[tokio::test]
    async fn llm_judge_grounded_answer_high_faithfulness() {
        let judge = FakeJudge::content("FAITHFULNESS: 0.95\nANSWER_RELEVANCY: 0.9");
        let g = score_grounding(
            Some(&judge),
            None,
            "what indexing does veda use?",
            "Veda uses HNSW graphs for ANN search.",
            &["Veda uses HNSW graphs for approximate nearest-neighbor search.".to_owned()],
        )
        .await;
        assert_eq!(g.method, "llm_judge");
        assert_eq!(g.faithfulness, Some(0.95));
        assert_eq!(g.answer_relevancy, Some(0.9));
    }

    #[tokio::test]
    async fn llm_judge_hallucinated_answer_low_faithfulness() {
        let judge = FakeJudge::content("FAITHFULNESS: 0.05\nANSWER_RELEVANCY: 0.2");
        let g = score_grounding(
            Some(&judge),
            None,
            "what indexing does veda use?",
            "Veda is a brand of bottled water from the Alps.",
            &["Veda uses HNSW graphs for ANN search.".to_owned()],
        )
        .await;
        assert_eq!(g.method, "llm_judge");
        assert!(
            g.faithfulness.unwrap() < 0.2,
            "hallucination → low faithfulness"
        );
    }

    #[tokio::test]
    async fn no_context_returns_none_none() {
        let judge = FakeJudge::content("FAITHFULNESS: 1.0\nANSWER_RELEVANCY: 1.0");
        let g = score_grounding(Some(&judge), None, "q", "some answer", &[]).await;
        assert_eq!(g.method, "no_context");
        assert_eq!(g.faithfulness, None);
        assert_eq!(g.answer_relevancy, None);
    }

    #[tokio::test]
    async fn judge_parse_failure_falls_back_to_embedding() {
        // The judge returns unparseable prose → fall through to embedding.
        let judge = FakeJudge::content("I cannot rate this, sorry.");
        let embedder = mock_embedder();
        let g = score_grounding(
            Some(&judge),
            Some(&embedder),
            "query text",
            "answer text",
            &["context text".to_owned()],
        )
        .await;
        assert_eq!(
            g.method, "embedding",
            "unparseable judge → embedding fallback"
        );
        assert!(g.faithfulness.is_some(), "embedding faithfulness computed");
        assert!(
            g.answer_relevancy.is_some(),
            "embedding relevancy computed from query"
        );
    }

    #[tokio::test]
    async fn no_llm_uses_embedding_path() {
        let embedder = mock_embedder();
        // Identical answer + single context → cosine 1.0 on the mock
        // deterministic embedder.
        let g = score_grounding::<FakeJudge>(
            None,
            Some(&embedder),
            "the query",
            "identical text",
            &["identical text".to_owned()],
        )
        .await;
        assert_eq!(g.method, "embedding");
        assert!(
            (g.faithfulness.unwrap() - 1.0).abs() < 1e-6,
            "identical answer/context → cosine 1.0, got {:?}",
            g.faithfulness
        );
    }

    #[tokio::test]
    async fn no_llm_no_embedder_is_none_not_error() {
        let g = score_grounding::<FakeJudge>(None, None, "q", "a", &["c".to_owned()]).await;
        assert_eq!(g.method, "embedding");
        assert_eq!(g.faithfulness, None);
        assert_eq!(g.answer_relevancy, None);
    }
}
