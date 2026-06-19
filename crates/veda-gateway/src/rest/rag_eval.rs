//! RAG / agent evaluation harness + production-trace flywheel.
//!
//! Three pieces, all tenant-scoped through the gateway's existing
//! [`SavedObjectStore`](crate::rest::saved_objects::SavedObjectStore)
//! (no new persistence layer):
//!
//!   * **Eval sets** ([`EvalSet`] / [`EvalCase`]) — a labeled set of
//!     queries with optional `expected_chunk_ids` (the known-good
//!     retrieval) and an optional `gold_answer`. CRUD over
//!     `/v1/eval-sets`, stored as `SavedObject` type `rag_eval_set`.
//!
//!   * **Eval runs** ([`EvalRun`]) — running an agent over every case in
//!     a set, scoring per-case `recall_at_k` (citations ∩ expected /
//!     |expected|, a *proxy* retrieval signal), `faithfulness`, and
//!     `answer_relevancy` (from the run's [`Grounding`], plus an optional
//!     cosine(answer, gold_answer) when a gold answer + text embedder are
//!     available). Aggregates the means + a `pass_rate`, persists the run
//!     as `rag_eval_run`. The agent turn is injected so the core harness
//!     ([`run_eval_set`]) is unit-testable with a fake model.
//!
//!   * **Flywheel** — graduate a production conversation turn (or a
//!     provided turn) into a new [`EvalCase`]: the user query becomes the
//!     case query, that turn's citations become `expected_chunk_ids` (a
//!     regression guard on retrieval), and the assistant answer becomes
//!     the `gold_answer`.
//!
//! Endpoints (all under the tenant/RLS-scoped protected router, like
//! `/converse` — not admin-only):
//!   GET    /v1/eval-sets                              list
//!   POST   /v1/eval-sets                              create/upsert
//!   GET    /v1/eval-sets/{id}                          fetch
//!   PUT    /v1/eval-sets/{id}                          upsert
//!   DELETE /v1/eval-sets/{id}                          delete
//!   POST   /v1/eval-sets/{id}/run                      run the harness
//!   GET    /v1/eval-sets/{id}/runs                     recent runs
//!   POST   /v1/eval-sets/{id}/cases/from-conversation  flywheel

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::agent::grounding::{Citation, Grounding};
use crate::agent::store as agent_store;
use crate::auth::Claims;
use crate::rest::saved_objects::{SavedObject, SavedObjectStore};
use crate::tenant::{self, Tenant, TenantId};

use super::agents::{self, AgentTurnError};
use super::collections::ErrorResponse;
use super::state::AppState;

type GatewayError = (StatusCode, Json<ErrorResponse>);

/// `SavedObject.type` for an [`EvalSet`].
pub const TYPE_EVAL_SET: &str = "rag_eval_set";
/// `SavedObject.type` for an [`EvalRun`].
pub const TYPE_EVAL_RUN: &str = "rag_eval_run";

/// Default pass bars: a case passes when its recall (when scored) and
/// faithfulness (when scored) both clear these. Tunable per run.
const DEFAULT_RECALL_BAR: f64 = 0.8;
const DEFAULT_FAITHFULNESS_BAR: f64 = 0.8;

fn not_found(msg: impl Into<String>) -> GatewayError {
    (StatusCode::NOT_FOUND, Json(ErrorResponse { error: msg.into() }))
}
fn service_unavailable(msg: impl Into<String>) -> GatewayError {
    (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: msg.into() }))
}
fn bad_request(msg: impl Into<String>) -> GatewayError {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg.into() }))
}

// ── Data model ───────────────────────────────────────────────────────────

/// One labeled evaluation case.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EvalCase {
    /// Case id. Optional on create — the server assigns `case-{uuid}` when
    /// absent/empty (mirrors `SloDef`/`SyntheticProbe`).
    #[serde(default)]
    pub id: String,
    pub query: String,
    /// Known-good retrieval for this query — recall@k is measured against
    /// these. Empty → recall is not scored for this case.
    #[serde(default)]
    pub expected_chunk_ids: Vec<String>,
    /// Optional reference answer for answer-similarity scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gold_answer: Option<String>,
}

/// A labeled evaluation set: many cases run against one agent.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EvalSet {
    /// Set id. Optional on create — the server assigns `eval-{uuid}` when
    /// absent/empty.
    #[serde(default)]
    pub id: String,
    pub name: String,
    /// Agent the set evaluates by default; `None`/absent → `veda-agent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub cases: Vec<EvalCase>,
}

/// Per-case scored result of an eval run.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub query: String,
    /// The agent's answer for this case.
    pub response: String,
    /// Chunk ids the answer cited (from the run's citations).
    #[serde(default)]
    pub retrieved_chunk_ids: Vec<String>,
    /// recall@k = |retrieved ∩ expected| / |expected|. `None` when the
    /// case has no `expected_chunk_ids`. A retrieval *proxy*.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_at_k: Option<f64>,
    /// Faithfulness from the run's grounding (`None` when not scored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faithfulness: Option<f64>,
    /// Answer relevancy: cosine(answer, gold_answer) when a gold answer +
    /// embedder are available, else the run's grounding answer_relevancy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_relevancy: Option<f64>,
    /// Whether this case met the pass bars (recall + faithfulness, where
    /// each is ignored when unscored).
    pub passed: bool,
}

/// Aggregate statistics over an eval run's cases.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RunAggregate {
    pub case_count: usize,
    /// Mean recall@k over cases that had `expected_chunk_ids`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_recall_at_k: Option<f64>,
    /// Mean faithfulness over cases that were faithfulness-scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_faithfulness: Option<f64>,
    /// Mean answer_relevancy over cases that were relevancy-scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_answer_relevancy: Option<f64>,
    /// Fraction of cases that met the pass bars.
    pub pass_rate: f64,
}

/// A persisted eval run: per-case results + aggregate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EvalRun {
    pub id: String,
    pub eval_set_id: String,
    pub agent_id: String,
    pub created_at: i64,
    pub recall_bar: f64,
    pub faithfulness_bar: f64,
    pub aggregate: RunAggregate,
    #[serde(default)]
    pub cases: Vec<CaseResult>,
}

/// Body for `POST /v1/eval-sets/{id}/run` — all fields optional.
#[derive(Deserialize, Default)]
pub struct RunRequest {
    /// Override the set's agent for this run.
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub recall_bar: Option<f64>,
    #[serde(default)]
    pub faithfulness_bar: Option<f64>,
}

/// Body for the flywheel endpoint. Either carries a `conversation_id`
/// (graduate the last user→assistant turn) or an explicit turn.
#[derive(Deserialize, Default)]
pub struct FromConversationRequest {
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub retrieved_chunk_ids: Option<Vec<String>>,
    #[serde(default)]
    pub answer: Option<String>,
}

// ── Persistence helpers (SavedObjects) ───────────────────────────────────

fn set_to_saved(set: &EvalSet) -> SavedObject {
    SavedObject {
        id: set.id.clone(),
        kind: TYPE_EVAL_SET.to_owned(),
        name: set.name.clone(),
        body: serde_json::to_value(set).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    }
}

async fn save_set(store: &SavedObjectStore, set: &EvalSet) {
    store.put(set_to_saved(set)).await;
}

async fn get_set(store: &SavedObjectStore, id: &str) -> Option<EvalSet> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_EVAL_SET {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

async fn list_sets(store: &SavedObjectStore) -> Vec<EvalSet> {
    store
        .list_by_type(TYPE_EVAL_SET)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}

async fn save_run(store: &SavedObjectStore, run: &EvalRun) {
    let obj = SavedObject {
        id: run.id.clone(),
        kind: TYPE_EVAL_RUN.to_owned(),
        name: format!("run {}", &run.id),
        body: serde_json::to_value(run).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    };
    store.put(obj).await;
}

async fn list_runs_for_set(store: &SavedObjectStore, eval_set_id: &str) -> Vec<EvalRun> {
    store
        .list_by_type(TYPE_EVAL_RUN)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value::<EvalRun>(o.body).ok())
        .filter(|r| r.eval_set_id == eval_set_id)
        .collect()
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ── Scoring (pure, unit-testable) ────────────────────────────────────────

/// Cosine similarity of two equal-length vectors; 0.0 on mismatch / zero
/// norm. (Mirrors `agent::grounding::cosine`, kept local — that one is
/// private to its module.)
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
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

/// recall@k for one case: |retrieved ∩ expected| / |expected|. `None`
/// when the case carries no expected ids.
fn recall_at_k(retrieved: &[String], expected: &[String]) -> Option<f64> {
    if expected.is_empty() {
        return None;
    }
    let got: std::collections::HashSet<&str> = retrieved.iter().map(String::as_str).collect();
    let hit = expected.iter().filter(|e| got.contains(e.as_str())).count();
    Some(hit as f64 / expected.len() as f64)
}

/// Assemble a [`CaseResult`] from one case + the run outputs. `gold_cosine`
/// is a precomputed cosine(answer, gold_answer) when available (best-effort,
/// done by the caller off the embedder); it takes precedence over the
/// grounding's answer_relevancy for the relevancy signal.
fn score_case(
    case: &EvalCase,
    response: &str,
    citations: &[Citation],
    grounding: Option<&Grounding>,
    gold_cosine: Option<f64>,
    recall_bar: f64,
    faithfulness_bar: f64,
) -> CaseResult {
    let retrieved_chunk_ids: Vec<String> =
        citations.iter().map(|c| c.chunk_id.clone()).collect();
    let recall = recall_at_k(&retrieved_chunk_ids, &case.expected_chunk_ids);
    let faithfulness = grounding.and_then(|g| g.faithfulness);
    let answer_relevancy = gold_cosine.or_else(|| grounding.and_then(|g| g.answer_relevancy));

    // A case passes when each *scored* bar is met. An unscored signal is
    // not held against the case (it can't fail on a metric it lacks).
    let recall_ok = recall.map(|r| r >= recall_bar).unwrap_or(true);
    let faith_ok = faithfulness.map(|f| f >= faithfulness_bar).unwrap_or(true);
    let passed = recall_ok && faith_ok;

    CaseResult {
        case_id: case.id.clone(),
        query: case.query.clone(),
        response: response.to_owned(),
        retrieved_chunk_ids,
        recall_at_k: recall,
        faithfulness,
        answer_relevancy,
        passed,
    }
}

/// Mean of the `Some` values in `xs`, or `None` when all are `None`.
fn mean_some<I: IntoIterator<Item = Option<f64>>>(xs: I) -> Option<f64> {
    let vals: Vec<f64> = xs.into_iter().flatten().collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// Aggregate per-case results into a [`RunAggregate`].
fn aggregate(cases: &[CaseResult]) -> RunAggregate {
    let case_count = cases.len();
    let mean_recall_at_k = mean_some(cases.iter().map(|c| c.recall_at_k));
    let mean_faithfulness = mean_some(cases.iter().map(|c| c.faithfulness));
    let mean_answer_relevancy = mean_some(cases.iter().map(|c| c.answer_relevancy));
    let pass_rate = if case_count == 0 {
        0.0
    } else {
        cases.iter().filter(|c| c.passed).count() as f64 / case_count as f64
    };
    RunAggregate {
        case_count,
        mean_recall_at_k,
        mean_faithfulness,
        mean_answer_relevancy,
        pass_rate,
    }
}

/// One agent turn's outputs, as the harness consumes them.
pub struct TurnOutcome {
    pub response: String,
    pub citations: Vec<Citation>,
    pub grounding: Option<Grounding>,
    /// Precomputed cosine(answer, gold_answer), when scored.
    pub gold_cosine: Option<f64>,
}

/// Core harness: run `turn` over every case in `set` and assemble the
/// scored [`EvalRun`]. The turn-runner is injected so this is unit-testable
/// with a fake model (the HTTP handler passes a closure over
/// [`agents::run_agent_turn`] + the embedder). The closure is `async` per
/// case and receives the case; it must never panic (errors should surface
/// as an empty/low-scored outcome at the call site, not here).
async fn run_eval_set<F, Fut>(
    set: &EvalSet,
    agent_id: &str,
    recall_bar: f64,
    faithfulness_bar: f64,
    mut turn: F,
) -> EvalRun
where
    F: FnMut(EvalCase) -> Fut,
    Fut: std::future::Future<Output = TurnOutcome>,
{
    let mut cases = Vec::with_capacity(set.cases.len());
    for case in &set.cases {
        let outcome = turn(case.clone()).await;
        cases.push(score_case(
            case,
            &outcome.response,
            &outcome.citations,
            outcome.grounding.as_ref(),
            outcome.gold_cosine,
            recall_bar,
            faithfulness_bar,
        ));
    }
    let aggregate = aggregate(&cases);
    EvalRun {
        id: uuid::Uuid::new_v4().to_string(),
        eval_set_id: set.id.clone(),
        agent_id: agent_id.to_owned(),
        created_at: now_ms(),
        recall_bar,
        faithfulness_bar,
        aggregate,
        cases,
    }
}

// ── CRUD handlers ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ListSetsResponse {
    pub eval_sets: Vec<EvalSet>,
}

pub async fn list_eval_sets(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ListSetsResponse> {
    let mut sets = list_sets(&state.inner.saved_objects).await;
    sets.retain(|s| tenant::strip_prefix(&tenant, &s.id).is_some());
    for s in &mut sets {
        if let Some(name) = tenant::strip_prefix(&tenant, &s.id) {
            s.id = name.to_owned();
        }
    }
    Json(ListSetsResponse { eval_sets: sets })
}

pub async fn create_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Json(mut set): Json<EvalSet>,
) -> Result<Json<EvalSet>, GatewayError> {
    if set.id.trim().is_empty() {
        set.id = format!("eval-{}", uuid::Uuid::new_v4());
    }
    assign_case_ids(&mut set);
    save_and_return_set(&state, &tenant, set).await
}

/// Assign a server id (`case-{uuid}`) to every case whose id is empty.
/// Cases that arrive with an explicit id keep it.
fn assign_case_ids(set: &mut EvalSet) {
    for case in &mut set.cases {
        if case.id.trim().is_empty() {
            case.id = format!("case-{}", uuid::Uuid::new_v4());
        }
    }
}

pub async fn get_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<EvalSet>, GatewayError> {
    let scoped = tenant::scope(&tenant, &id);
    let mut set = get_set(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| not_found(format!("eval set '{id}' not found")))?;
    set.id = id;
    Ok(Json(set))
}

pub async fn put_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(mut set): Json<EvalSet>,
) -> Result<Json<EvalSet>, GatewayError> {
    set.id = id;
    assign_case_ids(&mut set);
    save_and_return_set(&state, &tenant, set).await
}

pub async fn delete_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> StatusCode {
    let scoped = tenant::scope(&tenant, &id);
    state.inner.saved_objects.remove(&scoped).await;
    StatusCode::NO_CONTENT
}

async fn save_and_return_set(
    state: &AppState,
    tenant: &TenantId,
    mut set: EvalSet,
) -> Result<Json<EvalSet>, GatewayError> {
    let user_id = set.id.clone();
    set.id = tenant::scope(tenant, &user_id);
    save_set(&state.inner.saved_objects, &set).await;
    set.id = user_id;
    Ok(Json(set))
}

// ── Run + runs-over-time handlers ────────────────────────────────────────

/// Resolve the text embedder (clone the config off the lock, build the
/// client off the lock — mirrors `run_agent_turn`'s grounding path).
async fn text_embedder(state: &AppState) -> Option<veda_core::embedding_client::EmbeddingClient> {
    use veda_core::embedding::Modality;
    let registry = state.inner.embedding_registry.read().await;
    registry
        .find_by_modality(Modality::Text)
        .first()
        .map(|c| veda_core::embedding_client::EmbeddingClient::new((*c).clone()))
}

/// Best-effort cosine(answer, gold). `None` when no gold answer / embedder
/// or any embed call fails — never an error.
async fn gold_cosine(
    embedder: Option<&veda_core::embedding_client::EmbeddingClient>,
    answer: &str,
    gold: Option<&str>,
) -> Option<f64> {
    let (embedder, gold) = (embedder?, gold?);
    let av = embedder.embed_text(answer).await.ok()?;
    let gv = embedder.embed_text(gold).await.ok()?;
    Some((cosine(&av, &gv) as f64).clamp(0.0, 1.0))
}

pub async fn run_eval(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    claims: Option<axum::Extension<Claims>>,
    body: Option<Json<RunRequest>>,
) -> Result<Json<EvalRun>, GatewayError> {
    let req = body.map(|Json(b)| b).unwrap_or_default();

    // Resolve the set (tenant-scoped).
    let scoped_set_id = tenant::scope(&tenant, &id);
    let mut set = get_set(&state.inner.saved_objects, &scoped_set_id)
        .await
        .ok_or_else(|| not_found(format!("eval set '{id}' not found")))?;
    // Keep a user-facing id on the persisted run + reply.
    set.id = id.clone();

    // Agent: request override → set's agent → veda-agent default.
    let agent_id = req
        .agent_id
        .clone()
        .or_else(|| set.agent_id.clone())
        .unwrap_or_else(|| "veda-agent".to_owned());
    let recall_bar = req.recall_bar.unwrap_or(DEFAULT_RECALL_BAR);
    let faithfulness_bar = req.faithfulness_bar.unwrap_or(DEFAULT_FAITHFULNESS_BAR);

    let claims = claims.map(|axum::Extension(c)| c);
    let user = agents::rls_user(&claims);
    let embedder = text_embedder(&state).await;

    // Fail fast with 503 (mirror /converse) when no LLM is configured,
    // BEFORE running any cases, by probing one turn — but to avoid a
    // pointless empty-set probe we only short-circuit on the first case.
    // The injected closure maps any turn error to a low/empty outcome,
    // except NoLlm which we surface as 503 via a shared flag.
    let no_llm = std::sync::atomic::AtomicBool::new(false);
    let run = {
        let state = &state;
        let tenant = &tenant;
        let user = user.as_deref();
        let embedder = embedder.as_ref();
        let no_llm = &no_llm;
        let turn_agent_id = agent_id.clone();
        run_eval_set(&set, &agent_id, recall_bar, faithfulness_bar, move |case| {
            let gold = case.gold_answer.clone();
            let turn_agent_id = turn_agent_id.clone();
            async move {
                match agents::run_agent_turn(
                    state, tenant, user, &turn_agent_id, &[], &case.query, None,
                )
                .await
                {
                    Ok((_agent, result, citations, grounding, _cache_hit, _guardrails)) => {
                        let gc = gold_cosine(embedder, &result.response, gold.as_deref()).await;
                        TurnOutcome {
                            response: result.response,
                            citations,
                            grounding,
                            gold_cosine: gc,
                        }
                    }
                    Err(AgentTurnError::NoLlm) => {
                        no_llm.store(true, std::sync::atomic::Ordering::SeqCst);
                        TurnOutcome {
                            response: String::new(),
                            citations: Vec::new(),
                            grounding: None,
                            gold_cosine: None,
                        }
                    }
                    Err(AgentTurnError::NotFound(_)) => TurnOutcome {
                        response: String::new(),
                        citations: Vec::new(),
                        grounding: None,
                        gold_cosine: None,
                    },
                }
            }
        })
        .await
    };

    if no_llm.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(service_unavailable(
            "VEDA_LLM_ENDPOINT not configured. Set the env var to enable eval runs.",
        ));
    }

    // Persist under a tenant-scoped id; return the user-facing run.
    let mut stored = run.clone();
    stored.id = tenant::scope(&tenant, &run.id);
    save_run(&state.inner.saved_objects, &stored).await;
    Ok(Json(run))
}

#[derive(Serialize)]
pub struct ListRunsResponse {
    pub runs: Vec<EvalRun>,
}

pub async fn list_runs(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Json<ListRunsResponse> {
    // Runs reference the user-facing set id (we store that on the body),
    // but are persisted under tenant-scoped run ids — filter by tenant
    // prefix on the run id, then strip it for display.
    let scoped_prefix_ok = |rid: &str| tenant::strip_prefix(&tenant, rid).is_some();
    let mut runs: Vec<EvalRun> = list_runs_for_set(&state.inner.saved_objects, &id)
        .await
        .into_iter()
        .filter(|r| scoped_prefix_ok(&r.id))
        .collect();
    for r in &mut runs {
        if let Some(name) = tenant::strip_prefix(&tenant, &r.id) {
            r.id = name.to_owned();
        }
    }
    runs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(ListRunsResponse { runs })
}

// ── Flywheel: graduate a conversation turn into a case ───────────────────

pub async fn case_from_conversation(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(req): Json<FromConversationRequest>,
) -> Result<Json<EvalSet>, GatewayError> {
    let scoped_set_id = tenant::scope(&tenant, &id);
    let mut set = get_set(&state.inner.saved_objects, &scoped_set_id)
        .await
        .ok_or_else(|| not_found(format!("eval set '{id}' not found")))?;

    // Build the case either from an explicit turn or a stored conversation.
    let case = if let Some(cid) = &req.conversation_id {
        let scoped_cid = tenant::scope(&tenant, cid);
        let conv = agent_store::get_conversation(&state.inner.saved_objects, &scoped_cid)
            .await
            .ok_or_else(|| not_found(format!("conversation '{cid}' not found")))?;
        let turn = extract_turn(&conv).ok_or_else(|| {
            bad_request(format!(
                "conversation '{cid}' has no user→assistant turn to graduate"
            ))
        })?;
        EvalCase {
            id: format!("case-{}", uuid::Uuid::new_v4()),
            query: turn.query,
            expected_chunk_ids: turn.expected_chunk_ids,
            gold_answer: turn.gold_answer,
        }
    } else {
        let query = req
            .query
            .clone()
            .filter(|q| !q.trim().is_empty())
            .ok_or_else(|| bad_request("provide either conversation_id or a non-empty query"))?;
        EvalCase {
            id: format!("case-{}", uuid::Uuid::new_v4()),
            query,
            expected_chunk_ids: req.retrieved_chunk_ids.clone().unwrap_or_default(),
            gold_answer: req.answer.clone(),
        }
    };

    set.cases.push(case);
    // Persist (id is already tenant-scoped here).
    save_set(&state.inner.saved_objects, &set).await;
    set.id = id;
    Ok(Json(set))
}

/// The fields pulled from a conversation turn for the flywheel.
struct GraduatedTurn {
    query: String,
    expected_chunk_ids: Vec<String>,
    gold_answer: Option<String>,
}

/// Extract the last user→assistant turn from a conversation: the last
/// user message is the query, the assistant answer is the gold answer,
/// and the citations from that turn's steps are the expected ids.
fn extract_turn(conv: &crate::agent::model::Conversation) -> Option<GraduatedTurn> {
    // Last user message = query.
    let query = conv
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.clone())?;
    // Last assistant message = gold answer (best-effort).
    let gold_answer = conv
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant")
        .and_then(|m| m.content.clone());
    // Expected ids = citations extracted from the conversation's step
    // trace (the known-good retrieval for this turn).
    let citations = crate::agent::grounding::extract_citations(&conv.steps);
    let expected_chunk_ids = citations.into_iter().map(|c| c.chunk_id).collect();
    Some(GraduatedTurn {
        query,
        expected_chunk_ids,
        gold_answer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::grounding::{Citation, Grounding};
    use crate::agent::model::{AgentStep, Conversation, ConversationMessage};

    fn cite(id: &str) -> Citation {
        Citation {
            chunk_id: id.to_owned(),
            collection: Some("docs".to_owned()),
            snippet: Some("snip".to_owned()),
            score: Some(0.1),
        }
    }

    fn grounding(f: f64, r: f64) -> Grounding {
        Grounding {
            faithfulness: Some(f),
            answer_relevancy: Some(r),
            method: "llm_judge".to_owned(),
        }
    }

    // ── recall@k ──────────────────────────────────────────────────────────

    #[test]
    fn recall_full_partial_zero_and_none() {
        let exp = vec!["a".to_owned(), "b".to_owned()];
        assert_eq!(recall_at_k(&["a".into(), "b".into()], &exp), Some(1.0));
        assert_eq!(recall_at_k(&["a".into(), "z".into()], &exp), Some(0.5));
        assert_eq!(recall_at_k(&["z".into()], &exp), Some(0.0));
        // No expected ids → None (not scored).
        assert_eq!(recall_at_k(&["a".into()], &[]), None);
    }

    // ── score_case ──────────────────────────────────────────────────────────

    #[test]
    fn score_case_uses_gold_cosine_over_grounding_relevancy() {
        let case = EvalCase {
            id: "c1".to_owned(),
            query: "q".to_owned(),
            expected_chunk_ids: vec!["a".to_owned(), "b".to_owned()],
            gold_answer: Some("gold".to_owned()),
        };
        let cites = vec![cite("a"), cite("b")];
        let g = grounding(0.9, 0.4);
        let r = score_case(&case, "ans", &cites, Some(&g), Some(0.95), 0.8, 0.8);
        assert_eq!(r.recall_at_k, Some(1.0));
        assert_eq!(r.faithfulness, Some(0.9));
        // gold cosine (0.95) wins over grounding relevancy (0.4).
        assert_eq!(r.answer_relevancy, Some(0.95));
        assert!(r.passed, "recall 1.0 + faithfulness 0.9 clear the 0.8 bars");
        assert_eq!(r.retrieved_chunk_ids, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn score_case_fails_on_low_recall() {
        let case = EvalCase {
            id: "c1".to_owned(),
            query: "q".to_owned(),
            expected_chunk_ids: vec!["a".to_owned(), "b".to_owned()],
            gold_answer: None,
        };
        let cites = vec![cite("a")]; // recall 0.5 < 0.8
        let g = grounding(0.95, 0.9);
        let r = score_case(&case, "ans", &cites, Some(&g), None, 0.8, 0.8);
        assert_eq!(r.recall_at_k, Some(0.5));
        assert!(!r.passed, "recall 0.5 below the 0.8 bar fails the case");
        // No gold → relevancy falls back to grounding.
        assert_eq!(r.answer_relevancy, Some(0.9));
    }

    #[test]
    fn score_case_unscored_signals_do_not_fail() {
        // No expected ids and no grounding → both bars vacuously pass.
        let case = EvalCase {
            id: "c1".to_owned(),
            query: "q".to_owned(),
            expected_chunk_ids: vec![],
            gold_answer: None,
        };
        let r = score_case(&case, "ans", &[], None, None, 0.8, 0.8);
        assert_eq!(r.recall_at_k, None);
        assert_eq!(r.faithfulness, None);
        assert!(r.passed, "a case with no scorable bars is not penalized");
    }

    // ── aggregate ──────────────────────────────────────────────────────────

    #[test]
    fn aggregate_means_skip_none_and_compute_pass_rate() {
        let cases = vec![
            CaseResult {
                case_id: "c1".to_owned(),
                query: "q1".to_owned(),
                response: "r1".to_owned(),
                retrieved_chunk_ids: vec!["a".to_owned()],
                recall_at_k: Some(1.0),
                faithfulness: Some(0.9),
                answer_relevancy: Some(0.8),
                passed: true,
            },
            CaseResult {
                case_id: "c2".to_owned(),
                query: "q2".to_owned(),
                response: "r2".to_owned(),
                retrieved_chunk_ids: vec![],
                recall_at_k: None, // skipped in the mean
                faithfulness: Some(0.5),
                answer_relevancy: None,
                passed: false,
            },
        ];
        let agg = aggregate(&cases);
        assert_eq!(agg.case_count, 2);
        // Only c1 contributes to recall mean.
        assert_eq!(agg.mean_recall_at_k, Some(1.0));
        assert_eq!(agg.mean_faithfulness, Some((0.9 + 0.5) / 2.0));
        assert_eq!(agg.mean_answer_relevancy, Some(0.8));
        assert_eq!(agg.pass_rate, 0.5);
    }

    // ── run_eval_set with a fake injected turn ──────────────────────────────

    #[tokio::test]
    async fn run_eval_set_scores_per_case_and_aggregate() {
        let set = EvalSet {
            id: "s1".to_owned(),
            name: "set".to_owned(),
            agent_id: None,
            cases: vec![
                EvalCase {
                    id: "c1".to_owned(),
                    query: "what index?".to_owned(),
                    expected_chunk_ids: vec!["id-1".to_owned()],
                    gold_answer: None,
                },
                EvalCase {
                    id: "c2".to_owned(),
                    query: "off topic?".to_owned(),
                    expected_chunk_ids: vec!["id-9".to_owned()],
                    gold_answer: None,
                },
            ],
        };

        // Fake turn: c1 retrieves id-1 (recall 1.0, faithful), c2 retrieves
        // the wrong chunk (recall 0.0).
        let run = run_eval_set(&set, "veda-agent", 0.8, 0.8, |case| async move {
            if case.id == "c1" {
                TurnOutcome {
                    response: "HNSW".to_owned(),
                    citations: vec![cite("id-1")],
                    grounding: Some(grounding(0.95, 0.9)),
                    gold_cosine: None,
                }
            } else {
                TurnOutcome {
                    response: "no idea".to_owned(),
                    citations: vec![cite("id-x")],
                    grounding: Some(grounding(0.2, 0.1)),
                    gold_cosine: None,
                }
            }
        })
        .await;

        assert_eq!(run.eval_set_id, "s1");
        assert_eq!(run.agent_id, "veda-agent");
        assert_eq!(run.cases.len(), 2);
        // c1: recall 1.0, faithful 0.95 → passed.
        assert_eq!(run.cases[0].recall_at_k, Some(1.0));
        assert!(run.cases[0].passed);
        // c2: recall 0.0, faithful 0.2 → failed.
        assert_eq!(run.cases[1].recall_at_k, Some(0.0));
        assert!(!run.cases[1].passed);
        // Aggregate: mean recall 0.5, pass_rate 0.5.
        assert_eq!(run.aggregate.case_count, 2);
        assert_eq!(run.aggregate.mean_recall_at_k, Some(0.5));
        assert_eq!(run.aggregate.pass_rate, 0.5);
    }

    // ── flywheel: extract_turn ──────────────────────────────────────────────

    #[test]
    fn extract_turn_builds_case_with_query_expected_and_gold() {
        let conv = Conversation {
            id: "conv-1".to_owned(),
            agent_id: "veda-agent".to_owned(),
            title: None,
            messages: vec![
                ConversationMessage {
                    role: "user".to_owned(),
                    content: Some("what index does veda use?".to_owned()),
                },
                ConversationMessage {
                    role: "assistant".to_owned(),
                    content: Some("Veda uses HNSW.".to_owned()),
                },
            ],
            steps: vec![
                AgentStep::ToolCall {
                    tool_call_id: "c1".to_owned(),
                    tool_id: "search".to_owned(),
                    args: serde_json::json!({ "collection": "docs", "query": "hnsw" }),
                },
                AgentStep::ToolResult {
                    tool_call_id: "c1".to_owned(),
                    result: serde_json::json!({
                        "count": 1,
                        "hits": [{ "id": "id-1", "distance": 0.1, "fields": { "body": "HNSW" } }]
                    }),
                    is_error: false,
                    duration_ms: None,
                },
            ],
            feedback: Vec::new(),
        };
        let turn = extract_turn(&conv).expect("turn present");
        assert_eq!(turn.query, "what index does veda use?");
        assert_eq!(turn.gold_answer.as_deref(), Some("Veda uses HNSW."));
        assert_eq!(turn.expected_chunk_ids, vec!["id-1".to_owned()]);
    }

    #[test]
    fn extract_turn_none_without_user_message() {
        let conv = Conversation {
            id: "conv-2".to_owned(),
            agent_id: "veda-agent".to_owned(),
            title: None,
            messages: vec![],
            steps: vec![],
            feedback: Vec::new(),
        };
        assert!(extract_turn(&conv).is_none());
    }
}
