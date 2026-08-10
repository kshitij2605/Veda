//! OLAP-style analytics aggregation endpoints.
//!
//! Provides three read-only analytics endpoints over a collection's stored records:
//!
//! - `GET /v1/collections/{name}/aggregate`  — scalar metric (count/sum/avg/min/max)
//!   with optional `group_by` and `filter`.
//! - `GET /v1/collections/{name}/histogram`  — equal-width histogram over a numeric field.
//! - `GET /v1/collections/{name}/field-stats` — cardinality, null count, numeric stats,
//!   string length stats.
//!
//! Record walk mirrors `list_records`: iterates growing + sealed segments, skips deleted
//! and TTL-expired records, and applies the optional filter DSL via `evaluate_record`.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use veda_core::record::{FieldValue, Record};
use veda_query::filter::parser::FilterParser;
use veda_query::filter::record_eval::evaluate_record;

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::tenant::{self, Tenant};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Extract a numeric (f64) value from a FieldValue. Returns None for non-numeric.
fn as_f64(fv: &FieldValue) -> Option<f64> {
    match fv {
        FieldValue::Float64(v) => Some(*v),
        FieldValue::Float32(v) => Some(*v as f64),
        FieldValue::Int64(v) => Some(*v as f64),
        FieldValue::Int32(v) => Some(*v as f64),
        _ => None,
    }
}

/// Stringify a FieldValue for use as a group-by key.
fn field_as_key(fv: &FieldValue) -> String {
    match fv {
        FieldValue::String(s) => s.clone(),
        FieldValue::Int64(v) => v.to_string(),
        FieldValue::Int32(v) => v.to_string(),
        FieldValue::Float64(v) => v.to_string(),
        FieldValue::Float32(v) => v.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        _ => "<other>".to_owned(),
    }
}

type GatewayError = (StatusCode, Json<ErrorResponse>);

fn not_found(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse { error: msg.into() }),
    )
}
fn bad_request(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg.into() }),
    )
}
fn internal(msg: impl Into<String>) -> GatewayError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: msg.into() }),
    )
}

// ── Aggregate ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AggregateQuery {
    /// The field to aggregate over (must be numeric for sum/avg/min/max).
    pub field: String,
    /// Metric: count | sum | avg | min | max
    pub metric: String,
    /// Optional field to group by.
    pub group_by: Option<String>,
    /// Optional filter DSL string.
    pub filter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    pub groups: Vec<AggregateGroup>,
}

#[derive(Debug, Serialize)]
pub struct AggregateGroup {
    pub key: String,
    pub value: f64,
}

struct Accumulator {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
}

impl Accumulator {
    fn push(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }
}

/// Coarse failure category for [`aggregate_core`], so non-HTTP callers
/// (e.g. the agent `aggregate` tool) can map it to their own error kind
/// without depending on `StatusCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggErrorKind {
    NotFound,
    BadRequest,
    Internal,
}

/// Structured error from [`aggregate_core`].
#[derive(Debug, Clone)]
pub struct AggError {
    pub kind: AggErrorKind,
    pub message: String,
}

impl AggError {
    fn new(kind: AggErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

pub async fn aggregate(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Query(q): Query<AggregateQuery>,
) -> Result<Json<AggregateResponse>, GatewayError> {
    let name = tenant::scope(&tenant, &name);
    // Resolve the alias like `/search` and `/records` do — otherwise the
    // aggregation runs over the abandoned base collection while every other
    // read of the same name follows the alias to the swapped-in sibling.
    let name = state.inner.aliases.resolve(&name).await;
    aggregate_core(&state, &name, &q)
        .await
        .map(Json)
        .map_err(|e| match e.kind {
            AggErrorKind::NotFound => not_found(e.message),
            AggErrorKind::BadRequest => bad_request(e.message),
            AggErrorKind::Internal => internal(e.message),
        })
}

/// Core aggregation over a *tenant-scoped* collection name. Shared by the
/// REST `/aggregate` handler and the agent `aggregate` tool so both apply
/// identical record-walk, filter, and metric semantics.
pub async fn aggregate_core(
    state: &AppState,
    name: &str,
    q: &AggregateQuery,
) -> Result<AggregateResponse, AggError> {
    // Validate collection exists.
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(name) {
            return Err(AggError::new(
                AggErrorKind::NotFound,
                format!("collection '{name}' not found"),
            ));
        }
    }

    let metric = q.metric.as_str();
    let is_numeric_metric = matches!(metric, "sum" | "avg" | "min" | "max");

    // Parse optional filter.
    let filter_expr = q
        .filter
        .as_deref()
        .map(|s| {
            FilterParser::parse(s).map_err(|e| {
                AggError::new(AggErrorKind::BadRequest, format!("invalid filter: {e}"))
            })
        })
        .transpose()?;

    let now = now_secs();

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(name).ok_or_else(|| {
        AggError::new(AggErrorKind::Internal, "no segment managers for collection")
    })?;

    // key → Accumulator
    let mut accums: HashMap<String, Accumulator> = HashMap::new();
    let no_group = q.group_by.is_none();
    let bucket_key = "".to_owned(); // used when no group_by

    let mut first_non_numeric = false; // track whether we saw any non-numeric values to detect type error

    for sm in shard_map.values() {
        // Growing segment
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                if rec.is_expired_at(now) {
                    continue;
                }
                if let Some(ref fe) = filter_expr
                    && !evaluate_record(fe, rec)
                {
                    continue;
                }
                accumulate_record(
                    rec,
                    &q.field,
                    q.group_by.as_deref(),
                    no_group,
                    is_numeric_metric,
                    &bucket_key,
                    &mut accums,
                    &mut first_non_numeric,
                )?
            }
        }

        // Sealed segments
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if rec.is_expired_at(now) {
                        continue;
                    }
                    if let Some(ref fe) = filter_expr
                        && !evaluate_record(fe, &rec)
                    {
                        continue;
                    }
                    accumulate_record(
                        &rec,
                        &q.field,
                        q.group_by.as_deref(),
                        no_group,
                        is_numeric_metric,
                        &bucket_key,
                        &mut accums,
                        &mut first_non_numeric,
                    )?
                }
            }
        }
    }
    drop(managers);

    // Build result groups sorted by key.
    let mut groups: Vec<AggregateGroup> = accums
        .into_iter()
        .map(|(key, acc)| {
            let value = match metric {
                "count" => acc.count as f64,
                "sum" => acc.sum,
                "avg" => {
                    if acc.count == 0 {
                        0.0
                    } else {
                        acc.sum / acc.count as f64
                    }
                }
                "min" => {
                    if acc.count == 0 {
                        0.0
                    } else {
                        acc.min
                    }
                }
                "max" => {
                    if acc.count == 0 {
                        0.0
                    } else {
                        acc.max
                    }
                }
                _ => acc.count as f64,
            };
            AggregateGroup { key, value }
        })
        .collect();
    groups.sort_by(|a, b| a.key.cmp(&b.key));

    Ok(AggregateResponse { groups })
}

#[allow(clippy::too_many_arguments)]
fn accumulate_record(
    rec: &Record,
    field: &str,
    group_by: Option<&str>,
    no_group: bool,
    is_numeric_metric: bool,
    bucket_key: &str,
    accums: &mut HashMap<String, Accumulator>,
    saw_non_numeric: &mut bool,
) -> Result<(), AggError> {
    let fv = rec.get(field);

    // For numeric metrics, validate the value is numeric.
    if is_numeric_metric {
        match fv {
            None | Some(FieldValue::Null) => {
                // null/missing — skip for numeric metrics
                return Ok(());
            }
            Some(v) => {
                if as_f64(v).is_none() {
                    if !*saw_non_numeric {
                        *saw_non_numeric = true;
                        return Err(AggError::new(
                            AggErrorKind::BadRequest,
                            format!("field '{field}' is not numeric; cannot apply numeric metric"),
                        ));
                    }
                    return Ok(());
                }
            }
        }
    }

    let key = if no_group {
        bucket_key.to_owned()
    } else if let Some(gb) = group_by {
        match rec.get(gb) {
            Some(v) => field_as_key(v),
            None => "<null>".to_owned(),
        }
    } else {
        bucket_key.to_owned()
    };

    let acc = accums.entry(key).or_default();

    if is_numeric_metric {
        if let Some(v) = fv.and_then(as_f64) {
            acc.push(v);
        }
    } else {
        // count: just increment regardless of value
        acc.count += 1;
    }

    Ok(())
}

// ── Histogram ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct HistogramQuery {
    pub field: String,
    #[serde(default = "default_bins")]
    pub bins: usize,
    pub filter: Option<String>,
}

fn default_bins() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct HistogramResponse {
    pub bins: Vec<HistogramBin>,
}

#[derive(Debug, Serialize)]
pub struct HistogramBin {
    pub lo: f64,
    pub hi: f64,
    pub count: u64,
}

pub async fn histogram(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Query(q): Query<HistogramQuery>,
) -> Result<Json<HistogramResponse>, GatewayError> {
    let name = tenant::scope(&tenant, &name);
    // See `aggregate` — the alias must be resolved before the read.
    let name = state.inner.aliases.resolve(&name).await;

    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err(not_found(format!("collection '{name}' not found")));
        }
    }

    let bins = q.bins.max(1);

    let filter_expr = q
        .filter
        .as_deref()
        .map(|s| FilterParser::parse(s).map_err(|e| bad_request(format!("invalid filter: {e}"))))
        .transpose()?;

    let now = now_secs();

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers
        .get(&name)
        .ok_or_else(|| internal("no segment managers for collection"))?;

    // First pass: collect values.
    let mut values: Vec<f64> = Vec::new();

    for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                if rec.is_expired_at(now) {
                    continue;
                }
                if let Some(ref fe) = filter_expr
                    && !evaluate_record(fe, rec)
                {
                    continue;
                }
                if let Some(v) = rec.get(&q.field).and_then(as_f64) {
                    values.push(v);
                }
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if rec.is_expired_at(now) {
                        continue;
                    }
                    if let Some(ref fe) = filter_expr
                        && !evaluate_record(fe, &rec)
                    {
                        continue;
                    }
                    if let Some(v) = rec.get(&q.field).and_then(as_f64) {
                        values.push(v);
                    }
                }
            }
        }
    }
    drop(managers);

    if values.is_empty() {
        // Return `bins` empty bins with lo=0, hi=0.
        let empty = (0..bins)
            .map(|_| HistogramBin {
                lo: 0.0,
                hi: 0.0,
                count: 0,
            })
            .collect();
        return Ok(Json(HistogramResponse { bins: empty }));
    }

    let min = values.iter().cloned().fold(f64::MAX, f64::min);
    let max = values.iter().cloned().fold(f64::MIN, f64::max);

    let width = if (max - min).abs() < f64::EPSILON {
        1.0 // all values equal; single bin covers [v, v+1)
    } else {
        (max - min) / bins as f64
    };

    let mut counts = vec![0u64; bins];
    for v in &values {
        let idx = ((v - min) / width) as usize;
        // Clamp the last value (max) into the last bin.
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let result_bins: Vec<HistogramBin> = (0..bins)
        .map(|i| HistogramBin {
            lo: min + i as f64 * width,
            hi: min + (i + 1) as f64 * width,
            count: counts[i],
        })
        .collect();

    Ok(Json(HistogramResponse { bins: result_bins }))
}

// ── Field-stats ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FieldStatsQuery {
    pub field: String,
    pub filter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FieldStatsResponse {
    pub field: String,
    pub total_count: u64,
    pub null_count: u64,
    pub cardinality: u64,
    // Numeric-only fields (absent for non-numeric)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg: Option<f64>,
    // String-only fields (absent for non-string)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u64>,
}

pub async fn field_stats(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Query(q): Query<FieldStatsQuery>,
) -> Result<Json<FieldStatsResponse>, GatewayError> {
    let name = tenant::scope(&tenant, &name);
    // See `aggregate` — the alias must be resolved before the read.
    let name = state.inner.aliases.resolve(&name).await;

    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err(not_found(format!("collection '{name}' not found")));
        }
    }

    let filter_expr = q
        .filter
        .as_deref()
        .map(|s| FilterParser::parse(s).map_err(|e| bad_request(format!("invalid filter: {e}"))))
        .transpose()?;

    let now = now_secs();

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers
        .get(&name)
        .ok_or_else(|| internal("no segment managers for collection"))?;

    let mut total_count: u64 = 0;
    let mut null_count: u64 = 0;
    let mut distinct: HashSet<String> = HashSet::new();

    // Numeric accumulators
    let mut num_sum: f64 = 0.0;
    let mut num_min: f64 = f64::MAX;
    let mut num_max: f64 = f64::MIN;
    let mut num_count: u64 = 0;
    let mut is_numeric = false;

    // String accumulators
    let mut str_min_len: u64 = u64::MAX;
    let mut str_max_len: u64 = 0;
    let mut is_string = false;

    for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                if rec.is_expired_at(now) {
                    continue;
                }
                if let Some(ref fe) = filter_expr
                    && !evaluate_record(fe, rec)
                {
                    continue;
                }
                collect_field_stats(
                    rec,
                    &q.field,
                    &mut total_count,
                    &mut null_count,
                    &mut distinct,
                    &mut num_sum,
                    &mut num_min,
                    &mut num_max,
                    &mut num_count,
                    &mut is_numeric,
                    &mut str_min_len,
                    &mut str_max_len,
                    &mut is_string,
                );
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if rec.is_expired_at(now) {
                        continue;
                    }
                    if let Some(ref fe) = filter_expr
                        && !evaluate_record(fe, &rec)
                    {
                        continue;
                    }
                    collect_field_stats(
                        &rec,
                        &q.field,
                        &mut total_count,
                        &mut null_count,
                        &mut distinct,
                        &mut num_sum,
                        &mut num_min,
                        &mut num_max,
                        &mut num_count,
                        &mut is_numeric,
                        &mut str_min_len,
                        &mut str_max_len,
                        &mut is_string,
                    );
                }
            }
        }
    }
    drop(managers);

    let (min, max, avg) = if is_numeric && num_count > 0 {
        (
            Some(num_min),
            Some(num_max),
            Some(num_sum / num_count as f64),
        )
    } else {
        (None, None, None)
    };

    let (min_length, max_length) = if is_string {
        let lo = if str_min_len == u64::MAX {
            0
        } else {
            str_min_len
        };
        (Some(lo), Some(str_max_len))
    } else {
        (None, None)
    };

    Ok(Json(FieldStatsResponse {
        field: q.field,
        total_count,
        null_count,
        cardinality: distinct.len() as u64,
        min,
        max,
        avg,
        min_length,
        max_length,
    }))
}

#[allow(clippy::too_many_arguments)]
fn collect_field_stats(
    rec: &Record,
    field: &str,
    total_count: &mut u64,
    null_count: &mut u64,
    distinct: &mut HashSet<String>,
    num_sum: &mut f64,
    num_min: &mut f64,
    num_max: &mut f64,
    num_count: &mut u64,
    is_numeric: &mut bool,
    str_min_len: &mut u64,
    str_max_len: &mut u64,
    is_string: &mut bool,
) {
    *total_count += 1;
    match rec.get(field) {
        None | Some(FieldValue::Null) => {
            *null_count += 1;
        }
        Some(fv) => {
            // Cardinality key
            let key = field_as_key(fv);
            distinct.insert(key);

            // Numeric stats
            if let Some(v) = as_f64(fv) {
                *is_numeric = true;
                *num_count += 1;
                *num_sum += v;
                if v < *num_min {
                    *num_min = v;
                }
                if v > *num_max {
                    *num_max = v;
                }
            }

            // String length stats
            if let FieldValue::String(s) = fv {
                *is_string = true;
                let l = s.chars().count() as u64;
                if l < *str_min_len {
                    *str_min_len = l;
                }
                if l > *str_max_len {
                    *str_max_len = l;
                }
            }
        }
    }
}
