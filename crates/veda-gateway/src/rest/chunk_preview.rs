//! Phase 5.5 — chunker A/B preview.
//!
//! `POST /v1/chunk/preview` runs the chunking layer against a chunk of
//! markdown text under N strategies and returns the resulting chunks
//! WITHOUT touching storage or embedding. The dashboard uses this to
//! diff chunk boundaries / size distributions side-by-side before
//! committing to a strategy for a real ingest.

use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use veda_core::chunking::{chunk_markdown, TextChunk};

use super::collections::ErrorResponse;
use super::document::parse_chunking_strategy;

const MAX_CHUNKS_RETURNED: usize = 50;

#[derive(Deserialize)]
pub struct StrategySpec {
    /// Human label shown in the UI. Optional — defaults to the
    /// strategy name. Useful when comparing two configs of the same
    /// strategy (e.g. "2000c/300o" vs "800c/100o").
    pub label: Option<String>,
    pub strategy_name: String,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
}

#[derive(Deserialize)]
pub struct ChunkPreviewRequest {
    pub text: String,
    pub runs: Vec<StrategySpec>,
}

#[derive(Serialize)]
pub struct ChunkPreview {
    pub index: usize,
    pub text: String,
    pub char_count: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    pub heading_path: Vec<String>,
    pub page_number: Option<usize>,
}

#[derive(Serialize)]
pub struct RunOutput {
    pub label: String,
    pub strategy_name: String,
    pub elapsed_ms: f64,
    pub total_chunks: usize,
    pub total_chars: usize,
    pub avg_chars: f64,
    pub min_chars: usize,
    pub max_chars: usize,
    /// True when more than MAX_CHUNKS_RETURNED chunks were produced and
    /// only the prefix is returned. UI shows a "showing first N of M"
    /// notice.
    pub truncated: bool,
    pub chunks: Vec<ChunkPreview>,
}

#[derive(Serialize)]
pub struct ChunkPreviewResponse {
    pub input_chars: usize,
    pub runs: Vec<RunOutput>,
}

pub async fn preview(
    Json(req): Json<ChunkPreviewRequest>,
) -> Result<Json<ChunkPreviewResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text must not be empty".into(),
            }),
        ));
    }
    if req.runs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "at least one run is required".into(),
            }),
        ));
    }
    if req.runs.len() > 6 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "at most 6 runs per request".into(),
            }),
        ));
    }

    let input_chars = req.text.chars().count();
    let mut runs = Vec::with_capacity(req.runs.len());
    for spec in req.runs {
        let strategy = parse_chunking_strategy(
            Some(&spec.strategy_name),
            spec.chunk_size,
            spec.chunk_overlap,
        );
        let label = spec
            .label
            .unwrap_or_else(|| spec.strategy_name.clone());

        let start = Instant::now();
        let chunks: Vec<TextChunk> = chunk_markdown(&req.text, &strategy);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let total_chunks = chunks.len();
        let total_chars: usize = chunks.iter().map(|c| c.text.chars().count()).sum();
        let lens: Vec<usize> = chunks.iter().map(|c| c.text.chars().count()).collect();
        let min_chars = lens.iter().min().copied().unwrap_or(0);
        let max_chars = lens.iter().max().copied().unwrap_or(0);
        let avg_chars = if total_chunks > 0 {
            total_chars as f64 / total_chunks as f64
        } else {
            0.0
        };

        let truncated = total_chunks > MAX_CHUNKS_RETURNED;
        let returned: Vec<ChunkPreview> = chunks
            .into_iter()
            .take(MAX_CHUNKS_RETURNED)
            .map(|c| ChunkPreview {
                index: c.index,
                char_count: c.text.chars().count(),
                text: c.text,
                start_offset: c.start_offset,
                end_offset: c.end_offset,
                heading_path: c.heading_path,
                page_number: c.page_number,
            })
            .collect();

        runs.push(RunOutput {
            label,
            strategy_name: spec.strategy_name,
            elapsed_ms,
            total_chunks,
            total_chars,
            avg_chars,
            min_chars,
            max_chars,
            truncated,
            chunks: returned,
        });
    }

    Ok(Json(ChunkPreviewResponse { input_chars, runs }))
}
