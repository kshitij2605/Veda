//! Shared "ingest a markdown document into a collection" pipeline.
//!
//! Both the OCR document path (`process-document`) and the web-ingestion
//! path (`/ingest-url`) need the exact same chunk -> embed -> insert flow:
//! split markdown into chunks via the configured chunker, auto-embed the
//! chunk text into the collection's text vector fields (and SPLADE sparse
//! fields), build one record per chunk carrying `content` + provenance
//! metadata, insert, seal, and rebuild query executors.
//!
//! [`ingest_markdown`] factors that flow out so neither caller duplicates
//! it. The OCR path keeps its image-specific handling separately and calls
//! this helper only for the text chunks.

use axum::http::StatusCode;
use veda_core::chunking::{ChunkingStrategy, chunk_markdown};
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;

use super::records::rebuild_executors;
use super::state::AppState;

/// Provenance metadata stamped onto every chunk record produced from a
/// single markdown document. Only the fields the collection schema
/// actually declares are written; the rest are silently skipped (same
/// "drop keys not in schema" behaviour as `/insert`).
#[derive(Debug, Clone, Default)]
pub struct SourceMeta {
    /// Value for a `source_file` string field (OCR path: the file name).
    pub source_file: Option<String>,
    /// Value for a `source_url` string field (web path: the crawled URL).
    pub source_url: Option<String>,
    /// Value for a `title` string field (web path: the page title).
    pub title: Option<String>,
    /// When set, stamp the reserved `_expires_at` TTL field (unix seconds).
    pub expires_at: Option<i64>,
}

/// Outcome of ingesting one markdown document.
#[derive(Debug, Default)]
pub struct IngestResult {
    /// Number of chunk records inserted.
    pub chunks_inserted: usize,
    /// `(record_id, chunk_text)` for each inserted chunk, in order. The OCR
    /// caller uses the text to build response previews; the web caller
    /// ignores it.
    pub chunks: Vec<(String, String)>,
}

/// Reserved TTL field name (unix seconds); records past it are swept at
/// query time. Mirrors the convention in `ingest_logs.rs` / `records.rs`.
const EXPIRES_AT_FIELD: &str = "_expires_at";

/// Chunk `markdown`, auto-embed each chunk, and insert one record per
/// chunk into `collection_name`. Seals growing segments and rebuilds the
/// collection's query executors before returning.
///
/// On error returns `(status, message)`; callers wrap the message into
/// their own error envelope.
pub async fn ingest_markdown(
    state: &AppState,
    collection_name: &str,
    schema: &CollectionSchema,
    markdown: &str,
    source_meta: &SourceMeta,
    chunker: &ChunkingStrategy,
) -> Result<IngestResult, (StatusCode, String)> {
    let text_chunks = chunk_markdown(markdown, chunker);

    let shard_count = schema.shard_count();
    let has_content_field = schema.get_field("content").is_some();
    let has_source_file_field = schema.get_field("source_file").is_some();
    let has_source_url_field = schema.get_field("source_url").is_some();
    let has_title_field = schema.get_field("title").is_some();
    let has_chunk_index_field = schema.get_field("chunk_index").is_some();
    // Needed for `/reindex`'s rechunk-by-parent grouping (default field
    // name "parent_doc_id", see `ReindexRequest.parent_id_field`) — without
    // this, every chunk from every document is an "orphan" to the reindex
    // path and gets copied as-is, silently making chunker-strategy changes
    // a no-op for any real ingested document (found via live E2E testing).
    let has_parent_doc_id_field = schema.get_field("parent_doc_id").is_some();

    // Discover text vector fields and SPLADE sparse-text targets, mirroring
    // the document path. Image fields are not handled here — the OCR caller
    // does its own image insertion.
    let registry = state.inner.embedding_registry.read().await;
    let multimodal_models = registry.find_by_modality(Modality::Image);
    let text_only_models = registry.find_by_modality(Modality::Text);

    let mut text_embed_targets: Vec<(String, EmbeddingClient)> = Vec::new();
    let mut sparse_text_targets: Vec<(String, String)> = Vec::new(); // (target_field, source_field)

    for field_def in schema.fields() {
        if let FieldType::Vector { auto_embed, .. } = field_def.field_type() {
            let field_name = field_def.name();

            if let Some(cfg) = auto_embed {
                use veda_core::types::AutoEmbedModality;
                // Only text vector fields are fed here; image/video/audio
                // fields have no text source on the markdown path.
                if matches!(cfg.modality, AutoEmbedModality::Text) {
                    let models = registry.find_by_modality(Modality::Text);
                    if !models.is_empty() {
                        text_embed_targets.push((
                            field_name.to_owned(),
                            EmbeddingClient::new(models[0].clone()),
                        ));
                    }
                }
                continue;
            }

            let lower = field_name.to_lowercase();
            let is_image_field = lower.contains("image") && !lower.contains("text");
            if !is_image_field {
                let models = if text_only_models.is_empty() {
                    &multimodal_models
                } else {
                    &text_only_models
                };
                if !models.is_empty() {
                    text_embed_targets.push((
                        field_name.to_owned(),
                        EmbeddingClient::new(models[0].clone()),
                    ));
                }
            }
        }
        if let FieldType::SparseVector {
            auto_embed: Some(cfg),
            ..
        } = field_def.field_type()
        {
            use veda_core::types::AutoEmbedModality;
            if matches!(cfg.modality, AutoEmbedModality::Text) {
                sparse_text_targets.push((field_def.name().to_owned(), cfg.source.clone()));
            }
        }
    }
    drop(registry);

    let mut chunks_out: Vec<(String, String)> = Vec::new();

    {
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(collection_name).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "no segment managers for collection".to_owned(),
            )
        })?;

        for chunk in &text_chunks {
            let mut record = Record::new();
            if has_content_field {
                record.set("content", FieldValue::String(chunk.text.clone()));
            }
            if has_source_file_field && let Some(sf) = &source_meta.source_file {
                record.set("source_file", FieldValue::String(sf.clone()));
            }
            if has_source_url_field && let Some(su) = &source_meta.source_url {
                record.set("source_url", FieldValue::String(su.clone()));
            }
            if has_title_field && let Some(t) = &source_meta.title {
                record.set("title", FieldValue::String(t.clone()));
            }
            if has_parent_doc_id_field
                && let Some(pid) = source_meta
                    .source_file
                    .as_ref()
                    .or(source_meta.source_url.as_ref())
            {
                record.set("parent_doc_id", FieldValue::String(pid.clone()));
            }
            if has_chunk_index_field {
                record.set("chunk_index", FieldValue::Int32(chunk.index as i32));
            }
            if let Some(exp) = source_meta.expires_at {
                record.set(EXPIRES_AT_FIELD, FieldValue::Int64(exp));
            }

            for (target_field, client) in &text_embed_targets {
                match client.embed_text(&chunk.text).await {
                    Ok(vector) => {
                        record.set(target_field, FieldValue::Vector(DenseVector::new(vector)));
                    }
                    Err(e) => {
                        tracing::warn!(field = %target_field, error = %e, "text embedding failed, skipping vector field");
                    }
                }
            }

            if !sparse_text_targets.is_empty() {
                let splade_guard = state.inner.splade_client.read().await;
                if let Some(splade) = splade_guard.as_ref() {
                    for (target_field, source_field) in &sparse_text_targets {
                        let input = match record.get(source_field) {
                            Some(FieldValue::String(s)) => s.clone(),
                            _ => chunk.text.clone(),
                        };
                        match splade.encode(&input) {
                            Ok(enc) => {
                                if let Ok(sv) =
                                    veda_core::sparse::SparseVec::new(enc.indices, enc.values)
                                {
                                    record.set(target_field, FieldValue::SparseVector(sv));
                                }
                            }
                            Err(e) => {
                                tracing::warn!(field = %target_field, error = %e, "splade encoding failed, skipping sparse field");
                            }
                        }
                    }
                } else {
                    tracing::warn!(
                        target_count = sparse_text_targets.len(),
                        "sparse_vector field requested SPLADE auto_embed but VEDA_SPLADE_ENDPOINT is unset"
                    );
                }
            }

            let record_id = record.id().to_string();
            let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
                &record.id(),
                shard_count,
            );
            let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("shard {shard_id} not found"),
                )
            })?;
            sm.insert(record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("insert failed: {e}"),
                )
            })?;
            chunks_out.push((record_id, chunk.text.clone()));
        }

        for sm in shard_map.values_mut() {
            sm.seal_growing().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to seal segment: {e}"),
                )
            })?;
        }
    }

    rebuild_executors(state, collection_name)
        .await
        .map_err(|(status, json_err)| (status, json_err.error.clone()))?;

    Ok(IngestResult {
        chunks_inserted: chunks_out.len(),
        chunks: chunks_out,
    })
}
