//! Shared "ingest a markdown document into a collection" pipeline.
//!
//! Both the OCR document path (`process-document`) and the web-ingestion
//! path (`/ingest-url`) need the exact same chunk -> embed -> insert flow:
//! split markdown into chunks via the configured chunker, auto-embed the
//! chunk text into the collection's dense vector fields (and SPLADE sparse
//! fields), build one record per chunk carrying `content` + provenance
//! metadata, insert, seal, and rebuild query executors.
//!
//! [`ingest_markdown`] factors that flow out so neither caller duplicates
//! it. The OCR path keeps its image-record insertion separate and calls this
//! helper only for the text chunks — but both must fill the SAME dense
//! vector columns, since storage makes every one of them mandatory on every
//! record. [`resolve_dense_targets`] is the single place that decides which
//! column is filled by which embedder, shared by both loops.

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
#[derive(Debug, Clone)]
pub struct SourceMeta {
    /// Value for a `source_file` string field (OCR path: the file name).
    pub source_file: Option<String>,
    /// Value for a `source_url` string field (web path: the crawled URL).
    pub source_url: Option<String>,
    /// Value for a `title` string field (web path: the page title).
    pub title: Option<String>,
    /// When set, stamp the reserved `_expires_at` TTL field (unix seconds).
    pub expires_at: Option<i64>,
    /// The tenant this ingestion ran under — needed so the auto-
    /// optimization trigger (spawned after this function returns) can
    /// correctly scope its eval-set-generation and campaign calls.
    pub tenant: crate::tenant::TenantId,
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

/// One dense vector column, the embedder resolved for it, and which record
/// types that embedder can actually fill it from.
///
/// Storage treats every `FieldType::Vector` column as **mandatory**: a record
/// that omits one is rejected outright (`growing_segment.rs` / `segment.rs`:
/// "record missing required vector field"). So the two `process-document`
/// ingest loops — text chunks here, image records in `document.rs` — cannot
/// each own their own half of the schema; whichever record type is written
/// first fails. Both resolve their targets through [`resolve_dense_targets`]
/// instead, so both fill the *same* columns, and Veda's unified-vector-space
/// design point falls straight out of it: one omni model (`jina-v5-omni-small`
/// and friends support `Text` and `Image`) embeds chunk text and image bytes
/// into one column, one space.
///
/// The capability flags exist so an unsatisfiable schema — a column whose
/// model cannot accept one of the record types — is caught up front by
/// [`unfillable_columns`] instead of surfacing as the storage layer's message,
/// which names no fix.
pub struct DenseVectorTarget {
    /// Name of the dense vector column.
    pub field: String,
    /// The embedder resolved for the column's declared modality. `None` when
    /// no registered model declares it — the column cannot be filled at all.
    pub client: Option<EmbeddingClient>,
    /// The resolved model accepts text input.
    pub supports_text: bool,
    /// The resolved model accepts image input.
    pub supports_image: bool,
    /// The schema declares this column for a non-text modality — explicitly
    /// via `auto_embed`, or through the legacy `*image*` name heuristic. Such
    /// a column is a standing request for image records.
    pub declared_media: bool,
}

/// Resolve every dense vector column of `schema` to an embedder, using the
/// column's declared modality (`auto_embed.modality`, else the legacy
/// `*image*`-but-not-`*text*` name heuristic) to pick the model — first
/// registered model that declares that modality, unchanged from before.
///
/// What changed is that the result is shared: the caller decides which
/// columns it can fill from the capability flags rather than each ingest loop
/// applying its own modality filter and silently ignoring the other's
/// columns.
pub async fn resolve_dense_targets(
    state: &AppState,
    schema: &CollectionSchema,
) -> Vec<DenseVectorTarget> {
    use veda_core::types::AutoEmbedModality;

    let registry = state.inner.embedding_registry.read().await;
    let mut targets = Vec::new();
    for field_def in schema.fields() {
        let FieldType::Vector { auto_embed, .. } = field_def.field_type() else {
            continue;
        };
        let field_name = field_def.name();
        let declared = match auto_embed {
            Some(cfg) => match cfg.modality {
                AutoEmbedModality::Text => Modality::Text,
                AutoEmbedModality::Image => Modality::Image,
                AutoEmbedModality::Video => Modality::Video,
                AutoEmbedModality::Audio => Modality::Audio,
            },
            None => {
                let lower = field_name.to_lowercase();
                if lower.contains("image") && !lower.contains("text") {
                    Modality::Image
                } else {
                    Modality::Text
                }
            }
        };
        let model = registry
            .find_by_modality(declared)
            .first()
            .copied()
            .cloned();
        targets.push(DenseVectorTarget {
            field: field_name.to_owned(),
            supports_text: model
                .as_ref()
                .is_some_and(|m| m.supported_modalities.contains(&Modality::Text)),
            supports_image: model
                .as_ref()
                .is_some_and(|m| m.supported_modalities.contains(&Modality::Image)),
            declared_media: declared != Modality::Text,
            client: model.map(EmbeddingClient::new),
        });
    }
    targets
}

/// The columns a record of `modality` cannot fill — every one of which would
/// make the insert fail, because dense columns are mandatory.
pub fn unfillable_columns(targets: &[DenseVectorTarget], modality: Modality) -> Vec<&str> {
    targets
        .iter()
        .filter(|t| !match modality {
            Modality::Image => t.supports_image,
            _ => t.supports_text,
        })
        .map(|t| t.field.as_str())
        .collect()
}

/// The 409 message for a schema no ingest can satisfy. Named fields, named
/// fix — the same contract as `/ingest-media`'s `INCOMPATIBLE_SCHEMA` guard,
/// which exists for this identical invariant.
pub fn incompatible_schema_message(
    collection_name: &str,
    columns: &[&str],
    modality: Modality,
) -> String {
    let (record_kind, input) = match modality {
        Modality::Image => ("an extracted image record", "images"),
        _ => ("a text chunk", "text"),
    };
    format!(
        "collection '{}' declares dense vector field(s) [{}] that {record_kind} cannot fill: \
         no registered embedding model can embed {input} into them, and every record must carry \
         a value for every dense vector field. Back the collection's vector field with a unified \
         multimodal model so text chunks and images share one vector space, or use one collection \
         per modality.",
        crate::tenant::display_name(collection_name),
        columns.join(", ")
    )
}

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
    // Every read path (search, records, tuning) resolves the caller's
    // name through the alias map before touching segment_managers; this
    // write path must too, or a collection with a swapped alias (e.g.
    // after a reindex applies a trial) silently ingests into the
    // abandoned base collection while every read hits the alias target
    // -- found via live E2E testing: a process-document response reported
    // success but the record was never queryable.
    let resolved_name = state.inner.aliases.resolve(collection_name).await;
    let collection_name = resolved_name.as_str();

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

    // Every dense vector column, resolved once — a chunk record must fill
    // ALL of them, not just the ones this loop used to claim as "text"
    // fields, because storage rejects a record that omits any of them. The
    // OCR image loop resolves the same set the same way, so both record types
    // land in the same columns.
    let dense_targets = resolve_dense_targets(state, schema).await;
    let unfillable = unfillable_columns(&dense_targets, Modality::Text);
    if !unfillable.is_empty() {
        // Reject before writing anything: the storage-level "record missing
        // required vector field" an unsatisfiable schema would otherwise
        // produce names no fix, and by then half the document is inserted.
        return Err((
            StatusCode::CONFLICT,
            incompatible_schema_message(collection_name, &unfillable, Modality::Text),
        ));
    }

    let mut sparse_text_targets: Vec<(String, String)> = Vec::new(); // (target_field, source_field)
    for field_def in schema.fields() {
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

    // Screening uses the TEXT-modality embedding. A collection with no dense
    // vector column yields no candidate, so its chunks are skipped rather
    // than compared against text-phrased patterns.
    let primary_text_vector_field: String = dense_targets
        .first()
        .map(|t| t.field.clone())
        .unwrap_or_default();

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

            for target in &dense_targets {
                let Some(client) = target.client.as_ref() else {
                    continue;
                };
                match client.embed_text(&chunk.text).await {
                    Ok(vector) => {
                        record.set(&target.field, FieldValue::Vector(DenseVector::new(vector)));
                    }
                    Err(e) => {
                        tracing::warn!(field = %target.field, error = %e, "text embedding failed, skipping vector field");
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

            // Screen the chunk using the embedding just computed — no extra
            // embed call. Fails open: a skipped verdict stamps only
            // `_indexed_at`.
            let verdict = match record.get(primary_text_vector_field.as_str()) {
                Some(veda_core::record::FieldValue::Vector(v)) => {
                    crate::agent::guardrails::screen_chunk(state, v.as_slice(), &chunk.text).await
                }
                _ => crate::agent::guardrails::ChunkVerdict::skipped(),
            };
            stamp_guardrail_fields(&mut record, &verdict, now_secs());

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

    // Fire-and-forget: never blocks or fails this ingest call. See
    // `auto_optimization::maybe_trigger_auto_pipeline`'s doc comment.
    {
        let state = state.clone();
        let collection_name = collection_name.to_owned();
        let tenant = source_meta.tenant.clone();
        tokio::spawn(async move {
            super::auto_optimization::maybe_trigger_auto_pipeline(state, tenant, collection_name)
                .await;
        });
    }

    Ok(IngestResult {
        chunks_inserted: chunks_out.len(),
        chunks: chunks_out,
    })
}

/// Stamp `_indexed_at` plus, when screening ran, the guardrail verdict.
///
/// A `skipped` verdict stamps ONLY `_indexed_at`: leaving
/// `_guardrail_flagged` absent is what makes the chunk report as
/// `unscreened` rather than as screened-clean.
pub fn stamp_guardrail_fields(
    record: &mut veda_core::record::Record,
    verdict: &crate::agent::guardrails::ChunkVerdict,
    now_secs: i64,
) {
    use veda_core::record::{
        FieldValue, GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD, GUARDRAIL_SCORE_FIELD,
        INDEXED_AT_FIELD,
    };

    record.set(INDEXED_AT_FIELD, FieldValue::Int64(now_secs));
    if !verdict.screened {
        return;
    }
    record.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(verdict.flagged));
    record.set(GUARDRAIL_SCORE_FIELD, FieldValue::Float64(verdict.score));
    // A re-verdict must not leave a stale category from a PREVIOUS stamp:
    // clear it first, then set it again only if this verdict carries one.
    // Otherwise a chunk that used to flag "pii" but now screens clean
    // (pattern set changed, or a re-scan recomputes it) would end up
    // `flagged = false, category = "pii"` — reads as "clean but still
    // tagged pii" to any downstream consumer.
    record.remove(GUARDRAIL_CATEGORY_FIELD);
    if let Some(cat) = verdict.category {
        let name = serde_json::to_value(cat)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        if !name.is_empty() {
            record.set(GUARDRAIL_CATEGORY_FIELD, FieldValue::String(name));
        }
    }
}

/// Unix epoch seconds, injected nowhere — call sites pass this in so tests
/// stay deterministic where they need to.
pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::guardrails::{ChunkVerdict, GuardrailCategory};
    use veda_core::record::{GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD, Record};

    /// M8: a record previously flagged `pii` that re-screens clean must not
    /// keep reporting `_guardrail_category = "pii"` alongside
    /// `_guardrail_flagged = false` — that combination reads as "clean but
    /// still tagged pii" to any downstream consumer.
    #[test]
    fn a_clean_re_verdict_clears_a_stale_category() {
        let mut record = Record::new();
        // Simulate a prior stamp that flagged this record as PII.
        let flagged = ChunkVerdict {
            flagged: true,
            category: Some(GuardrailCategory::Pii),
            score: 0.9,
            matched: Some("email".into()),
            screened: true,
        };
        stamp_guardrail_fields(&mut record, &flagged, 1_000);
        assert!(record.get(GUARDRAIL_CATEGORY_FIELD).is_some());

        // Re-stamp with a clean verdict (e.g. a re-scan after the pattern
        // set changed, or content that no longer matches).
        stamp_guardrail_fields(&mut record, &ChunkVerdict::clean(), 2_000);

        assert_eq!(
            record.get(GUARDRAIL_FLAGGED_FIELD),
            Some(&veda_core::record::FieldValue::Bool(false))
        );
        assert!(
            record.get(GUARDRAIL_CATEGORY_FIELD).is_none(),
            "a clean re-verdict must clear the stale category, not just flip flagged to false"
        );
    }

    /// A `skipped` verdict must stamp ONLY `_indexed_at` — never
    /// `_guardrail_flagged = false`. Absent means "unscreened"; a present
    /// `false` means "screened and clean". Conflating the two is the
    /// fail-open bug this reserved-field scheme exists to prevent.
    #[test]
    fn a_skipped_verdict_never_stamps_flagged_false() {
        let mut record = Record::new();
        stamp_guardrail_fields(&mut record, &ChunkVerdict::skipped(), 1_000);
        assert!(record.get(GUARDRAIL_FLAGGED_FIELD).is_none());
        assert!(record.get(GUARDRAIL_CATEGORY_FIELD).is_none());
    }
}
