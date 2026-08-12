//! `POST /v1/collections/{name}/ingest-media` — ingest a single media file
//! (image, audio, or video) into a collection.
//!
//! This generalizes the image-insert loop of `process-document` over all
//! non-text modalities: the file's MIME type selects the modality, the
//! collection's matching `auto_embed` vector field selects the embedding
//! model, and the omni multimodal model embeds image/audio/video into ONE
//! shared vector space — so a single collection can hold all three.
//!
//! Flow: store the raw bytes as a blob -> resolve the vector field whose
//! `auto_embed` modality matches the file's modality -> embed the bytes ->
//! build one record carrying the vector plus any [`MediaMeta`] fields the
//! schema declares -> insert -> seal -> rebuild query executors.
//!
//! Tenant/RLS-scoped like `process-document` and `/ingest-url`.

use axum::Json;
use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use serde::Serialize;
use tracing::instrument;
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;
use veda_core::types::{AutoEmbedModality, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::blob_store::{ArcBlobStore, BlobStore};

use super::records::rebuild_executors;
use super::state::AppState;
use crate::tenant::{self, Tenant};

/// Provenance/description metadata stamped onto the media record. Only the
/// fields the collection schema actually declares are written; the rest are
/// silently skipped (same "drop keys not in schema" behaviour as `/insert`).
#[derive(Debug, Clone, Default)]
pub struct MediaMeta {
    /// Value for a `source_path` string field (the upload's origin path).
    pub source_path: Option<String>,
    /// Value for a `filename` string field.
    pub filename: Option<String>,
    /// Value for a `class` string field (caller-supplied label).
    pub class: Option<String>,
    /// Value for a `duration_secs` float field (audio/video length).
    pub duration_secs: Option<f64>,
    /// Value for a `content` string field (human-readable description).
    pub content: String,
}

/// Outcome of ingesting one media file.
#[derive(Debug, Serialize)]
pub struct MediaIngestResult {
    pub records_inserted: usize,
    pub vector_dim: usize,
    pub modality: String,
    pub blob_key: Option<String>,
}

/// HTTP response body for the `/ingest-media` route.
#[derive(Debug, Serialize)]
pub struct IngestMediaResponse {
    pub status: String,
    pub records_inserted: usize,
    pub vector_dim: usize,
    pub modality: String,
    pub blob_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MediaErrorResponse {
    pub error: MediaError,
}

#[derive(Debug, Serialize)]
pub struct MediaError {
    pub code: String,
    pub message: String,
}

fn err(status: StatusCode, code: &str, message: String) -> (StatusCode, Json<MediaErrorResponse>) {
    (
        status,
        Json(MediaErrorResponse {
            error: MediaError {
                code: code.to_owned(),
                message,
            },
        }),
    )
}

/// Map a `veda-core` [`Modality`] to the schema-side [`AutoEmbedModality`]
/// used by vector fields. Text/Pdf have no media-ingest representation here.
fn auto_embed_modality_of(modality: Modality) -> Option<AutoEmbedModality> {
    match modality {
        Modality::Image => Some(AutoEmbedModality::Image),
        Modality::Audio => Some(AutoEmbedModality::Audio),
        Modality::Video => Some(AutoEmbedModality::Video),
        Modality::Text | Modality::Pdf => None,
    }
}

/// Ingest a single media file into `collection_name`.
///
/// `collection_name` must already be tenant-scoped by the caller; `schema`
/// is that collection's schema. The `mime` selects the modality (via
/// [`Modality::from_mime`]); a non-media MIME (text/pdf/unknown) is a clean
/// 400. On error returns `(status, message)`; callers wrap the message into
/// their own error envelope.
pub async fn ingest_media(
    state: &AppState,
    collection_name: &str,
    schema: &CollectionSchema,
    bytes: &[u8],
    mime: &str,
    meta: MediaMeta,
) -> Result<MediaIngestResult, (StatusCode, String)> {
    // Infer modality from the MIME type. Only image/audio/video are media
    // here — text/pdf/unknown are rejected.
    let modality = Modality::from_mime(mime).filter(|m| auto_embed_modality_of(*m).is_some());
    let modality = modality.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported media MIME '{mime}': expected an image/*, audio/*, or video/* type"
            ),
        )
    })?;
    let want_modality = auto_embed_modality_of(modality).expect("checked above");

    // Resolve the vector field whose auto_embed modality matches, and an
    // embedding client for the model registered for that modality (the omni
    // multimodal model, since image/audio/video share one vector space).
    let embed_target: Option<(String, EmbeddingClient)> = {
        let registry = state.inner.embedding_registry.read().await;
        let mut found = None;
        for field_def in schema.fields() {
            if let FieldType::Vector {
                auto_embed: Some(cfg),
                ..
            } = field_def.field_type()
                && cfg.modality == want_modality
            {
                let models = registry.find_by_modality(modality);
                if let Some(model) = models.first() {
                    found = Some((
                        field_def.name().to_owned(),
                        EmbeddingClient::new((*model).clone()),
                    ));
                    break;
                }
            }
        }
        found
    };

    let (target_field, client) = embed_target.ok_or_else(|| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "no {modality} auto_embed vector field with a registered model in collection '{}'",
                tenant::display_name(collection_name)
            ),
        )
    })?;

    // Every dense vector field is a dense column in the segment, so a record
    // must carry a value for each one. This upload only fills `target_field`,
    // so a collection declaring a second vector field can never accept it:
    // the storage layer rejects the insert with a bare "record missing
    // required vector field '<other>'" surfaced as INGEST_FAILED, which says
    // nothing about what to change. Fail early and name the fix.
    let unfillable: Vec<&str> = schema
        .fields()
        .iter()
        .filter(|f| matches!(f.field_type(), FieldType::Vector { .. }))
        .map(|f| f.name())
        .filter(|n| *n != target_field)
        .collect();
    if !unfillable.is_empty() {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "collection '{}' declares dense vector field(s) [{}] that a {modality} upload \
                 cannot fill, and every record must carry a value for every dense vector field. \
                 Give a media collection exactly one auto_embed vector field — use one \
                 collection per modality.",
                tenant::display_name(collection_name),
                unfillable.join(", ")
            ),
        ));
    }

    // Schema field presence — only stamp metadata the schema declares.
    let has_source_path = schema.get_field("source_path").is_some();
    let has_filename = schema.get_field("filename").is_some();
    let has_class = schema.get_field("class").is_some();
    let has_duration = schema.get_field("duration_secs").is_some();
    let has_content = schema.get_field("content").is_some();
    let has_blob_field = matches!(
        schema.get_field("media_blob").map(|f| f.field_type()),
        Some(&FieldType::Blob)
    );

    let shard_count = schema.shard_count();

    // Embed the media bytes. A failure here is fatal — without the vector the
    // record would not be searchable.
    let vector = client.embed_bytes(bytes, modality).await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("media embedding failed: {e}"),
        )
    })?;
    let vector_dim = vector.len();

    let blob_key = {
        let mut record = Record::new();
        record.set(&target_field, FieldValue::Vector(DenseVector::new(vector)));

        // Store the raw bytes as a blob keyed by collection/record/filename.
        let fname = meta.filename.as_deref().unwrap_or("media");
        let blob_key = format!("{}/{}/{}", collection_name, record.id(), fname);
        {
            let store = ArcBlobStore(state.inner.blob_store.read().await.clone());
            store.put(&blob_key, bytes).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to store media blob: {e}"),
                )
            })?;
        }
        if has_blob_field {
            record.set("media_blob", FieldValue::BlobRef(blob_key.clone()));
        }

        if has_source_path && let Some(sp) = &meta.source_path {
            record.set("source_path", FieldValue::String(sp.clone()));
        }
        if has_filename && let Some(fname) = &meta.filename {
            record.set("filename", FieldValue::String(fname.clone()));
        }
        if has_class && let Some(class) = &meta.class {
            record.set("class", FieldValue::String(class.clone()));
        }
        if has_duration && let Some(dur) = meta.duration_secs {
            record.set("duration_secs", FieldValue::Float64(dur));
        }
        if has_content {
            record.set("content", FieldValue::String(meta.content.clone()));
        }

        // Media records carry a placeholder like "[media: clip.mp4]" and an
        // image/audio embedding, so text-phrased screening is not interpretable.
        // Stamp `_indexed_at` only → reported as `skipped`, never as clean.
        crate::rest::ingest_common::stamp_guardrail_fields(
            &mut record,
            &crate::agent::guardrails::ChunkVerdict::skipped(),
            crate::rest::ingest_common::now_secs(),
        );

        let shard_id =
            veda_coordinator::shard_router::ShardRouter::assign_shard(&record.id(), shard_count);

        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(collection_name).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "no segment managers for collection".to_owned(),
            )
        })?;
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

        // Seal growing segments so the record is queryable after rebuild.
        for sm in shard_map.values_mut() {
            sm.seal_growing().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to seal segment: {e}"),
                )
            })?;
        }
        blob_key
    };

    rebuild_executors(state, collection_name)
        .await
        .map_err(|(status, json_err)| (status, json_err.error.clone()))?;

    Ok(MediaIngestResult {
        records_inserted: 1,
        vector_dim,
        modality: modality.to_string(),
        blob_key: Some(blob_key),
    })
}

/// `POST /v1/collections/{name}/ingest-media` handler. Multipart upload
/// mirroring `process-document`: a required `file` part plus optional
/// `class`, `source_path`, and `content` text fields.
#[instrument(name = "ingest_media", skip(state, multipart), fields(collection = %name))]
pub async fn ingest_media_route(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<IngestMediaResponse>), (StatusCode, Json<MediaErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    // Resolve the alias like `/insert` does — otherwise the media record is
    // written into the abandoned base collection while reads follow the alias.
    let name = state.inner.aliases.resolve(&name).await;

    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                "COLLECTION_NOT_FOUND",
                format!("Collection '{}' not found", tenant::display_name(&name)),
            )
        })?
    };

    // Extract the file and optional metadata fields from the multipart upload.
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut content_type: Option<String> = None;
    let mut class: Option<String> = None;
    let mut source_path: Option<String> = None;
    let mut content: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_owned();
        match field_name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_owned());
                content_type = field.content_type().map(|s| s.to_owned());
                file_data = field.bytes().await.ok().map(|b| b.to_vec());
            }
            "class" => {
                if let Ok(bytes) = field.bytes().await {
                    class = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            "source_path" => {
                if let Ok(bytes) = field.bytes().await {
                    source_path = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            "content" => {
                if let Ok(bytes) = field.bytes().await {
                    content = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            _ => {}
        }
    }

    let file_data = match file_data {
        Some(d) if !d.is_empty() => d,
        _ => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                "MISSING_FILE",
                "No file provided in the multipart upload. Include a 'file' field.".to_owned(),
            ));
        }
    };

    // Prefer the part's declared content-type; fall back to inferring from
    // the filename extension.
    let mime = content_type
        .filter(|s| !s.is_empty() && s != "application/octet-stream")
        .or_else(|| file_name.as_deref().and_then(mime_from_filename))
        .ok_or_else(|| {
            err(
                StatusCode::BAD_REQUEST,
                "UNKNOWN_MIME",
                "Could not determine the media type. Send a typed 'file' part or a filename with a known extension.".to_owned(),
            )
        })?;

    let fname = file_name.unwrap_or_else(|| "media".to_owned());
    let meta = MediaMeta {
        source_path,
        filename: Some(fname.clone()),
        class,
        duration_secs: None,
        content: content.unwrap_or_else(|| format!("[media: {fname}]")),
    };

    let result = ingest_media(&state, &name, &schema, &file_data, &mime, meta)
        .await
        .map_err(|(status, message)| {
            let code = if status == StatusCode::BAD_REQUEST {
                "UNSUPPORTED_MEDIA"
            } else if status == StatusCode::UNPROCESSABLE_ENTITY {
                "NO_MEDIA_FIELD"
            } else if status == StatusCode::CONFLICT {
                "INCOMPATIBLE_SCHEMA"
            } else {
                "INGEST_FAILED"
            };
            err(status, code, message)
        })?;

    Ok((
        StatusCode::CREATED,
        Json(IngestMediaResponse {
            status: "ingested".to_owned(),
            records_inserted: result.records_inserted,
            vector_dim: result.vector_dim,
            modality: result.modality,
            blob_key: result.blob_key,
        }),
    ))
}

/// Best-effort MIME inference from a filename extension, used only when the
/// multipart part carries no usable content-type.
fn mime_from_filename(name: &str) -> Option<String> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    let mime = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        _ => return None,
    };
    Some(mime.to_owned())
}
