use std::io::Read;

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::{FieldValue, Record};
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;

use super::records::rebuild_executors;
use super::state::AppState;

// ── Response types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ProcessDocumentResponse {
    pub status: String,
    pub source_file: String,
    pub pages: usize,
    pub images_extracted: usize,
    pub tables_extracted: usize,
    pub records_inserted: usize,
    pub record_ids: Vec<String>,
    pub chunks: Vec<ChunkSummary>,
}

#[derive(Debug, Serialize)]
pub struct BatchProcessResponse {
    pub status: String,
    pub files_processed: usize,
    pub total_records_inserted: usize,
    pub results: Vec<ProcessDocumentResponse>,
}

#[derive(Debug, Serialize)]
pub struct ChunkSummary {
    pub id: String,
    #[serde(rename = "type")]
    pub chunk_type: String,
    pub preview: String,
}

#[derive(Debug, Serialize)]
pub struct OcrErrorResponse {
    pub error: OcrError,
}

#[derive(Debug, Serialize)]
pub struct OcrError {
    pub code: String,
    pub message: String,
    pub suggestion: String,
}

// ── Handler ─────────────────────────────────────────────────────────────

pub async fn process_document(
    State(state): State<AppState>,
    Path(name): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<ProcessDocumentResponse>), (StatusCode, Json<OcrErrorResponse>)> {
    // Check that the collection exists.
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "COLLECTION_NOT_FOUND".to_owned(),
                        message: format!("Collection '{}' not found", name),
                        suggestion: "Create the collection first.".to_owned(),
                    },
                }),
            )
        })?
    };

    // Resolve FlashParse endpoint.
    let endpoint = match &state.inner.flashparse_endpoint {
        Some(ep) => ep.clone(),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "OCR_SERVICE_UNAVAILABLE".to_owned(),
                        message: "Document processing requires FlashParse. Start Veda with: docker compose --profile ocr up".to_owned(),
                        suggestion: "See documentation for FlashParse setup: https://github.com/kshitij2605/Veda#document-processing".to_owned(),
                    },
                }),
            ));
        }
    };

    // Extract file from multipart upload.
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut skip_captions = false;
    let mut dpi: Option<u32> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_owned();
        match field_name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_owned());
                file_data = field.bytes().await.ok().map(|b| b.to_vec());
            }
            "skip_captions" => {
                if let Ok(bytes) = field.bytes().await {
                    let val = String::from_utf8_lossy(&bytes);
                    skip_captions = val == "true" || val == "1";
                }
            }
            "dpi" => {
                if let Ok(bytes) = field.bytes().await {
                    let val = String::from_utf8_lossy(&bytes);
                    dpi = val.parse().ok();
                }
            }
            _ => {}
        }
    }

    let file_data = match file_data {
        Some(d) if !d.is_empty() => d,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "MISSING_FILE".to_owned(),
                        message: "No file provided in the multipart upload.".to_owned(),
                        suggestion: "Include a 'file' field in the multipart form data.".to_owned(),
                    },
                }),
            ));
        }
    };

    let fname = file_name.unwrap_or_else(|| "document".to_owned());

    let result = process_single_file(&state, &name, &schema, &endpoint, file_data, &fname, skip_captions, dpi).await?;

    Ok((StatusCode::CREATED, Json(result)))
}

/// Batch document processing: accepts multiple files and processes each through FlashParse.
pub async fn process_documents_batch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<BatchProcessResponse>), (StatusCode, Json<OcrErrorResponse>)> {
    // Check that the collection exists.
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "COLLECTION_NOT_FOUND".to_owned(),
                        message: format!("Collection '{}' not found", name),
                        suggestion: "Create the collection first.".to_owned(),
                    },
                }),
            )
        })?
    };

    let endpoint = match &state.inner.flashparse_endpoint {
        Some(ep) => ep.clone(),
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "OCR_SERVICE_UNAVAILABLE".to_owned(),
                        message: "Document processing requires FlashParse. Start Veda with: docker compose --profile ocr up".to_owned(),
                        suggestion: "See documentation for FlashParse setup: https://github.com/kshitij2605/Veda#document-processing".to_owned(),
                    },
                }),
            ));
        }
    };

    // Collect all files and options from multipart.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut skip_captions = false;
    let mut dpi: Option<u32> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_owned();
        match field_name.as_str() {
            "files" | "file" => {
                let fname = field.file_name().unwrap_or("document").to_owned();
                if let Ok(bytes) = field.bytes().await {
                    let data = bytes.to_vec();
                    if !data.is_empty() {
                        files.push((fname, data));
                    }
                }
            }
            "skip_captions" => {
                if let Ok(bytes) = field.bytes().await {
                    let val = String::from_utf8_lossy(&bytes);
                    skip_captions = val == "true" || val == "1";
                }
            }
            "dpi" => {
                if let Ok(bytes) = field.bytes().await {
                    let val = String::from_utf8_lossy(&bytes);
                    dpi = val.parse().ok();
                }
            }
            _ => {}
        }
    }

    if files.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "MISSING_FILE".to_owned(),
                    message: "No files provided in the multipart upload.".to_owned(),
                    suggestion: "Include one or more 'files' fields in the multipart form data.".to_owned(),
                },
            }),
        ));
    }

    let mut results = Vec::new();
    let mut total_inserted = 0usize;

    for (fname, data) in files {
        let result = process_single_file(&state, &name, &schema, &endpoint, data, &fname, skip_captions, dpi).await?;
        total_inserted += result.records_inserted;
        results.push(result);
    }

    Ok((
        StatusCode::CREATED,
        Json(BatchProcessResponse {
            status: "processed".to_owned(),
            files_processed: results.len(),
            total_records_inserted: total_inserted,
            results,
        }),
    ))
}

/// Process a single file through FlashParse and insert results into the collection.
async fn process_single_file(
    state: &AppState,
    collection_name: &str,
    schema: &veda_core::schema::CollectionSchema,
    flashparse_endpoint: &str,
    file_data: Vec<u8>,
    file_name: &str,
    skip_captions: bool,
    dpi: Option<u32>,
) -> Result<ProcessDocumentResponse, (StatusCode, Json<OcrErrorResponse>)> {
    // Forward file to FlashParse.
    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(file_data)
        .file_name(file_name.to_owned())
        .mime_str("application/octet-stream")
        .unwrap();
    let form = reqwest::multipart::Form::new().part("file", part);

    let form = if skip_captions {
        form.text("skip_captions", "true")
    } else {
        form
    };
    let form = if let Some(dpi_val) = dpi {
        form.text("dpi", dpi_val.to_string())
    } else {
        form
    };

    let url = format!("{}/api/v1/pdf/process", flashparse_endpoint);
    let resp = client.post(&url).multipart(form).send().await.map_err(|e| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "OCR_SERVICE_UNAVAILABLE".to_owned(),
                    message: format!("Failed to connect to FlashParse: {}", e),
                    suggestion: "Ensure FlashParse is running: docker compose --profile ocr up".to_owned(),
                },
            }),
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "OCR_PROCESSING_FAILED".to_owned(),
                    message: format!("FlashParse returned {}: {}", status, body),
                    suggestion: "Check FlashParse logs for details.".to_owned(),
                },
            }),
        ));
    }

    // Read ZIP response.
    let zip_bytes = resp.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "OCR_RESPONSE_ERROR".to_owned(),
                    message: format!("Failed to read FlashParse response: {}", e),
                    suggestion: "Check FlashParse logs for details.".to_owned(),
                },
            }),
        )
    })?;

    // Extract contents from ZIP.
    let cursor = std::io::Cursor::new(&zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "OCR_RESPONSE_ERROR".to_owned(),
                    message: format!("Invalid ZIP from FlashParse: {}", e),
                    suggestion: "Check FlashParse logs for details.".to_owned(),
                },
            }),
        )
    })?;

    let mut markdown_content = String::new();
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();
    let mut _metadata_json: Option<String> = None;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "OCR_RESPONSE_ERROR".to_owned(),
                        message: format!("Failed to read ZIP entry: {}", e),
                        suggestion: "Check FlashParse logs for details.".to_owned(),
                    },
                }),
            )
        })?;

        let entry_name = file.name().to_owned();
        if entry_name.ends_with(".md") || entry_name.ends_with(".markdown") {
            let mut content = String::new();
            file.read_to_string(&mut content).ok();
            if !markdown_content.is_empty() {
                markdown_content.push_str("\n\n---\n\n");
            }
            markdown_content.push_str(&content);
        } else if entry_name.ends_with(".json") && entry_name.contains("metadata") {
            let mut content = String::new();
            file.read_to_string(&mut content).ok();
            _metadata_json = Some(content);
        } else if entry_name.ends_with(".png")
            || entry_name.ends_with(".jpg")
            || entry_name.ends_with(".jpeg")
            || entry_name.ends_with(".gif")
            || entry_name.ends_with(".webp")
        {
            let mut data = Vec::new();
            file.read_to_end(&mut data).ok();
            if !data.is_empty() {
                images.push((entry_name, data));
            }
        }
    }

    // Chunk markdown by headings or page separators.
    let chunks: Vec<String> = if markdown_content.is_empty() {
        Vec::new()
    } else {
        markdown_content
            .split("\n---\n")
            .flat_map(|section| {
                let parts: Vec<&str> = section.split("\n## ").collect();
                if parts.len() > 1 {
                    parts
                } else {
                    section.split("\n# ").collect()
                }
            })
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_owned())
            .collect()
    };

    // Count tables (simple heuristic: lines with | pipes).
    let tables_extracted = markdown_content
        .lines()
        .filter(|l| l.starts_with('|') && l.ends_with('|') && l.contains("---"))
        .count();

    // Build Record objects and insert them into the collection.
    let shard_count = schema.shard_count();
    let has_content_field = schema.get_field("content").is_some();
    let has_source_file_field = schema.get_field("source_file").is_some();
    let has_chunk_index_field = schema.get_field("chunk_index").is_some();
    let has_image_blob_field = schema.get_field("image_blob").is_some();

    let mut inserted_ids: Vec<String> = Vec::new();
    let mut response_chunks: Vec<ChunkSummary> = Vec::new();

    let ocr_err = |msg: String| -> (StatusCode, Json<OcrErrorResponse>) {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "INSERT_FAILED".to_owned(),
                    message: msg,
                    suggestion: "Check collection schema and server logs.".to_owned(),
                },
            }),
        )
    };

    // Discover vector fields and find embedding models for auto-embedding.
    // For each vector field in the schema, try to find a matching model in the registry.
    let registry = state.inner.embedding_registry.read().await;
    let mut text_embed_targets: Vec<(String, EmbeddingClient)> = Vec::new();
    let mut image_embed_targets: Vec<(String, EmbeddingClient)> = Vec::new();

    for field_def in schema.fields() {
        if let FieldType::Vector { .. } = field_def.field_type() {
            // Try to find a model that supports the appropriate modality.
            // Convention: fields named "*_embedding" auto-embed from the corresponding content.
            let field_name = field_def.name();

            // Check for text embedding models.
            let text_models = registry.find_by_modality(Modality::Text);
            if !text_models.is_empty() && (field_name.contains("text") || field_name.contains("content")) {
                text_embed_targets.push((
                    field_name.to_owned(),
                    EmbeddingClient::new(text_models[0].clone()),
                ));
            }

            // Check for image embedding models.
            let image_models = registry.find_by_modality(Modality::Image);
            if !image_models.is_empty() && field_name.contains("image") {
                image_embed_targets.push((
                    field_name.to_owned(),
                    EmbeddingClient::new(image_models[0].clone()),
                ));
            }
        }
    }
    drop(registry);

    // Insert text chunks.
    {
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(collection_name).ok_or_else(|| {
            ocr_err("no segment managers for collection".into())
        })?;

        for (idx, chunk_text) in chunks.iter().enumerate() {
            let mut record = Record::new();
            if has_content_field {
                record.set("content", FieldValue::String(chunk_text.clone()));
            }
            if has_source_file_field {
                record.set("source_file", FieldValue::String(file_name.to_owned()));
            }
            if has_chunk_index_field {
                record.set("chunk_index", FieldValue::Int32(idx as i32));
            }

            // Embed text content into vector fields.
            for (target_field, client) in &text_embed_targets {
                if let Ok(vector) = client.embed_text(chunk_text) {
                    record.set(target_field, FieldValue::Vector(DenseVector::new(vector)));
                }
            }

            let record_id = record.id().to_string();
            let shard_id = record.id().as_u128() as u32 % shard_count;

            let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
                ocr_err(format!("shard {shard_id} not found"))
            })?;
            sm.insert(record).map_err(|e| ocr_err(format!("insert failed: {e}")))?;

            let preview = if chunk_text.len() > 100 {
                format!("{}...", &chunk_text[..100])
            } else {
                chunk_text.clone()
            };
            response_chunks.push(ChunkSummary {
                id: record_id.clone(),
                chunk_type: "text".to_owned(),
                preview: preview.trim().to_owned(),
            });
            inserted_ids.push(record_id);
        }

        // Insert image records.
        for (img_name, img_data) in &images {
            let mut record = Record::new();
            if has_image_blob_field {
                let blob_key = format!("{}/{}/{}", collection_name, record.id(), img_name);
                record.set("image_blob", FieldValue::BlobRef(blob_key));
            }
            if has_content_field {
                record.set("content", FieldValue::String(format!("[image: {}] ({} bytes)", img_name, img_data.len())));
            }
            if has_source_file_field {
                record.set("source_file", FieldValue::String(file_name.to_owned()));
            }

            // Embed image data into vector fields.
            for (target_field, client) in &image_embed_targets {
                if let Ok(vector) = client.embed_bytes(img_data, Modality::Image) {
                    record.set(target_field, FieldValue::Vector(DenseVector::new(vector)));
                }
            }

            let record_id = record.id().to_string();
            let shard_id = record.id().as_u128() as u32 % shard_count;

            let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
                ocr_err(format!("shard {shard_id} not found"))
            })?;
            sm.insert(record).map_err(|e| ocr_err(format!("insert failed: {e}")))?;

            response_chunks.push(ChunkSummary {
                id: record_id.clone(),
                chunk_type: "image".to_owned(),
                preview: img_name.clone(),
            });
            inserted_ids.push(record_id);
        }

        // Seal growing segments.
        for sm in shard_map.values_mut() {
            sm.seal_growing().map_err(|e| ocr_err(format!("failed to seal segment: {e}")))?;
        }
    }

    // Rebuild query executors.
    rebuild_executors(state, collection_name).await.map_err(|(status, json_err)| {
        (
            status,
            Json(OcrErrorResponse {
                error: OcrError {
                    code: "REBUILD_FAILED".to_owned(),
                    message: json_err.error.clone(),
                    suggestion: "Check server logs.".to_owned(),
                },
            }),
        )
    })?;

    let pages = chunks.len().max(1);

    Ok(ProcessDocumentResponse {
        status: "processed".to_owned(),
        source_file: file_name.to_owned(),
        pages,
        images_extracted: images.len(),
        tables_extracted,
        records_inserted: inserted_ids.len(),
        record_ids: inserted_ids,
        chunks: response_chunks,
    })
}
