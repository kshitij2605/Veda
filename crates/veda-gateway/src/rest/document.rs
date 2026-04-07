use std::io::Read;

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use uuid::Uuid;

use super::state::AppState;

// ── Response types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ProcessDocumentResponse {
    pub status: String,
    pub pages: usize,
    pub images_extracted: usize,
    pub tables_extracted: usize,
    pub records_inserted: usize,
    pub chunks: Vec<ChunkSummary>,
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
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(OcrErrorResponse {
                    error: OcrError {
                        code: "COLLECTION_NOT_FOUND".to_owned(),
                        message: format!("Collection '{}' not found", name),
                        suggestion: "Create the collection first.".to_owned(),
                    },
                }),
            ));
        }
    }

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

    // Forward file to FlashParse.
    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(file_data)
        .file_name(fname)
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

    let url = format!("{}/api/v1/pdf/process", endpoint);
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

        let name_owned = file.name().to_owned();
        if name_owned.ends_with(".md") || name_owned.ends_with(".markdown") {
            let mut content = String::new();
            file.read_to_string(&mut content).ok();
            if !markdown_content.is_empty() {
                markdown_content.push_str("\n\n---\n\n");
            }
            markdown_content.push_str(&content);
        } else if name_owned.ends_with(".json") && name_owned.contains("metadata") {
            let mut content = String::new();
            file.read_to_string(&mut content).ok();
            _metadata_json = Some(content);
        } else if name_owned.ends_with(".png")
            || name_owned.ends_with(".jpg")
            || name_owned.ends_with(".jpeg")
            || name_owned.ends_with(".gif")
            || name_owned.ends_with(".webp")
        {
            let mut data = Vec::new();
            file.read_to_end(&mut data).ok();
            if !data.is_empty() {
                images.push((name_owned, data));
            }
        }
    }

    // Chunk markdown by headings or page separators.
    let chunks: Vec<&str> = if markdown_content.is_empty() {
        Vec::new()
    } else {
        markdown_content
            .split("\n---\n")
            .flat_map(|section| {
                // Further split by headings (## or #).
                let parts: Vec<&str> = section.split("\n## ").collect();
                if parts.len() > 1 {
                    parts
                } else {
                    section.split("\n# ").collect()
                }
            })
            .filter(|s| !s.trim().is_empty())
            .collect()
    };

    // Count tables (simple heuristic: lines with | pipes).
    let tables_extracted = markdown_content
        .lines()
        .filter(|l| l.starts_with('|') && l.ends_with('|') && l.contains("---"))
        .count();

    // Build response chunks.
    let mut response_chunks = Vec::new();
    for chunk in &chunks {
        let id = Uuid::new_v4().to_string();
        let preview = if chunk.len() > 100 {
            format!("{}...", &chunk[..100])
        } else {
            chunk.to_string()
        };
        response_chunks.push(ChunkSummary {
            id,
            chunk_type: "text".to_owned(),
            preview: preview.trim().to_owned(),
        });
    }

    for (img_name, _) in &images {
        let id = Uuid::new_v4().to_string();
        response_chunks.push(ChunkSummary {
            id,
            chunk_type: "image".to_owned(),
            preview: img_name.clone(),
        });
    }

    let total_records = chunks.len() + images.len();
    let pages = chunks.len().max(1);

    Ok((
        StatusCode::OK,
        Json(ProcessDocumentResponse {
            status: "processed".to_owned(),
            pages,
            images_extracted: images.len(),
            tables_extracted,
            records_inserted: total_records,
            chunks: response_chunks,
        }),
    ))
}
