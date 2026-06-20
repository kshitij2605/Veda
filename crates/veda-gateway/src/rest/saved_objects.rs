//! Phase 6.5 follow-up — server-side persistence for SavedObjects.
//!
//! The Niriksha dashboard has had a "SavedObject" abstraction since
//! Phase 0 (searches / atlas views / eval runs / pipelines /
//! dashboards). Until now it has been backed by `localStorage`, which
//! is good enough for one browser but breaks the moment a user opens
//! the dashboard on a second machine or wants to share a saved view
//! with a teammate.
//!
//! This module is a thin CRUD layer that mirrors the frontend store's
//! shape one-for-one and persists everything to a single JSON file at
//! `{data_dir}/_saved_objects.json`. Concurrent writes are serialised
//! through a `RwLock`. Reads are lock-free clones.
//!
//! Endpoints:
//!   GET    /v1/saved_objects                 → { objects: [...] }
//!   PUT    /v1/saved_objects/{id}            → upsert (body: SavedObject)
//!   DELETE /v1/saved_objects/{id}            → 204 (idempotent)
//!
//! The frontend store keeps localStorage as a fallback when these
//! endpoints aren't reachable, so the cutover is non-disruptive: when
//! the gateway is up, the store writes through to the server *and* to
//! localStorage; when offline, it falls back to localStorage alone.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::collections::ErrorResponse;
use super::state::AppState;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SavedObject {
    pub id: String,
    /// Free-form type tag — "search" / "atlas" / "eval" / "eval_run" /
    /// "console_query" / "pipeline" / "dashboard". The gateway does
    /// not interpret it; clients do.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    /// Object-shape body. The gateway treats this as opaque JSON.
    pub body: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<i64>,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub objects: Vec<SavedObject>,
}

/// Shared in-process store. Wrapped in Arc<RwLock<>> so the lock is
/// independent of AppStateInner's other RwLocks.
#[derive(Clone, Default)]
pub struct SavedObjectStore {
    inner: Arc<RwLock<HashMap<String, SavedObject>>>,
    path: PathBuf,
}

impl SavedObjectStore {
    pub fn new(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("_saved_objects.json");
        let inner: HashMap<String, SavedObject> = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "saved_objects: parse failed, starting empty"
                );
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
            path,
        }
    }

    /// Insert or replace an object by its id, stamping timestamps the
    /// same way the HTTP `upsert` handler does (server-authoritative
    /// `updated_at`; preserve `created_at` on an existing row, else
    /// set it now). Returns the stored object. Used by in-process
    /// features (e.g. Veda Agents) that persist through the same store
    /// rather than the REST surface.
    pub async fn put(&self, mut obj: SavedObject) -> SavedObject {
        let now = now_ms();
        {
            let mut m = self.inner.write().await;
            if let Some(prev) = m.get(&obj.id) {
                if obj.created_at == 0 {
                    obj.created_at = prev.created_at;
                }
            } else if obj.created_at == 0 {
                obj.created_at = now;
            }
            obj.updated_at = now;
            m.insert(obj.id.clone(), obj.clone());
        }
        self.write_through().await;
        obj
    }

    /// Fetch a single object by id.
    pub async fn get(&self, id: &str) -> Option<SavedObject> {
        self.inner.read().await.get(id).cloned()
    }

    /// All objects whose `type` tag equals `kind`, newest-first
    /// (same ordering as `snapshot`).
    pub async fn list_by_type(&self, kind: &str) -> Vec<SavedObject> {
        self.snapshot()
            .await
            .into_iter()
            .filter(|o| o.kind == kind)
            .collect()
    }

    /// Remove an object by id (idempotent). Returns whether a row was
    /// removed.
    pub async fn remove(&self, id: &str) -> bool {
        let removed = {
            let mut m = self.inner.write().await;
            m.remove(id).is_some()
        };
        self.write_through().await;
        removed
    }

    async fn snapshot(&self) -> Vec<SavedObject> {
        let m = self.inner.read().await;
        let mut out: Vec<SavedObject> = m.values().cloned().collect();
        out.sort_by(|a, b| {
            let la = a.last_used_at.unwrap_or(a.updated_at);
            let lb = b.last_used_at.unwrap_or(b.updated_at);
            lb.cmp(&la)
        });
        out
    }

    async fn write_through(&self) {
        let path = self.path.clone();
        let snapshot: HashMap<String, SavedObject> = self.inner.read().await.clone();
        // Persist on a blocking thread; failure is logged but never
        // surfaced to the client — the in-memory copy already
        // reflects the upsert.
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_vec_pretty(&snapshot) {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&path, bytes) {
                        tracing::warn!(error = %e, path = %path.display(), "saved_objects: write failed");
                    }
                }
                Err(e) => tracing::warn!(error = %e, "saved_objects: serialize failed"),
            }
        });
    }
}

pub async fn list(State(state): State<AppState>) -> Json<ListResponse> {
    let store = state.inner.saved_objects.clone();
    Json(ListResponse {
        objects: store.snapshot().await,
    })
}

pub async fn upsert(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut obj): Json<SavedObject>,
) -> Result<Json<SavedObject>, (StatusCode, Json<ErrorResponse>)> {
    if id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "id must not be empty".into(),
            }),
        ));
    }
    // Server is authoritative on id and timestamps; let the caller
    // propose name / body / type / owner.
    obj.id = id.clone();
    let now = now_ms();
    let store = state.inner.saved_objects.clone();
    {
        let mut m = store.inner.write().await;
        if let Some(prev) = m.get(&id) {
            if obj.created_at == 0 {
                obj.created_at = prev.created_at;
            }
        } else if obj.created_at == 0 {
            obj.created_at = now;
        }
        obj.updated_at = now;
        m.insert(id, obj.clone());
    }
    store.write_through().await;
    Ok(Json(obj))
}

pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> StatusCode {
    let store = state.inner.saved_objects.clone();
    {
        let mut m = store.inner.write().await;
        m.remove(&id);
    }
    store.write_through().await;
    StatusCode::NO_CONTENT
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
