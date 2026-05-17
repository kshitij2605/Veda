"""Thin Veda REST client used by rag_anything when BACKEND=veda.

Mirrors only the surface that rag_anything actually exercises:
- create_collection (idempotent — POST 409 == already exists)
- bulk_insert (POST /v1/collections/{name}/insert)
- search (POST /v1/collections/{name}/search, dense-only)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import requests


class VedaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        collection: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VEDA_BASE_URL") or "http://localhost:6340").rstrip("/")
        self.collection = collection or os.environ.get("VEDA_COLLECTION") or "rag_anything"
        self.timeout = timeout
        self._session = requests.Session()

    # ── lifecycle ────────────────────────────────────────────────────────

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/v1/admin/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def create_collection_if_missing(self) -> bool:
        """Idempotent collection create. Returns True if created, False if already existed."""
        get = self._session.get(f"{self.base_url}/v1/collections/{self.collection}", timeout=self.timeout)
        if get.status_code == 200:
            return False
        schema = {
            "name": self.collection,
            "fields": [
                {"name": "content", "type": "string"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "dimension": 1024,
                    "metric": "cosine",
                    "model_id": "BAAI/bge-m3",
                    "mrl_dims": [768, 512],
                },
                {"name": "parent_doc_id", "type": "string", "filterable": True},
                {"name": "chunk_id", "type": "int64", "filterable": True},
                {"name": "page_number", "type": "int64", "filterable": True},
            ],
            "shard_count": 2,
        }
        r = self._session.post(f"{self.base_url}/v1/collections", json=schema, timeout=self.timeout)
        if r.status_code in (200, 201):
            return True
        if r.status_code == 409:
            return False
        r.raise_for_status()
        return False

    # ── ingest ───────────────────────────────────────────────────────────

    def bulk_insert(self, records: List[Dict[str, Any]], batch_size: int = 200) -> Dict[str, Any]:
        """POST /v1/collections/{name}/insert in batches.

        Each record must already be shaped as {id, fields: {...}}.
        Returns aggregate {inserted, failed, errors}.
        """
        url = f"{self.base_url}/v1/collections/{self.collection}/insert"
        total_inserted = 0
        total_failed = 0
        errors: List[Dict[str, Any]] = []
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            try:
                r = self._session.post(url, json={"records": batch}, timeout=self.timeout)
                if r.status_code >= 400:
                    total_failed += len(batch)
                    errors.append({"status": r.status_code, "body": r.text[:500]})
                    continue
                body = r.json()
                total_inserted += int(body.get("inserted", len(batch)))
            except Exception as e:  # noqa: BLE001
                total_failed += len(batch)
                errors.append({"exception": str(e)})
        return {"inserted": total_inserted, "failed": total_failed, "errors": errors[:5]}

    # ── search ───────────────────────────────────────────────────────────

    def dense_search(
        self,
        query_vector: List[float],
        *,
        k: int = 10,
        filter_expr: Optional[str] = None,
        truncate_to: Optional[int] = None,
        explain: bool = False,
        rerank: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query": {"vector": {"field": "embedding", "values": query_vector}},
            "limit": k,
        }
        if filter_expr:
            body["filter"] = filter_expr
        if truncate_to is not None:
            body["truncate_to"] = truncate_to
        if explain:
            body["explain"] = True
        if rerank is not None:
            body["rerank"] = rerank
        url = f"{self.base_url}/v1/collections/{self.collection}/search"
        t0 = time.perf_counter()
        r = self._session.post(url, json=body, timeout=self.timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        out = r.json()
        out["_client_elapsed_ms"] = elapsed_ms
        return out
