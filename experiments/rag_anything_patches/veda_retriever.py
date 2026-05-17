"""Drop-in replacement for ElasticsearchRetriever when BACKEND=veda.

Mirrors the methods rag_anything's api/utils.py::search calls into:
- set_index(name)              — no-op alias for set_collection
- hybrid_search_linear(...)    — dense-only against Veda (no BM25 fusion)

rag_anything's chunks do not produce sparse vectors, so we cannot run a true
hybrid (BM25 + kNN) against Veda the way Elasticsearch does. The honest
comparison here is "ES hybrid 0.7 BM25 + 0.3 kNN" vs "Veda dense kNN only".
This is reported as-is in the comparison; sparse SPLADE ingest is logged as a
follow-up.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from veda_client import VedaClient
from embedder import get_embeddings


class VedaRetriever:
    def __init__(self, base_url: str, collection: str) -> None:
        self.client = VedaClient(base_url=base_url, collection=collection)
        self.index = collection  # alias to keep the rag_anything API surface

    def set_index(self, index: str) -> None:
        self.index = index
        self.client.collection = index

    async def hybrid_search_linear(
        self,
        query: str,
        filter: Optional[List[Dict]] = None,
        k: int = 10,
        fetch_k: int = 100,
        vector_query_field: str = "embedding",
        text_field: str = "content",
        retriever_weights: Optional[List[float]] = None,
        normalizer: str = "minmax",
        embed_query: Optional[str] = None,
        load_more: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        # Dense-only path. retriever_weights and normalizer are accepted but
        # ignored — Veda kNN is unweighted here.
        embed_text = embed_query if embed_query else query
        vector = await get_embeddings(embed_text)

        # The Veda call is synchronous (requests); run in a thread to keep the
        # caller's asyncio loop unblocked.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.dense_search(
                query_vector=vector,
                k=k,
                filter_expr=None,  # rag_anything's `filter` is ES DSL — not portable
                explain=False,
            ),
        )

        hits = result.get("results", []) or []
        docs: List[Dict[str, Any]] = []

        def _unwrap(field_value: Any) -> Any:
            # Veda serializes field values as e.g. {"String": "..."} or
            # {"Int64": 7}. Unwrap the single-key tag so downstream callers
            # see plain Python values.
            if isinstance(field_value, dict) and len(field_value) == 1:
                return next(iter(field_value.values()))
            return field_value

        for hit in hits[:k]:
            raw_fields = hit.get("fields") or {}
            fields = {k_: _unwrap(v) for k_, v in raw_fields.items()}
            text = fields.get(text_field, fields.get("content", ""))
            # Drop empty-content hits — these surface from the populate-
            # fields fallback when shard routing returns the same record
            # twice (one shard hit, one orphan). The downstream reranker
            # rejects empty strings with HTTP 422.
            if not isinstance(text, str) or not text.strip():
                continue
            docs.append(
                {
                    "id": hit.get("id"),
                    "score": float(1.0 - float(hit.get("distance", 0.0))),  # cosine sim
                    text_field: text,
                    "metadata": {
                        "parent_doc_id": fields.get("parent_doc_id"),
                        "chunk_id": fields.get("chunk_id"),
                        "page_number": fields.get("page_number"),
                        "minio_metadata": fields.get("minio_metadata", {}),
                    },
                    "_veda_latency_ms": result.get("_client_elapsed_ms"),
                }
            )
        return docs

    # rag_anything's search() sometimes uses *_cc / *_rrf entrypoints from
    # other call sites — expose aliases so flipping BACKEND works without
    # editing every caller.
    async def hybrid_search_cc(self, *args, **kwargs):
        return await self.hybrid_search_linear(*args, **kwargs)

    async def hybrid_search_rrf(self, *args, **kwargs):
        return await self.hybrid_search_linear(*args, **kwargs)
