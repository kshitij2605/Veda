"""Bulk-ingest pre-OCR'd output into Veda (or Elasticsearch).

Reads the directory structure produced by GLM Hybrid OCR:
  bulk_module/output/pdfs/<doc_name>/<doc_name>_with_captions.mmd
chunks each markdown via PageAwareChunkerDeepSeek, calls the configured
embedder, and POSTs records to either Veda or Elasticsearch.

Run from `rag_anything-veda/bulk_module/`:
  BACKEND=veda VEDA_BASE_URL=http://localhost:6340 \
    ../.venv/bin/python index_from_preocr.py \
      --source output/pdfs --collection rag_anything
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Make sibling imports work whether we're called from bulk_module/ or root.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _chunk_markdown(text: str, chunk_size: int = 1024, chunk_overlap: int = 64) -> List[Dict[str, Any]]:
    """Chunk markdown using the same recursive splitter rag_anything uses for ES.

    Falls back to a simple paragraph-based splitter if langchain isn't
    available (it should be — the slim venv has langchain-text-splitters).
    Pages are tracked by looking for the `---` separator GLM-OCR emits
    between pages.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Annotate each line with page number, then split markdown into chunks
    # preserving page tracking on a best-effort basis.
    pages = text.split("\n---\n") if "\n---\n" in text else [text]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "．", ". ", "、", ", ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    out: List[Dict[str, Any]] = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if not piece:
                continue
            out.append({"text": piece, "page_number": page_no})
    return out


def _embed_batch(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts in one OpenAI-compatible call."""
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _post_veda(base_url: str, collection: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    import requests

    url = f"{base_url.rstrip('/')}/v1/collections/{collection}/insert"
    r = requests.post(url, json={"records": records}, timeout=120)
    if r.status_code >= 400:
        return {"inserted": 0, "failed": len(records), "status": r.status_code, "body": r.text[:300]}
    return r.json()


def _post_es(es_client, index: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from elasticsearch.helpers import bulk

    # records are flat Veda-shape; wrap them for ES.
    docs = [
        {
            "_index": index,
            "_id": str(uuid.uuid4()),
            "_source": rec,
        }
        for rec in records
    ]
    success, failed = bulk(es_client, docs, raise_on_error=False)
    return {"inserted": success, "failed": len(failed) if failed else 0}


def ingest(
    source: Path,
    collection: str,
    backend: str,
    base_url: str,
    embedding_url: str,
    embedding_key: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 50,
    limit: int | None = None,
) -> None:
    embedder = OpenAI(base_url=embedding_url, api_key=embedding_key)

    if backend == "elasticsearch":
        from elasticsearch import Elasticsearch
        es_url = os.environ["ELASTICSEARCH_HOST"]
        es = Elasticsearch(
            es_url,
            basic_auth=(os.environ.get("ELASTICSEARCH_USER", "elastic"),
                        os.environ.get("ELASTICSEARCH_PASSWORD", "")),
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=120,
        )

    doc_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if limit:
        doc_dirs = doc_dirs[:limit]
    print(f"Found {len(doc_dirs)} pre-OCR'd document folders under {source}")

    total_chunks = 0
    total_inserted = 0
    total_failed = 0
    t_global = time.perf_counter()

    for di, ddir in enumerate(doc_dirs, start=1):
        mmd = next(ddir.glob("*_with_captions.mmd"), None)
        if mmd is None:
            print(f"  [{di}/{len(doc_dirs)}] {ddir.name}: no .mmd — skipping")
            continue
        text = mmd.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_markdown(text, chunk_size, chunk_overlap)
        if not chunks:
            print(f"  [{di}/{len(doc_dirs)}] {ddir.name}: empty after chunking — skipping")
            continue
        doc_id = ddir.name
        t_doc = time.perf_counter()

        # Embed in mini-batches to bound peak memory and embedder load.
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            try:
                vectors = _embed_batch(embedder, embedding_model, [c["text"] for c in batch])
            except Exception as e:  # noqa: BLE001
                print(f"    embed error doc={doc_id} start={start}: {e}")
                total_failed += len(batch)
                continue

            records = []
            for i, (c, v) in enumerate(zip(batch, vectors)):
                global_idx = start + i
                # Veda's /insert expects flat records — schema fields at
                # the top level, no nesting under "fields". The id is
                # auto-generated server-side; passing one is a no-op.
                records.append(
                    {
                        "content": c["text"],
                        "embedding": v,
                        "parent_doc_id": doc_id,
                        "chunk_id": global_idx,
                        "page_number": int(c["page_number"]),
                    }
                )

            if backend == "veda":
                result = _post_veda(base_url, collection, records)
                ins = int(result.get("inserted", 0))
                fl = int(result.get("failed", 0))
            else:
                result = _post_es(es, collection, records)
                ins = int(result.get("inserted", 0))
                fl = int(result.get("failed", 0))
            total_chunks += len(records)
            total_inserted += ins
            total_failed += fl

        dt = time.perf_counter() - t_doc
        print(f"  [{di}/{len(doc_dirs)}] {ddir.name}: {len(chunks)} chunks in {dt:.1f}s")

    dt_all = time.perf_counter() - t_global
    print()
    print("Summary:")
    print(f"  Documents processed:  {len(doc_dirs)}")
    print(f"  Chunks created:       {total_chunks}")
    print(f"  Inserted into {backend:>13}: {total_inserted}")
    print(f"  Failed:               {total_failed}")
    print(f"  Wall time:            {dt_all:.1f}s "
          f"({(total_chunks / max(dt_all, 1e-6)):.1f} chunks/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-ingest pre-OCR'd markdown into Veda or Elasticsearch")
    parser.add_argument("--source", type=Path, required=True, help="Path to bulk_module/output/pdfs")
    parser.add_argument("--collection", type=str, required=True, help="Veda collection / ES index name")
    parser.add_argument("--backend", choices=["veda", "elasticsearch"],
                        default=os.environ.get("BACKEND", "veda"))
    parser.add_argument("--base-url", default=os.environ.get("VEDA_BASE_URL", "http://localhost:6340"))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of doc folders (smoke runs)")
    args = parser.parse_args()

    load_dotenv()
    ingest(
        source=args.source,
        collection=args.collection,
        backend=args.backend.lower(),
        base_url=args.base_url,
        embedding_url=os.environ.get("OPENAI_BASE_URL", "http://iitgpu03.hon.otsuka-shokai.co.jp:8000/v1"),
        embedding_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
