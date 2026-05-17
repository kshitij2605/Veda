"""Dense vs BM25 vs Hybrid (dense + BM25) head-to-head against ES baseline.

Picks N queries from test_dataset_new.xlsx, calls each backend, captures
top-K parent doc IDs and latency. Writes a CSV.

Run from rag_anything-veda root:
    .venv/bin/python evals/hybrid_eval.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "api"))
sys.path.insert(0, str(HERE.parent / "bulk_module"))


def _embed(text: str) -> List[float]:
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "tensorflow"),
    )
    r = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"), input=text)
    return r.data[0].embedding


def _veda_search(base_url: str, collection: str, body: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/collections/{collection}/search", json=body, timeout=30)
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    out = r.json()
    out["_client_ms"] = dt
    return out


def _hit_parent(hit: Dict[str, Any]) -> str:
    f = hit.get("fields") or {}
    p = f.get("parent_doc_id")
    if isinstance(p, dict):
        p = next(iter(p.values()))
    return p or "?"


def _es_search(query: str, k: int) -> Dict[str, Any]:
    """Direct ES baseline using the existing rag_anything retriever path."""
    for mod in ("utils", "es_retriever", "veda_retriever", "veda_client"):
        sys.modules.pop(mod, None)
    os.environ["BACKEND"] = "elasticsearch"
    from utils import search as _search

    t0 = time.perf_counter()
    docs = asyncio.run(_search(query=query, filters="{}", k=k, n=k))
    dt = (time.perf_counter() - t0) * 1000
    parents = []
    for d in docs[:k]:
        doc_id = d.get("id") or ""
        if ":chunk_" in doc_id:
            parents.append(doc_id.split(":chunk_")[0])
        else:
            md = d.get("metadata") or {}
            parents.append(md.get("parent_doc_id") or "?")
    return {"_client_ms": dt, "parents": parents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="1turn,2turn")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--base-url", default=os.environ.get("VEDA_BASE_URL", "http://localhost:6340"))
    parser.add_argument("--collection", default="rag_anything")
    parser.add_argument("--out", default="evals/hybrid_eval_result.csv")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    args = parser.parse_args()

    load_dotenv()
    xlsx_path = HERE / "test_dataset_new.xlsx"
    sheets = [s.strip() for s in args.queries.split(",")]
    qs: List[str] = []
    for sheet in sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if "Q" in df.columns:
            qs.extend(df["Q"].dropna().tolist())
    qs = qs[: args.limit]
    print(f"Running {len(qs)} queries (k={args.k}, dense_weight={args.dense_weight})")

    rows: List[Dict[str, Any]] = []
    print()
    print(f"{'q':<3} {'ES':>8} {'Veda dense':>11} {'Veda bm25':>11} {'Veda hybrid':>13} {'overlap@1':>11}")
    print("-" * 72)
    for qi, q in enumerate(qs):
        try:
            es_r = _es_search(q, args.k)
            es_parents = es_r["parents"]
            es_ms = es_r["_client_ms"]
        except Exception as e:
            print(f"  q={qi} ES error: {e}")
            continue

        qv = _embed(q)

        # Dense only
        dense_r = _veda_search(
            args.base_url, args.collection,
            {"query": {"vector": {"field": "embedding", "values": qv}}, "limit": args.k},
        )
        dense_parents = [_hit_parent(h) for h in dense_r.get("results", [])]

        # BM25 only
        bm25_r = _veda_search(
            args.base_url, args.collection,
            {"query": {"bm25": {"field": "content", "query": q}}, "limit": args.k},
        )
        bm25_parents = [_hit_parent(h) for h in bm25_r.get("results", [])]

        # Hybrid dense + bm25, both fusion modes
        hybrid_rrf_r = _veda_search(
            args.base_url, args.collection,
            {
                "query": {
                    "hybrid": {
                        "dense": {"field": "embedding", "values": qv},
                        "bm25": {"field": "content", "query": q},
                        "dense_weight": args.dense_weight,
                        "fusion": "rrf",
                    }
                },
                "limit": args.k,
            },
        )
        hybrid_rrf_parents = [_hit_parent(h) for h in hybrid_rrf_r.get("results", [])]
        hybrid_ws_r = _veda_search(
            args.base_url, args.collection,
            {
                "query": {
                    "hybrid": {
                        "dense": {"field": "embedding", "values": qv},
                        "bm25": {"field": "content", "query": q},
                        "dense_weight": args.dense_weight,
                        "fusion": "weighted_sum",
                    }
                },
                "limit": args.k,
            },
        )
        hybrid_ws_parents = [_hit_parent(h) for h in hybrid_ws_r.get("results", [])]

        # Top-1 overlap with ES (per leg).
        def t1(es: List[str], v: List[str]) -> int:
            return 1 if (es and v and es[0] == v[0] and v[0] != "?") else 0

        ov_dense = t1(es_parents, dense_parents)
        ov_bm25 = t1(es_parents, bm25_parents)
        ov_hybrid_rrf = t1(es_parents, hybrid_rrf_parents)
        ov_hybrid_ws = t1(es_parents, hybrid_ws_parents)

        print(
            f"{qi:<3} {es_ms:>7.0f}ms "
            f"d{dense_r['_client_ms']:>4.0f} "
            f"b{bm25_r['_client_ms']:>4.0f} "
            f"rrf{hybrid_rrf_r['_client_ms']:>4.0f} "
            f"ws{hybrid_ws_r['_client_ms']:>4.0f} "
            f"ov d{ov_dense} b{ov_bm25} rrf{ov_hybrid_rrf} ws{ov_hybrid_ws}"
        )

        rows.append({
            "qi": qi, "query": q,
            "es_ms": es_ms,
            "veda_dense_ms": dense_r["_client_ms"],
            "veda_bm25_ms": bm25_r["_client_ms"],
            "veda_hybrid_rrf_ms": hybrid_rrf_r["_client_ms"],
            "veda_hybrid_ws_ms": hybrid_ws_r["_client_ms"],
            "es_top": "|".join(es_parents),
            "veda_dense_top": "|".join(dense_parents),
            "veda_bm25_top": "|".join(bm25_parents),
            "veda_hybrid_rrf_top": "|".join(hybrid_rrf_parents),
            "veda_hybrid_ws_top": "|".join(hybrid_ws_parents),
            "ov_dense_top1": ov_dense,
            "ov_bm25_top1": ov_bm25,
            "ov_hybrid_rrf_top1": ov_hybrid_rrf,
            "ov_hybrid_ws_top1": ov_hybrid_ws,
        })

    df = pd.DataFrame(rows)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False, encoding="utf-8")
    print()
    print(f"Wrote {out_p}")
    print()

    def pct(col):
        return f"{100.0 * df[col].sum() / len(df):.0f}%"

    def stats(col):
        vals = df[col].dropna()
        if vals.empty:
            return "n/a"
        return f"p50={vals.median():.0f}ms p95={vals.quantile(0.95):.0f}ms mean={vals.mean():.0f}ms"

    print("Latency:")
    print(f"  ES:                  {stats('es_ms')}")
    print(f"  Veda dense:          {stats('veda_dense_ms')}")
    print(f"  Veda BM25:           {stats('veda_bm25_ms')}")
    print(f"  Veda hybrid (rrf):   {stats('veda_hybrid_rrf_ms')}")
    print(f"  Veda hybrid (ws):    {stats('veda_hybrid_ws_ms')}")
    print()
    print("Top-1 overlap with ES:")
    print(f"  Veda dense:          {pct('ov_dense_top1')}")
    print(f"  Veda BM25:           {pct('ov_bm25_top1')}")
    print(f"  Veda hybrid (rrf):   {pct('ov_hybrid_rrf_top1')}")
    print(f"  Veda hybrid (ws):    {pct('ov_hybrid_ws_top1')}")


if __name__ == "__main__":
    main()
