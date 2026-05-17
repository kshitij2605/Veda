"""Smoke comparison: same query set against ES and Veda.

Reads queries from test_dataset_new.xlsx, calls each backend, captures
per-query latency and the top-K parent_doc_ids retrieved. Writes a CSV.

Run from rag_anything-veda root:
    .venv/bin/python evals/smoke_eval.py --queries 1turn --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "api"))
sys.path.insert(0, str(HERE.parent / "bulk_module"))


async def run_backend(backend: str, queries: List[str], k: int) -> List[Dict[str, Any]]:
    os.environ["BACKEND"] = backend
    # Force a fresh import of utils to pick up the new BACKEND.
    for mod in ("utils", "es_retriever", "veda_retriever", "veda_client"):
        sys.modules.pop(mod, None)
    from utils import search  # noqa: WPS433

    rows = []
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        try:
            docs = await search(query=q, filters="{}", k=k, n=k)
        except Exception as e:  # noqa: BLE001
            print(f"  [{backend} q={qi}] ERROR: {e}")
            rows.append({"backend": backend, "qi": qi, "query": q, "latency_ms": None, "error": str(e), "doc_ids": ""})
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        parent_ids = []
        for d in docs[:k]:
            md = d.get("metadata") or {}
            pid = md.get("parent_doc_id") or d.get("parent_doc_id")
            if not pid:
                # ES path: the chunk _id is "{parent_doc_id}:chunk_{N}".
                doc_id = d.get("id") or ""
                if ":chunk_" in doc_id:
                    pid = doc_id.split(":chunk_")[0]
            parent_ids.append(pid or "?")
        rows.append({
            "backend": backend,
            "qi": qi,
            "query": q,
            "latency_ms": round(latency_ms, 2),
            "doc_ids": "|".join(parent_ids),
            "n_hits": len(docs),
        })
        print(f"  [{backend} q={qi}] {latency_ms:.0f}ms  top: {parent_ids[:3]}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="1turn", help="comma-separated sheet names")
    parser.add_argument("--limit", type=int, default=10, help="number of queries to run")
    parser.add_argument("--k", type=int, default=10, help="top-k retrieval")
    parser.add_argument("--backends", default="elasticsearch,veda")
    parser.add_argument("--out", default="evals/smoke_eval_result.csv")
    args = parser.parse_args()

    load_dotenv()

    xlsx_path = HERE / "test_dataset_new.xlsx"
    sheets = [s.strip() for s in args.queries.split(",")]
    qs: List[str] = []
    for sheet in sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if "Q" not in df.columns:
            print(f"Sheet {sheet}: no 'Q' column, columns={list(df.columns)}")
            continue
        qs.extend(df["Q"].dropna().tolist())
    qs = qs[: args.limit]
    print(f"Running {len(qs)} queries against {args.backends}")

    all_rows: List[Dict[str, Any]] = []
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend in backends:
        print(f"\n=== {backend} ===")
        rows = asyncio.run(run_backend(backend, qs, args.k))
        all_rows.extend(rows)

        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        if latencies:
            print(f"  p50={statistics.median(latencies):.0f}ms  "
                  f"mean={statistics.mean(latencies):.0f}ms  "
                  f"max={max(latencies):.0f}ms  n={len(latencies)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\nWrote {out}")

    # Side-by-side comparison summary
    if len(backends) == 2:
        a, b = backends
        df = pd.DataFrame(all_rows)
        for side in (a, b):
            sub = df[df["backend"] == side]
            lat = sub["latency_ms"].dropna()
            if not lat.empty:
                print(f"  {side:>14}: p50={lat.median():.1f}ms  p95={lat.quantile(0.95):.1f}ms  mean={lat.mean():.1f}ms")

        # Top-1 overlap
        a_top1 = df[df["backend"] == a].sort_values("qi")["doc_ids"].str.split("|").str[0].tolist()
        b_top1 = df[df["backend"] == b].sort_values("qi")["doc_ids"].str.split("|").str[0].tolist()
        if len(a_top1) == len(b_top1):
            same = sum(1 for x, y in zip(a_top1, b_top1) if x == y and x != "?")
            print(f"  top-1 overlap: {same}/{len(a_top1)} ({100*same/max(1,len(a_top1)):.0f}%)")


if __name__ == "__main__":
    main()
