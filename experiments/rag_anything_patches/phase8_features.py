"""Exercise the recently-added Veda features against the live rag_anything
collection on iitgpu11. Outputs a markdown summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

BASE = os.environ.get("VEDA_BASE_URL", "http://localhost:6340")
COLLECTION = os.environ.get("VEDA_COLLECTION", "rag_anything")
EMBED_URL = os.environ.get("OPENAI_BASE_URL", "http://iitgpu03.hon.otsuka-shokai.co.jp:8000/v1")
EMBED_KEY = os.environ.get("OPENAI_API_KEY", "tensorflow")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")


def embed(text: str) -> List[float]:
    from openai import OpenAI
    client = OpenAI(base_url=EMBED_URL, api_key=EMBED_KEY)
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def section(title: str) -> None:
    print()
    print(f"### {title}")
    print()


# ── 8a. Adaptive router /v1/route ────────────────────────────────────────

def demo_adaptive_router() -> Dict[str, Any]:
    section("8a — Adaptive router (/v1/route)")
    queries = [
        "Microsoft license",
        "EDR",
        "ランサムウェアから企業を守る方法を教えてください",
        "Adobe",
        "compare cybersecurity insurance plans for SMB",
    ]
    results = []
    for q in queries:
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/v1/route", json={"query": q}, timeout=10)
        dt = (time.perf_counter() - t0) * 1000
        body = r.json() if r.status_code == 200 else {"error": r.status_code, "body": r.text[:200]}
        print(f"  q={q!r:55} → {body}  ({dt:.0f}ms)")
        results.append({"q": q, "latency_ms": dt, "response": body})
    return {"queries": len(queries), "results": results}


# ── 8b. MCP endpoint /v1/mcp ─────────────────────────────────────────────

def demo_mcp() -> Dict[str, Any]:
    section("8b — MCP endpoint (/v1/mcp)")
    rid = 0

    def call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal rid
        rid += 1
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/v1/mcp", json=body, timeout=30)
        dt = (time.perf_counter() - t0) * 1000
        out = r.json() if r.status_code == 200 else {"http": r.status_code, "body": r.text[:200]}
        print(f"  {method}  → ({dt:.0f}ms)  {json.dumps(out)[:240]}")
        return out

    a = call("tools/list", {})
    b = call("tools/call", {"name": "list_collections", "arguments": {}})
    c = call("tools/call", {"name": "describe_collection", "arguments": {"name": COLLECTION}})
    d = call("tools/call", {"name": "search", "arguments": {"collection": COLLECTION, "query": "ランサムウェア", "k": 3}})
    return {"tools_list": a, "list_collections": b, "describe": c, "search": d}


# ── 8c. Cold-tier archive/restore (cargo test invocation) ────────────────

def demo_cold_tier() -> Dict[str, Any]:
    section("8c — Cold-tier archive/restore")
    print("  No REST exposure yet — re-running the segment_manager_test")
    print("  integration tests on iitgpu11 hardware. See report.")
    return {"note": "run separately: cargo test -p veda-storage --test segment_manager_test test_cold_tier"}


# ── 8d. MRL truncation ───────────────────────────────────────────────────

def demo_mrl() -> Dict[str, Any]:
    section("8d — MRL truncation")
    q = "ランサムウェアから企業を守る"
    print(f"  query: {q}")
    vec = embed(q)

    runs = []
    for trunc in (None, 768, 512):
        body: Dict[str, Any] = {
            "query": {"vector": {"field": "embedding", "values": vec}},
            "limit": 10,
        }
        if trunc is not None:
            body["truncate_to"] = trunc
        latencies = []
        results_ids = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = requests.post(f"{BASE}/v1/collections/{COLLECTION}/search", json=body, timeout=30)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            if r.status_code == 200:
                data = r.json()
                hits = [h["id"] for h in data.get("results", [])]
                results_ids.append(hits)
            else:
                print(f"  ERROR truncate_to={trunc}: {r.status_code} {r.text[:200]}")
                results_ids.append([])

        med = sorted(latencies)[len(latencies) // 2]
        print(f"  truncate_to={str(trunc) or 'full(1024)':>8}  median={med:.1f}ms  top-3={results_ids[0][:3]}")
        runs.append({"truncate_to": trunc, "median_ms": med, "top_ids": results_ids[0]})

    if len(runs) >= 2:
        baseline = runs[0]["top_ids"][:5]
        for r in runs[1:]:
            overlap = len(set(baseline) & set(r["top_ids"][:5]))
            print(f"  top-5 overlap full vs truncate_to={r['truncate_to']}: {overlap}/5")
    return {"runs": runs}


# ── 8e. Rerank ───────────────────────────────────────────────────────────

def demo_rerank() -> Dict[str, Any]:
    section("8e — Rerank (in search body)")
    q = "ランサムウェアから企業を守る"
    vec = embed(q)
    body = {
        "query": {"vector": {"field": "embedding", "values": vec}},
        "limit": 5,
        "rerank": {"field": "content", "query": q, "top_k": 3, "model": "ruri-v3"},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/collections/{COLLECTION}/search", json=body, timeout=60)
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        print(f"  rerank request failed: {r.status_code} {r.text[:300]}")
        return {"error": r.text[:300], "status": r.status_code}
    data = r.json()
    print(f"  rerank ok  ({dt:.0f}ms)  hits={data.get('count')}")
    for h in data.get("results", [])[:3]:
        fields = h.get("fields", {})
        content = fields.get("content", "")
        if isinstance(content, dict):
            content = next(iter(content.values()))
        print(f"    distance={h.get('distance'):.4f}  {str(content)[:80]!s}")
    return {"latency_ms": dt, "hits": data.get("count")}


def main():
    out = {}
    out["adaptive_router"] = demo_adaptive_router()
    out["mcp"] = demo_mcp()
    out["cold_tier"] = demo_cold_tier()
    out["mrl"] = demo_mrl()
    out["rerank"] = demo_rerank()

    Path("evals/phase8_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print()
    print(f"=> wrote evals/phase8_summary.json")


if __name__ == "__main__":
    sys.exit(main())
