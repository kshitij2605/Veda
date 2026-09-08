#!/usr/bin/env python3
"""Bisect probe for #72: which commit closed the negative-answer gap.

Differences from stag_v2.py, all deliberate:

  1. The agent gets `filter_query` in `tool_ids`, not just `search`. Without it
     the embedding-strip commits (05661e7 / 214d5db) are invisible: a `search`
     hit carries content/doc/_guardrail_* and no vector, so `strip_embeddings`
     is a no-op on that path. This is the note's explicit precondition.

  2. Negative cases run REPEATS_NEG times (default 10), not 3. The historical
     figure is "3/6", i.e. six samples -- far too few to decide a bisect step.
     Twenty samples separates a ~50% arm from a ~100% arm decisively.

  3. Every probe re-runs the CONTROL cases as a validity gate. The corpus was
     re-embedded with bge-m3 rather than the original jina model, so retrieval
     quality is not assumed. If controls fail, the run says INVALID rather than
     reporting a negative rate that reflects a broken corpus.

Usage:
    stag_bisect.py ingest    # create collection + agent, ingest 2012 docs
    stag_bisect.py probe     # run controls + negatives, print a verdict line
"""
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/mac/test")
import stag_v2  # noqa: E402  (FACTS / CASES / filler live there)

BASE = "http://localhost:6357"
COLL = "stagv2"
AGENT_NAME = "Bisect probe #72"
FILLER = 2000
REPEATS_NEG = 10
REPEATS_CTL = 2


def post(path, body, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw or "{}")
        except Exception:
            return e.code, {"raw": raw[:300]}
    except Exception as e:
        return 0, {"err": str(e)}


def get(path, timeout=60):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except Exception as e:
        return 0, {"err": str(e)}


def ensure_agent():
    """Create the agent once; reuse it by name on later runs.

    tool_ids carries filter_query as well as search -- see the module docstring.
    """
    code, body = get("/v1/agents")
    agents = body.get("agents", body) if isinstance(body, dict) else body
    if isinstance(agents, list):
        for a in agents:
            if isinstance(a, dict) and a.get("name") == AGENT_NAME:
                return a.get("id")
    code, agent = post("/v1/agents", {
        "id": "", "name": AGENT_NAME,
        "description": "negative-answer bisect over a corpus one search cannot exhaust",
        "instructions": (
            f"Answer using the search tool over the '{COLL}' collection. "
            "A question may require combining facts from more than one document: "
            "if the first search does not contain everything you need, search again "
            "with different terms. Answer with only the specific names or the single "
            "word asked for."),
        "tool_ids": ["search", "filter_query"]})
    if code not in (200, 201):
        sys.exit(f"agent create failed: {code} {str(agent)[:200]}")
    return agent.get("id")


def ingest():
    rng = random.Random(20260818)   # same seed as stag_v2 -> same filler corpus
    schema = {"name": COLL, "shard_count": 1, "fields": [
        {"name": "content", "type": "string", "bm25": True},
        {"name": "doc", "type": "string", "filterable": True},
        {"name": "vec", "type": "vector", "dimension": 1024,
         "metric": "cosine", "index": "hnsw"}]}
    code, body = post("/v1/collections", schema)
    print("create:", code, str(body)[:120])
    if code not in (200, 201, 409):
        sys.exit("cannot create collection")

    recs = [{"content": t, "doc": n} for n, t in stag_v2.FACTS]
    recs += [{"content": stag_v2.filler(rng, i), "doc": f"filler-{i}"}
             for i in range(FILLER)]
    print(f"ingesting {len(recs)} documents ({len(stag_v2.FACTS)} fact + {FILLER} filler)...")
    t0 = time.time()
    for start in range(0, len(recs), 250):
        chunk = recs[start:start + 250]
        code, body = post(f"/v1/collections/{COLL}/embed-insert",
                          {"auto_embed": {"vec": {"source_field": "content"}},
                           "records": chunk})
        if code not in (200, 201):
            sys.exit(f"ingest failed at {start}: {code} {str(body)[:300]}")
        print(f"  {start + len(chunk)}/{len(recs)}  ({time.time()-t0:.0f}s)")
    code, _ = post(f"/v1/collections/{COLL}/seal", {})
    print("seal:", code)
    print("agent:", ensure_agent())
    print(f"INGEST_OK docs={len(recs)} secs={time.time()-t0:.0f}")


def present(low, tok):
    return re.search(r"\b" + re.escape(tok) + r"\b", low) is not None


def run_cases(aid, kinds, repeats):
    rows = []
    for cid, kind, q, must, must_not in stag_v2.CASES:
        if kind not in kinds:
            continue
        for rep in range(repeats):
            code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
            ans = body.get("response", "") if code == 200 else f"HTTP {code}"
            low = (ans or "").lower()
            ok = (code == 200 and all(present(low, m) for m in must)
                  and not any(present(low, m) for m in must_not))
            rows.append({"id": cid, "kind": kind, "rep": rep, "ok": ok,
                         "answer": (ans or "")[:120]})
            print(f"  {'PASS' if ok else 'FAIL'} {cid:12s} r{rep} {(ans or '')[:70]!r}",
                  flush=True)
    return rows


def probe():
    aid = ensure_agent()
    if not aid:
        sys.exit("no agent")

    print("--- controls (validity gate) ---", flush=True)
    ctl = run_cases(aid, {"control"}, REPEATS_CTL)
    ctl_ok = sum(1 for r in ctl if r["ok"])

    print("--- negatives (the measurement) ---", flush=True)
    neg = run_cases(aid, {"negative"}, REPEATS_NEG)
    neg_ok = sum(1 for r in neg if r["ok"])

    # A corpus that cannot answer the controls cannot be trusted to report a
    # meaningful negative rate. Fail loudly instead of returning a number.
    valid = ctl_ok >= 0.8 * len(ctl)
    verdict = "VALID" if valid else "INVALID"
    print()
    print(f"RESULT controls={ctl_ok}/{len(ctl)} negatives={neg_ok}/{len(neg)} "
          f"({100*neg_ok/len(neg):.0f}%) {verdict}")
    json.dump({"controls_ok": ctl_ok, "controls_n": len(ctl),
               "negatives_ok": neg_ok, "negatives_n": len(neg),
               "valid": valid, "rows": ctl + neg},
              open("/tmp/probe_result.json", "w"), indent=2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"ingest": ingest, "probe": probe}[mode]()
