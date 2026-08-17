#!/usr/bin/env python3
"""Measure the EXISTING Veda agent loop on multi-hop questions.

The corpus is built so no single document answers any eval question: each
answer needs a fact from one document joined to a fact in another. That is what
makes this a multi-hop measurement rather than a retrieval one.

Each question also has a single-hop control whose answer sits in one document.
Without the control, a low multi-hop score could just mean retrieval is broken;
the control separates "cannot retrieve" from "cannot combine".
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:6355"
OUT = "/home/mac/test/multihop-logs"
COLL = "multihop"


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw or "{}")
        except Exception:
            return e.code, {"raw": raw[:400]}


DOCS = [
    ("suppliers-a", "Nordwind Components is our primary supplier of pressure valves. Nordwind Components is headquartered in Estonia."),
    ("suppliers-b", "Kaito Fasteners supplies all industrial bolts. Kaito Fasteners is headquartered in Vietnam."),
    ("suppliers-c", "Meridian Alloys supplies specialty steel. Meridian Alloys is headquartered in Chile."),
    ("suppliers-d", "Braewick Seals supplies gasket assemblies. Braewick Seals is headquartered in Estonia."),
    ("tariff-2026", "The 2026 tariff schedule changed for Estonia and Chile. Rates for Vietnam and Poland were unchanged from 2025."),
    ("msa-q1", "The Q1 master service agreement was signed by Dana Whitfield on behalf of the company."),
    ("msa-q2", "The Q2 master service agreement was signed by Owen Marsh."),
    ("incident-q3", "The Q3 incident report lists Owen Marsh and Priya Raman as responders to the outage."),
    ("incident-q2", "The Q2 incident report lists Dana Whitfield as the escalation owner."),
    ("product-x", "Product Halcyon is assembled exclusively at the Riga plant."),
    ("product-y", "Product Vesper is assembled exclusively at the Da Nang plant."),
    ("plants", "The Riga plant holds ISO 14001 certification. The Da Nang plant holds ISO 9001 certification only."),
    ("noise-1", "Quarterly logistics review: shipping volumes rose 4% across all suppliers."),
    ("noise-2", "The tariff schedule is published annually by the trade ministry."),
    ("noise-3", "All master service agreements are retained for seven years."),
]

# (id, hops, question, must-all-appear, must-NOT-appear)
CASES = [
    # --- single-hop controls: answer lives in one document ---
    ("ctl-1", 1, "Where is Nordwind Components headquartered?", ["estonia"], []),
    ("ctl-2", 1, "Which plant assembles Product Halcyon?", ["riga"], []),
    ("ctl-3", 1, "Who signed the Q1 master service agreement?", ["dana whitfield"], []),
    ("ctl-4", 1, "Which countries had tariff changes in the 2026 schedule?", ["estonia", "chile"], []),
    # --- multi-hop: require joining two documents ---
    ("mh-1", 2,
     "Which of our suppliers are headquartered in countries where the 2026 tariff schedule changed? List the supplier names.",
     ["nordwind", "braewick", "meridian"], ["kaito"]),
    ("mh-2", 2,
     "Did the person who signed the Q1 master service agreement also appear in the Q3 incident report? Answer yes or no.",
     ["no"], []),
    ("mh-3", 2,
     "Did the person who signed the Q2 master service agreement also appear in the Q3 incident report? Answer yes or no.",
     ["yes"], []),
    ("mh-4", 2,
     "Is the plant that assembles Product Halcyon certified to ISO 14001? Answer yes or no.",
     ["yes"], []),
    ("mh-5", 2,
     "Is the plant that assembles Product Vesper certified to ISO 14001? Answer yes or no.",
     ["no"], []),
    ("mh-6", 2,
     "Which supplier of gasket assemblies is in a country affected by the 2026 tariff change? Give the supplier name.",
     ["braewick"], []),
]


def setup():
    print("### collection ###")
    schema = {
        "name": COLL,
        "shard_count": 1,
        "fields": [
            {"name": "content", "type": "string", "bm25": True},
            {"name": "doc", "type": "string", "filterable": True},
            {"name": "vec", "type": "vector", "dimension": 1024,
             "metric": "cosine", "index": "hnsw"},
        ],
    }
    code, body = post("/v1/collections", schema)
    print("create:", code, str(body)[:160])
    if code not in (201, 200, 409):
        sys.exit("cannot create collection")

    print("### embed-insert ###")
    ins = {
        "auto_embed": {"vec": {"source_field": "content"}},
        "records": [{"content": t, "doc": n} for n, t in DOCS],
    }
    code, body = post(f"/v1/collections/{COLL}/embed-insert", ins)
    print("embed-insert:", code, str(body)[:200])
    if code not in (200, 201):
        sys.exit("ingest failed")

    code, body = post(f"/v1/collections/{COLL}/seal", {})
    print("seal:", code)

    print("### sanity: does single-shot retrieval find a one-hop fact? ###")
    code, body = post("/v1/agent-tools/search/execute",
                      {"arguments": {"collection": COLL,
                                     "query": "where is Nordwind Components headquartered",
                                     "limit": 3}})
    print("search tool:", code)
    print("  ", str(body)[:300])

    print("### agent ###")
    code, body = post("/v1/agents", {
        "id": "",
        "name": "Multi-hop probe",
        "description": "measures the existing ReAct loop on questions needing two documents",
        "instructions": (
            "Answer using the search tool over the 'multihop' collection. "
            "A question may require combining facts from more than one document: "
            "if the first search does not contain everything you need, search again "
            "with different terms. Answer with only the specific names or the single "
            "word asked for."
        ),
        "tool_ids": ["search"],
    })
    print("agent:", code, str(body)[:120])
    return body.get("id")


def grade(text, must, must_not):
    """Word-boundary matching, not substring.

    Substring matching made this grader unsound: the expected "no" matches
    inside "not", "none", "cannot" and even "Nordwind", so an answer that
    recited the facts without ever concluding would have scored correct.
    Every expected token is now matched on word boundaries.
    """
    low = (text or "").lower()

    def present(tok):
        return re.search(r"\b" + re.escape(tok) + r"\b", low) is not None

    return all(present(m) for m in must) and not any(present(m) for m in must_not)


def main():
    aid = setup()
    if not aid:
        sys.exit("no agent id")
    print(f"\n### running {len(CASES)} cases against agent {aid} ###\n")

    results = []
    for cid, hops, q, must, must_not in CASES:
        t0 = time.time()
        code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
        dur = time.time() - t0
        answer = body.get("response", "") if code == 200 else f"HTTP {code}: {str(body)[:120]}"
        steps = body.get("steps", []) if code == 200 else []
        tool_calls = sum(1 for s in steps if isinstance(s, dict) and "tool" in json.dumps(s).lower()[:200])
        ok = code == 200 and grade(answer, must, must_not)
        results.append({
            "id": cid, "hops": hops, "ok": ok, "code": code,
            "secs": round(dur, 1), "steps": len(steps), "tool_calls": tool_calls,
            "q": q, "answer": (answer or "")[:300], "must": must, "must_not": must_not,
        })
        print(f"{'PASS' if ok else 'FAIL'}  {cid:6s} hops={hops} {dur:6.1f}s steps={len(steps):2d}  {answer[:110]!r}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    for h in (1, 2):
        sub = [r for r in results if r["hops"] == h]
        if not sub:
            continue
        p = sum(1 for r in sub if r["ok"])
        label = "single-hop control" if h == 1 else "MULTI-HOP"
        avg = sum(r["secs"] for r in sub) / len(sub)
        stp = sum(r["steps"] for r in sub) / len(sub)
        print(f"{label:20s}  {p}/{len(sub)} correct ({100*p/len(sub):.0f}%)   "
              f"mean {avg:.1f}s, mean {stp:.1f} steps")
    print("=" * 72)
    print(f"\nfull detail: {OUT}/results.json")


if __name__ == "__main__":
    main()
