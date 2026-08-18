#!/usr/bin/env python3
"""Multi-hop measurement, v2 - on a corpus where one search CANNOT return everything.

v1 used 15 documents. A single search with limit=20 therefore returned the whole
corpus, so the model had every fact in context and no hop was required: the run
measured context-scanning, not multi-hop retrieval. Diagnosis in
documentation/2026-08-17-multi-hop-retrieval-scope.md.

v2 buries the same fact documents in ~2,000 filler documents drawn from the same
business vocabulary. A top-k search cannot surface both halves of any answer, so
combining now genuinely requires more than one retrieval.

Each case runs REPEATS times because the loop is nondeterministic - v1 saw the
same case answer correctly and incorrectly across runs, and a single pass cannot
tell a fix from variance.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:6355"
OUT = "/home/mac/test/multihop-logs"
COLL = "multihop2"
FILLER = 2000
REPEATS = 3


def post(path, body, timeout=1200):
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


FACTS = [
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
]

# (id, kind, question, must-all, must-none)
CASES = [
    ("ctl-1", "control", "Where is Nordwind Components headquartered?", ["estonia"], []),
    ("ctl-2", "control", "Which plant assembles Product Halcyon?", ["riga"], []),
    ("ctl-3", "control", "Who signed the Q1 master service agreement?", ["dana", "whitfield"], []),

    ("one2one-1", "one-to-one",
     "Did the person who signed the Q2 master service agreement also appear in the Q3 incident report? Answer yes or no.",
     ["yes"], []),
    ("one2one-2", "one-to-one",
     "Did the person who signed the Q1 master service agreement also appear in the Q3 incident report? Answer yes or no.",
     ["no"], []),
    ("one2one-3", "one-to-one",
     "Is the plant that assembles Product Halcyon certified to ISO 14001? Answer yes or no.",
     ["yes"], []),
    ("one2one-4", "one-to-one",
     "Is the plant that assembles Product Vesper certified to ISO 14001? Answer yes or no.",
     ["no"], []),

    ("one2many-1", "one-to-many",
     "Which of our suppliers are headquartered in countries where the 2026 tariff schedule changed? List every supplier name.",
     ["nordwind", "braewick", "meridian"], ["kaito"]),
    ("one2many-2", "one-to-many",
     "List every supplier headquartered in Estonia.",
     ["nordwind", "braewick"], []),

    ("negative-1", "negative",
     "Which supplier is headquartered in Poland? If no supplier is headquartered there, answer exactly: none.",
     ["none"], ["nordwind", "kaito", "meridian", "braewick"]),
    ("negative-2", "negative",
     "Who signed the Q4 master service agreement? If the corpus does not say, answer exactly: unknown.",
     ["unknown"], ["dana", "owen"]),
]

TOPICS = ["logistics", "procurement", "compliance", "warehousing", "freight",
          "invoicing", "auditing", "inventory", "packaging", "customs"]
VERBS = ["reviewed", "updated", "scheduled", "consolidated", "reconciled",
         "archived", "escalated", "deferred", "approved", "circulated"]
NOUNS = ["shipment manifests", "supplier scorecards", "tariff notices",
         "service agreements", "incident summaries", "plant certifications",
         "quality reports", "freight quotations", "customs declarations",
         "vendor onboarding packs"]


def filler(rng, i):
    return (f"Operations note {i}: the {rng.choice(TOPICS)} team {rng.choice(VERBS)} "
            f"{rng.choice(NOUNS)} for the {rng.choice(['Riga','Da Nang','Tallinn','Santiago','Hanoi'])} "
            f"site during {rng.choice(['Q1','Q2','Q3','Q4'])}. "
            f"No supplier headquarters or tariff determinations are recorded here.")


def setup():
    rng = random.Random(20260818)
    schema = {"name": COLL, "shard_count": 1, "fields": [
        {"name": "content", "type": "string", "bm25": True},
        {"name": "doc", "type": "string", "filterable": True},
        {"name": "vec", "type": "vector", "dimension": 1024, "metric": "cosine", "index": "hnsw"}]}
    code, body = post("/v1/collections", schema)
    print("create:", code, str(body)[:120])
    if code not in (200, 201, 409):
        sys.exit("cannot create collection")

    recs = [{"content": t, "doc": n} for n, t in FACTS]
    recs += [{"content": filler(rng, i), "doc": f"filler-{i}"} for i in range(FILLER)]
    print(f"ingesting {len(recs)} documents ({len(FACTS)} fact + {FILLER} filler)...")
    for start in range(0, len(recs), 250):
        chunk = recs[start:start + 250]
        code, body = post(f"/v1/collections/{COLL}/embed-insert",
                          {"auto_embed": {"vec": {"source_field": "content"}}, "records": chunk},
                          timeout=1800)
        if code not in (200, 201):
            sys.exit(f"ingest failed at {start}: {code} {str(body)[:200]}")
        print(f"  {start + len(chunk)}/{len(recs)}")
    code, _ = post(f"/v1/collections/{COLL}/seal", {})
    print("seal:", code)

    code, agent = post("/v1/agents", {
        "id": "", "name": "Multi-hop probe v2",
        "description": "multi-hop over a corpus where one search cannot return everything",
        "instructions": (
            f"Answer using the search tool over the '{COLL}' collection. "
            "A question may require combining facts from more than one document: "
            "if the first search does not contain everything you need, search again "
            "with different terms. Answer with only the specific names or the single "
            "word asked for."),
        "tool_ids": ["search"]})
    print("agent:", code, agent.get("id"))
    return agent.get("id")


def main():
    import re
    aid = setup()
    if not aid:
        sys.exit("no agent")

    def present(low, tok):
        return re.search(r"\b" + re.escape(tok) + r"\b", low) is not None

    rows = []
    for cid, kind, q, must, must_not in CASES:
        for rep in range(REPEATS):
            t0 = time.time()
            code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
            dur = time.time() - t0
            ans = body.get("response", "") if code == 200 else f"HTTP {code}"
            low = (ans or "").lower()
            ok = code == 200 and all(present(low, m) for m in must) \
                and not any(present(low, m) for m in must_not)
            steps = body.get("steps", []) if code == 200 else []
            searches = sum(1 for s in steps if isinstance(s, dict) and s.get("kind") == "tool_call")
            rows.append({"id": cid, "kind": kind, "rep": rep, "ok": ok, "secs": round(dur, 1),
                         "searches": searches, "answer": ans[:200], "q": q})
            print(f"{'PASS' if ok else 'FAIL'}  {cid:12s} r{rep} {dur:6.1f}s "
                  f"searches={searches}  {ans[:80]!r}")

    json.dump(rows, open(f"{OUT}/results_v2.json", "w"), indent=2)

    print("\n" + "=" * 74)
    for kind in ["control", "one-to-one", "one-to-many", "negative"]:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        p = sum(1 for r in sub if r["ok"])
        ids = sorted({r["id"] for r in sub})
        per = []
        for i in ids:
            s = [r for r in sub if r["id"] == i]
            per.append(f"{i}:{sum(1 for r in s if r['ok'])}/{len(s)}")
        avg_s = sum(r["searches"] for r in sub) / len(sub)
        print(f"{kind:12s} {p:2d}/{len(sub)} ({100*p/len(sub):3.0f}%)  "
              f"mean {avg_s:.1f} searches   [{'  '.join(per)}]")
    print("=" * 74)


if __name__ == "__main__":
    main()
