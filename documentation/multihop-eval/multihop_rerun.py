#!/usr/bin/env python3
"""Re-run the v2 suite against the SAME corpus, to compare loop changes.

multihop_v2.py builds the corpus and runs the suite. This runs only the suite,
against the collection v2 already ingested, so the only difference between a
baseline run and a post-change run is the server binary. Re-ingesting would
re-embed 2,012 documents and change the corpus, which would make the two runs
incomparable.

Cases, instructions and grader are copied verbatim from multihop_v2.py - if
they drift, the comparison against the recorded baseline is meaningless.

Usage: python3 -u multihop_rerun.py <label>
(Python block-buffers stdout to a file; -u is not optional here.)
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:6355"
OUT = "/home/mac/test/multihop-logs"
COLL = "multihop2"
REPEATS = 3
LABEL = sys.argv[1] if len(sys.argv) > 1 else "rerun"

# (id, kind, question, must-all, must-none) - verbatim from multihop_v2.py
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

INSTR = (
    f"Answer using the search tool over the '{COLL}' collection. "
    "A question may require combining facts from more than one document: "
    "if the first search does not contain everything you need, search again "
    "with different terms. Answer with only the specific names or the single "
    "word asked for."
)


def post(path, body, timeout=1200):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode()[:300]}


def present(low, tok):
    return re.search(r"\b" + re.escape(tok) + r"\b", low) is not None


code, agent = post("/v1/agents", {
    "id": "", "name": f"Multi-hop probe {LABEL}",
    "description": "v2 suite re-run against the existing corpus",
    "instructions": INSTR, "tool_ids": ["search"]})
if code != 200 or not agent.get("id"):
    sys.exit(f"agent create failed: {code} {agent}")
aid = agent["id"]
print(f"agent: {aid}  label={LABEL}", flush=True)

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
                     "searches": searches, "answer": ans[:200]})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:12s} r{rep} {dur:6.1f}s "
              f"searches={searches}  {ans[:80]!r}", flush=True)

json.dump(rows, open(f"{OUT}/results_{LABEL}.json", "w"), indent=2)

print("\n" + "=" * 74, flush=True)
for kind in ["control", "one-to-one", "one-to-many", "negative"]:
    sub = [r for r in rows if r["kind"] == kind]
    p = sum(1 for r in sub if r["ok"])
    per = []
    for i in sorted({r["id"] for r in sub}):
        s = [r for r in sub if r["id"] == i]
        per.append(f"{i}:{sum(1 for r in s if r['ok'])}/{len(s)}")
    print(f"{kind:12s} {p:2d}/{len(sub)} ({100*p/len(sub):3.0f}%)  "
          f"mean {sum(r['searches'] for r in sub)/len(sub):.1f} searches   [{'  '.join(per)}]",
          flush=True)
print("=" * 74, flush=True)
