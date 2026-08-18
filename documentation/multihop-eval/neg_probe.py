#!/usr/bin/env python3
"""Is the negative-answer gap a prompt problem or a loop problem?

Baseline measured 3/6 on negative cases (answers that should be "none" or
"unknown"), against 100% everywhere else. The failures recite retrieved context
instead of concluding. The probe's instructions never said what to do when
nothing is found, so test that before proposing any change to the agent loop:
adding machinery to fix a prompt omission would be building ahead of evidence.

Two arms, same corpus, same cases, 4 repeats:
  A = baseline instructions (as measured)
  B = identical + one sentence on how to answer when the corpus lacks the fact
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:6355"
COLL = "multihop2"
REPEATS = 4

BASE_INSTR = (
    f"Answer using the search tool over the '{COLL}' collection. "
    "A question may require combining facts from more than one document: "
    "if the first search does not contain everything you need, search again "
    "with different terms. Answer with only the specific names or the single "
    "word asked for."
)
NEG_CLAUSE = (
    " If your searches do not turn up the fact being asked for, say so in the "
    "exact word the question specifies rather than summarising what you did "
    "find. Do not list documents as a substitute for an answer."
)

CASES = [
    ("negative-1",
     "Which supplier is headquartered in Poland? If no supplier is headquartered there, answer exactly: none.",
     ["none"], ["nordwind", "kaito", "meridian", "braewick"]),
    ("negative-2",
     "Who signed the Q4 master service agreement? If the corpus does not say, answer exactly: unknown.",
     ["unknown"], ["dana", "owen"]),
    # A positive control in the same arm: the fix must not break cases that work.
    ("one2many-1",
     "Which of our suppliers are headquartered in countries where the 2026 tariff schedule changed? List every supplier name.",
     ["nordwind", "braewick", "meridian"], ["kaito"]),
    ("one2one-2",
     "Did the person who signed the Q1 master service agreement also appear in the Q3 incident report? Answer yes or no.",
     ["no"], []),
]


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


def make_agent(name, instr):
    code, a = post("/v1/agents", {
        "id": "", "name": name, "description": "negative-answer arm",
        "instructions": instr, "tool_ids": ["search"]})
    if code != 200:
        sys.exit(f"agent create failed: {code} {a}")
    return a["id"]


def run_arm(label, aid):
    rows = []
    for cid, q, must, must_not in CASES:
        for rep in range(REPEATS):
            t0 = time.time()
            code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
            dur = time.time() - t0
            ans = body.get("response", "") if code == 200 else f"HTTP {code}"
            low = (ans or "").lower()
            ok = code == 200 and all(present(low, m) for m in must) \
                and not any(present(low, m) for m in must_not)
            searches = sum(1 for s in body.get("steps", [])
                           if isinstance(s, dict) and s.get("kind") == "tool_call")
            rows.append({"arm": label, "id": cid, "rep": rep, "ok": ok,
                         "secs": round(dur, 1), "searches": searches,
                         "answer": ans[:220]})
            print(f"  [{label}] {'PASS' if ok else 'FAIL'} {cid:12s} r{rep} "
                  f"{dur:5.1f}s s={searches} {ans[:70]!r}")
    return rows


print("=== arm A: baseline instructions ===")
a_id = make_agent("neg arm A", BASE_INSTR)
rows = run_arm("A", a_id)

print("\n=== arm B: + negative-answer clause ===")
b_id = make_agent("neg arm B", BASE_INSTR + NEG_CLAUSE)
rows += run_arm("B", b_id)

json.dump(rows, open("/home/mac/test/multihop-logs/neg_arms.json", "w"), indent=2)

print("\n" + "=" * 70)
for arm in ("A", "B"):
    for cid, _, _, _ in CASES:
        sub = [r for r in rows if r["arm"] == arm and r["id"] == cid]
        p = sum(1 for r in sub if r["ok"])
        print(f"arm {arm}  {cid:12s} {p}/{len(sub)}   mean {sum(r['searches'] for r in sub)/len(sub):.1f} searches")
    tot = [r for r in rows if r["arm"] == arm]
    print(f"arm {arm}  TOTAL        {sum(1 for r in tot if r['ok'])}/{len(tot)}")
    print("-" * 70)
