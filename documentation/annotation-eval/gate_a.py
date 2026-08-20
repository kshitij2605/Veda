#!/usr/bin/env python3
"""Gate A - does the model actually READ field annotations?

The gate that must pass before annotations are made a tuned parameter.
It exists because of a specific failure: the stagnation detector was
correct, unit-tested and load-bearing, and fired in 1 run out of 6. Judged
on win rate alone it looked merely ineffective rather than inert. So the
first question is not "do annotations help?" but "does the model read them
at all?".

Design
------
Field names are deliberately non-obvious (`ctry_cd`, `amt_2`, `ref_no`) and
- the discriminating part - the STORED ENCODING differs from how a question
phrases it:

  ctry_cd  holds ISO-3166 alpha-2 ("EE"), questions say "Estonia"
  amt_2    holds an amount in CENTS (int), questions ask about dollars

An unannotated model must guess. It typically filters ctry_cd = 'Estonia'
and gets zero rows, or reports amt_2 as dollars and is wrong by 100x.
An annotated model is told semantic_type and value_domain and can encode
the filter correctly.

Two arms differ ONLY in whether the schema carries annotations; same data,
same questions, same agent instructions, same seed.

Measured
--------
1. FIRING RATE (the gate): did the model's own tool-call arguments use the
   stored encoding? Read off the returned steps, not inferred from the
   answer, so a lucky guess does not count as a read.
2. Answer correctness, secondary. Reported with per-case detail because at
   these sample sizes the aggregate hides more than it shows.

Run:  python3 -u gate_a.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:6357"
OUT = "/home/mac/test/anno-logs"
REPEATS = 5

SUPPLIERS = [
    # (ref_no, name, ctry_cd, amt_2 in cents)
    ("SUP-1001", "Nordwind Components", "EE", 4_250_00),
    ("SUP-1002", "Kaito Fasteners", "VN", 1_180_00),
    ("SUP-1003", "Meridian Alloys", "CL", 9_640_00),
    ("SUP-1004", "Braewick Seals", "EE", 2_700_00),
    ("SUP-1005", "Larkspur Tooling", "PL", 6_310_00),
]

ANNOTATIONS = {
    "ref_no": {
        "description": "Internal supplier reference number.",
        "semantic_type": "supplier_ref",
        "value_domain": ["SUP-1001", "SUP-1002", "SUP-1003"],
    },
    "ctry_cd": {
        "description": "Country where the supplier is headquartered, as an ISO-3166 alpha-2 code.",
        "semantic_type": "iso3166_alpha2",
        "value_domain": ["EE", "VN", "CL", "PL"],
    },
    "amt_2": {
        "description": "Annual contract value, stored in CENTS (divide by 100 for dollars).",
        "semantic_type": "currency_cents_usd",
    },
}

# (id, question, must-all, must-none)
CASES = [
    ("q-country", "How many suppliers are headquartered in Estonia? Answer with just the number.",
     ["2"], []),
    ("q-country-list", "List the names of every supplier headquartered in Estonia.",
     ["nordwind", "braewick"], ["kaito", "meridian", "larkspur"]),
    ("q-amount", "What is the annual contract value of Meridian Alloys in US dollars? Answer with just the number.",
     ["9640"], ["964000"]),
    ("q-chile", "Which supplier is headquartered in Chile? Answer with just the name.",
     ["meridian"], []),
]


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


def setup(coll, annotated):
    fields = [
        {"name": "content", "type": "string", "bm25": True},
        {"name": "ref_no", "type": "string", "filterable": True},
        {"name": "ctry_cd", "type": "string", "filterable": True},
        {"name": "amt_2", "type": "int64", "filterable": True},
        {"name": "vec", "type": "vector", "dimension": 1024,
         "metric": "cosine", "index": "hnsw"},
    ]
    if annotated:
        for f in fields:
            if f["name"] in ANNOTATIONS:
                f["annotations"] = ANNOTATIONS[f["name"]]

    code, body = post("/v1/collections", {"name": coll, "shard_count": 1, "fields": fields})
    print(f"create {coll}: {code} {str(body)[:160]}")
    if code not in (200, 201, 409):
        sys.exit("cannot create collection")

    recs = [{"content": f"Supplier record for {name}.",
             "ref_no": ref, "ctry_cd": cc, "amt_2": amt}
            for ref, name, cc, amt in SUPPLIERS]
    code, body = post(f"/v1/collections/{coll}/embed-insert",
                      {"auto_embed": {"vec": {"source_field": "content"}}, "records": recs},
                      timeout=600)
    if code not in (200, 201):
        sys.exit(f"ingest failed: {code} {str(body)[:300]}")
    post(f"/v1/collections/{coll}/seal", {})

    code, agent = post("/v1/agents", {
        "id": "", "name": f"anno probe {'B' if annotated else 'A'}",
        "description": "field-annotation gate",
        "instructions": (
            f"Answer questions about the '{coll}' collection. "
            "Call describe_collection first to learn the schema, then use "
            "filter_query or search to read records. Answer with only the "
            "number or names asked for."),
        "tool_ids": ["describe_collection", "filter_query", "search", "aggregate"]})
    print(f"agent {coll}: {code} {agent.get('id')}")
    return agent.get("id")


def tool_calls(steps):
    """Every tool call's (name, argument-json-string) from a converse response."""
    out = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        name = s.get("tool") or s.get("name") or s.get("tool_name")
        args = s.get("args") or s.get("arguments") or s.get("input")
        if name or args:
            out.append((str(name), json.dumps(args) if args is not None else ""))
    return out


def read_signal(calls):
    """Did the model use the STORED encoding in its own tool arguments?

    This is the firing-rate signal. Read from the arguments the model
    chose, never from the final answer, so that guessing right by luck
    does not register as having read the annotation.

    Returns (used_code, used_plaintext).
    """
    blob = " ".join(a for _, a in calls)
    used_code = bool(re.search(r"\b(EE|CL|VN|PL)\b", blob))
    used_plain = bool(re.search(r"estonia|chile|vietnam|poland", blob, re.I))
    return used_code, used_plain


def run_arm(label, coll, annotated):
    aid = setup(coll, annotated)
    if not aid:
        sys.exit("no agent")
    rows = []
    for cid, q, must, must_not in CASES:
        for rep in range(REPEATS):
            t0 = time.time()
            code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
            dur = time.time() - t0
            ans = body.get("response", "") if code == 200 else f"HTTP {code}"
            low = (ans or "").lower()
            ok = code == 200 \
                and all(re.search(r"\b" + re.escape(m) + r"\b", low) for m in must) \
                and not any(re.search(r"\b" + re.escape(m) + r"\b", low) for m in must_not)
            steps = body.get("steps", []) if code == 200 else []
            calls = tool_calls(steps)
            used_code, used_plain = read_signal(calls)
            rows.append({
                "arm": label, "id": cid, "rep": rep, "ok": ok,
                "secs": round(dur, 1), "calls": len(calls),
                "used_code": used_code, "used_plain": used_plain,
                "answer": ans[:200], "tool_args": [a for _, a in calls][:6],
            })
            print(f"[{label}] {'PASS' if ok else 'FAIL'} {cid:15s} r{rep} "
                  f"{dur:5.1f}s calls={len(calls)} code={used_code} plain={used_plain} "
                  f"{ans[:60]!r}")
    return rows


def summarise(rows):
    print("\n" + "=" * 78)
    for arm in ["A-unannotated", "B-annotated"]:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        ok = sum(1 for r in sub if r["ok"])
        code_used = sum(1 for r in sub if r["used_code"])
        plain_used = sum(1 for r in sub if r["used_plain"])
        print(f"{arm:16s} correct {ok:2d}/{len(sub)} ({100*ok/len(sub):3.0f}%)   "
              f"used stored encoding {code_used:2d}/{len(sub)} ({100*code_used/len(sub):3.0f}%)   "
              f"used plaintext {plain_used:2d}/{len(sub)}")
        per = {}
        for r in sub:
            per.setdefault(r["id"], []).append(r)
        for cid, rs in per.items():
            print(f"    {cid:15s} {sum(1 for r in rs if r['ok'])}/{len(rs)} correct, "
                  f"{sum(1 for r in rs if r['used_code'])}/{len(rs)} used code")
    print("=" * 78)
    print("GATE A passes only if arm B uses the stored encoding in a clear")
    print("majority of runs AND does so materially more often than arm A.")
    print("=" * 78)


def main():
    rows = []
    rows += run_arm("A-unannotated", "anno_plain", False)
    rows += run_arm("B-annotated", "anno_rich", True)
    import os
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/gate_a.json", "w"), indent=2)
    summarise(rows)


if __name__ == "__main__":
    main()
