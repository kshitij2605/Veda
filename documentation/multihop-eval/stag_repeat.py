#!/usr/bin/env python3
"""Re-run the v2 stagnation suite N times against the EXISTING corpus.

stag_v2.py re-ingests 2,012 documents on every invocation. The corpus has not
changed, so repeats reuse it: the question here is variance in the *loop*, which
is nondeterministic, not variance in the data.

Also records whether the stagnation note actually fired on each call. A result
that improved while the mechanism never fired would mean something else moved.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/mac/test")
import stag_v2  # noqa: E402  (CASES/COLL live there; main() is __main__-guarded)

BASE = stag_v2.BASE
COLL = stag_v2.COLL
NOTE_MARK = "[note: this call surfaced no records that are both"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
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
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:300]}


def get(path, timeout=60):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except Exception as e:
        return 0, {"err": str(e)}


def ensure_agent():
    code, body = get("/v1/agents")
    agents = body.get("agents", body) if isinstance(body, dict) else body
    if isinstance(agents, list):
        for a in agents:
            if isinstance(a, dict) and a.get("name") == "Multi-hop probe v2":
                return a.get("id")
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
    return agent.get("id")


def stagnation_counter():
    """(fired, not_fired) from the server's own counter.

    The note is appended to the message sent to the model, NOT to the
    AgentStep::ToolResult in the returned trace -- so scanning `steps` for it
    reports 0 no matter what actually happened. This counter is incremented on
    every check, fired or not, precisely so a rate is observable.
    """
    try:
        with urllib.request.urlopen(BASE + "/v1/admin/metrics", timeout=30) as r:
            body = r.read().decode()
    except Exception:
        return (0, 0)
    fired = notfired = 0
    for line in body.splitlines():
        if not line.startswith("veda_agent_stagnation_checks_total"):
            continue
        val = int(float(line.rsplit(None, 1)[-1]))
        if 'fired="true"' in line:
            fired = val
        elif 'fired="false"' in line:
            notfired = val
    return (fired, notfired)


def present(low, tok):
    return re.search(r"\b" + re.escape(tok) + r"\b", low) is not None


def main():
    code, info = get(f"/v1/collections/{COLL}")
    if code != 200:
        sys.exit(f"corpus '{COLL}' missing ({code}) -- refusing to re-ingest; "
                 f"run stag_v2.py once if it is genuinely gone")
    n = info.get("record_count", info.get("num_records", "?"))
    print(f"corpus '{COLL}': {n} records")
    aid = ensure_agent()
    if not aid:
        sys.exit("no agent")
    print(f"reusing corpus '{COLL}' and agent {aid}\n")

    f0_case, n0_case = stagnation_counter()
    print(f"counter at start: fired={f0_case} not_fired={n0_case}")
    all_runs = []
    for run in range(RUNS):
        rows = []
        for cid, kind, q, must, must_not in stag_v2.CASES:
            for rep in range(REPEATS):
                t0 = time.time()
                code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
                dur = time.time() - t0
                ans = body.get("response", "") if code == 200 else f"HTTP {code}"
                low = (ans or "").lower()
                ok = code == 200 and all(present(low, m) for m in must) \
                    and not any(present(low, m) for m in must_not)
                steps = body.get("steps", []) if code == 200 else []
                searches = sum(1 for s in steps
                               if isinstance(s, dict) and s.get("kind") == "tool_call")
                f1, n1 = stagnation_counter()
                fired = f1 - f0_case
                checks = (f1 + n1) - (f0_case + n0_case)
                f0_case, n0_case = f1, n1
                rows.append({"run": run, "id": cid, "kind": kind, "rep": rep, "ok": ok,
                             "secs": round(dur, 1), "searches": searches,
                             "fired": fired, "checks": checks, "answer": ans[:200]})
                print(f"run{run} {'PASS' if ok else 'FAIL'}  {cid:12s} r{rep} "
                      f"{dur:5.1f}s searches={searches} fired={int(fired)}  {ans[:60]!r}")
        all_runs.append(rows)
        print()

    out = "/home/mac/test/stag-logs/results_repeats.json"
    json.dump([r for rows in all_runs for r in rows], open(out, "w"), indent=1)
    print(f"wrote {out}\n")

    kinds = ["control", "one-to-one", "one-to-many", "negative"]
    print(f"{'kind':<13}" + "".join(f"{'run'+str(i):>12}" for i in range(RUNS))
          + f"{'fired/checks':>13} {'rate':>5}")
    for k in kinds:
        cells = ""
        fired = checks = 0
        for rows in all_runs:
            sel = [r for r in rows if r["kind"] == k]
            cells += f"{sum(r['ok'] for r in sel):>6}/{len(sel):<6}"
            fired += sum(r["fired"] for r in sel)
            checks += sum(r.get("checks", 0) for r in sel)
        rate = f"{100*fired/checks:.0f}%" if checks else "n/a"
        print(f"{k:<13}{cells}{fired:>5}/{checks:<5} {rate:>5}")


if __name__ == "__main__":
    main()
