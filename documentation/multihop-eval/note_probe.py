#!/usr/bin/env python3
"""Did stagnation detection ever fire on the failing negative cases?

A null result has two very different causes: the signal reached the model and
did not change its behaviour, or the signal never fired. Replay the same
bookkeeping the loop does over the returned steps to tell them apart.
"""
import json
import urllib.error
import urllib.request

BASE = "http://localhost:6355"
INSTR = ("Answer using the search tool over the 'multihop2' collection. "
         "A question may require combining facts from more than one document: "
         "if the first search does not contain everything you need, search again "
         "with different terms. Answer with only the specific names or the single "
         "word asked for.")
QS = [
    "Which supplier is headquartered in Poland? If no supplier is headquartered there, answer exactly: none.",
    "Who signed the Q4 master service agreement? If the corpus does not say, answer exactly: unknown.",
]


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode()[:200]}


code, agent = post("/v1/agents", {
    "id": "", "name": "note probe", "description": "stagnation firing check",
    "instructions": INSTR, "tool_ids": ["search"]})
aid = agent["id"]

for q in QS:
    for rep in range(3):
        code, body = post(f"/v1/agents/{aid}/converse", {"input": q})
        seen, calls, fired = set(), 0, 0
        for s in body.get("steps", []):
            if s.get("kind") != "tool_result":
                continue
            res = s.get("result") or {}
            hits = res.get("hits") or res.get("records") or []
            ids = [h.get("id") for h in hits if isinstance(h, dict)]
            fresh = [i for i in ids if i not in seen]
            seen.update(ids)
            calls += 1
            if not fresh:
                fired += 1
        print(f"{q[:38]!r} r{rep}: {calls} retrievals, note fired {fired}x "
              f"-> {body.get('response','')[:60]!r}", flush=True)
