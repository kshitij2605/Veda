# Connecting Hermes Agent to Veda over MCP

**Date:** 2026-08-20
**Status:** verified end to end on `iitgpu11`
**Scope:** Phase 1 of the Hermes integration plan — Veda as an MCP tool server for
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).

## Summary

Hermes Agent talks to Veda with **no Veda-side adapter and no Hermes-side plugin** — it is
config only. Getting there required two spec-compliance fixes in `crates/veda-gateway/src/rest/mcp.rs`;
both were real bugs in Veda that would have broken any conforming MCP client, not just Hermes.

The integration was verified by capturing the wire traffic through a logging proxy. A green
console line is not evidence, so every claim below is backed by a request/response pair.

## The two bugs Hermes found

### 1. The `MCP-Protocol-Version` gate rejected `initialize`

Veda applied the header check to every request. Hermes's client (`mcp 2.0.0`) offers
`2025-11-25`, newer than anything Veda supports, so the very first `initialize` was answered
with `400 unsupported MCP-Protocol-Version`. Hermes retried four times and gave up.

This inverted the purpose of the header. The header governs requests made *after* initialization;
`initialize` negotiates the version in its own body, and the server is supposed to answer with a
version it does support. Veda already had that negotiation logic — the gate ran before it could
be reached.

**Fix:** exempt `initialize` from the header gate. A newer client now negotiates *down* to
`2025-06-18` instead of failing.

### 2. Notifications received a JSON-RPC error response

After `initialize`, Hermes sends `notifications/initialized`. A JSON-RPC *notification* carries no
`id`, and JSON-RPC 2.0 is explicit that "the Server MUST NOT reply to a Notification". Veda
replied with an error object, because its handler treated a missing `id` as a malformed request.

**Fix:** return `202 Accepted` with an empty body for any request without an `id`.

Both fixes are covered by unit tests in `mcp.rs`, including one that asserts the advertised
default version is itself in the supported list — the class of mistake that produced bug 1.

## Recipe

### 1. Veda side

Run Veda with auth on. The relevant variable is `VEDA_AUTH_MODE`, **not** `VEDA_AUTH_ENABLED`
(which does not exist and silently leaves auth off):

```bash
export VEDA_AUTH_MODE=static
export VEDA_JWT_SECRET=<secret>
export VEDA_STATIC_USERS='admin:adminpw:admin,reader:readerpw:viewer'
veda-server ./veda-data 0.0.0.0:6340
```

Get a token:

```bash
TOKEN=$(curl -s -X POST http://localhost:6340/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"username":"reader","password":"readerpw"}' | jq -r .token)
```

### 2. Hermes side

In `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  veda:
    url: "http://veda-host:6340/v1/mcp"
    headers:
      Authorization: "Bearer <jwt>"
    tools:
      include:
        - search
        - filter_query
        - describe_collection
        - list_collections
        - get_record
```

**Set the `tools.include` allowlist.** Veda's catalog has 15 tools including `upsert_record` and
`delete_collection`. Handing an autonomous agent the full catalog by default is not what most
operators want. The five above are the read-only working set.

### 3. Verify

```bash
hermes mcp test veda
```

Note that `hermes mcp test` reports raw **discovery** (all 15 tools), while `hermes mcp list`
reports what the agent is actually **exposed** to (5). Both numbers are correct; they measure
different things. The allowlist applies at agent exposure, so an excluded tool never reaches
the model — see the evidence below.

## Evidence

### Handshake and tool call, captured on the wire

Asked "which supplier is headquartered in Estonia?", Hermes answered `Nordwind Components` —
correct, and only retrievable from Veda. The full exchange:

```
--> GET  /v1/mcp                                        <-- 405 (Allow: POST)
--> POST /v1/mcp method='initialize'             id=1   <-- 200
--> POST /v1/mcp method='notifications/initialized'     <-- 202 (empty body)
--> POST /v1/mcp method='tools/list'             id=2   <-- 200
--> POST /v1/mcp method='tools/call'             id=3   <-- 200
--> POST /v1/mcp method='tools/call'             id=4   <-- 200
```

The client does probe `GET` first and receives `405`. That was the transport risk flagged before
this work started; Hermes recovers from it and proceeds to POST. No change needed.

### The allowlist holds

Asked to insert a record, the agent replied `NO WRITE TOOL AVAILABLE` and listed exactly the five
allowed tools. **Zero `tools/call` requests reached Veda** — the refusal happened in Hermes,
before the wire. The collection was unchanged.

### Auth failure is clean, not a hang

With a corrupted token, `hermes mcp test veda` failed in 16 seconds with
`✗ Connection failed: Server returned an error response`, and printed the token masked as
`Bear***oken`. Veda returned `401` to the `GET` probe and to both `initialize` attempts. No hang,
no retry storm, no credential leak into logs.

### Schema-declared field sensitivity survives the hop

This is the property that justifies *not* embedding Hermes in Veda: security is enforced at
Veda's tool-dispatch choke point, so no transport can bypass it.

A `suppliers` collection was created with `salary_band` marked `sensitivity: restricted`. Through
MCP:

| caller | fields returned | withheld |
|---|---|---|
| admin | `content`, `ctry_cd`, `vec` | `salary_band` |
| reader (viewer role) | `content`, `ctry_cd`, `vec` | `salary_band` |

Filtering on the restricted field is refused outright:

> `field 'salary_band' is restricted by the collection schema and cannot be read, filtered, or aggregated by this caller`

Responses carry a `field_guard` report (`{"affected_records": 1, "withheld_fields": ["salary_band"]}`)
so the omission is visible rather than silent.

**Note the admin row.** MCP callers are treated as non-admin regardless of their JWT role, because
the MCP handler does not set the admin flag on its `ToolContext`. This is deliberate and
fail-closed: an autonomous agent holding an admin token still cannot read restricted fields. If an
operator needs an agent to see them, the field should not be marked `restricted`.

## What was not done

**Veda as a Hermes memory provider** (Phase 2) is gated on this phase and remains unstarted.
Hermes's built-in memory is `MEMORY.md` plus SQLite FTS5 full-text history — no embeddings, no
semantic search, no multimodal recall — against a documented `memory.provider` interface with
eight existing plugins. That is a Veda-shaped hole, but the claim to test there is that semantic
recall beats FTS5 on Hermes's own workload, which needs a real transcript recall set and a
measured number. No number, no claim.

## Environment notes

Recorded because both cost real time:

- Hermes pins `.python-version = 3.11`, which on this host resolved to `/usr/bin/python3.11` —
  a **3.11.0 release candidate**. Symptom: `hermes --version` and `.venv/bin/python --version`
  disagree. Fix: `uv python install 3.11.15` with `UV_PYTHON_PREFERENCE=only-managed`.
- The `mcp` client package is an **extra**, not a base dependency — deliberate supply-chain
  scoping. MCP support is silently absent without `uv sync --extra mcp`.
- `hermes run` does not exist; the one-shot form is `hermes -z "<prompt>"`. `hermes tools`
  requires a TTY and cannot be scripted.
