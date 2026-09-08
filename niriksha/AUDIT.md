# Niriksha Feature Audit

> **STATUS: the "what still needs building" list below is from 2026-05-18 and
> was never updated. Most of it was built in the following weeks. Re-verified
> 2026-09-09 — do not plan from the original text.**
>
> | Original claim | Verified 2026-09-09 |
> |---|---|
> | "`process-document` … no UI — flagship feature" | **Built.** `pages/Pipelines.tsx` (678 lines) + `api/pipelines.ts` multipart-POSTs `/v1/collections/{name}/process-document`; routed at `/pipelines`. Landed 2026-06-10 (`6445f03`). |
> | "No login page; token never set" | **Built.** `pages/Login.tsx`, `lib/auth.ts` posts `/v1/auth/login`; routed at `/login`. Landed 2026-06-11 (`f64ca6d`). |
> | "SearchPlayground … only `vector` is reachable" | **Built.** `QueryMode = 'dense' \| 'sparse' \| 'bm25' \| 'hybrid'`, with bm25 field/weight and hybrid explain. Last touched 2026-08-17 (`02896ba`). |
> | "VectorAtlas … PCA … method dropdown is locked" | **Built.** `atlas/AtlasPage.tsx` defaults to UMAP, runs it in a web worker with epoch progress. Landed 2026-06-10 (`6445f03`). |
> | "IndexManager — read-only" | **Still true.** Zero mutating API calls in the page. Blocked on the server, which exposes no reindex / metric / quantization mutation — a backend addition has to come first. |
>
> A session on 2026-09-09 came within one step of rebuilding the
> `process-document` UI because this file said it did not exist. This header
> exists so the next reader checks the code before planning against the list.
>
> **What is genuinely open here is one item — IndexManager mutations — and it
> is blocked on backend endpoints that do not exist.** Adding them is engine
> work, not UI work.
>
> The endpoint-coverage table below has NOT been re-audited line by line; only
> the five headline claims above were checked. Treat the rest as 2026-05-18
> data.

---

Dev server: `http://iitgpu11.hon.otsuka-shokai.co.jp:5174/niriksha`
(`npm run dev -- --host 0.0.0.0 --port 5174` from `niriksha/`).

This document maps each Niriksha page to the gateway endpoints it
calls and lists what still needs to be built.

## 1. Endpoints the gateway exposes vs. what Niriksha uses

| Endpoint                                           | Method | In UI?  | Notes |
|----------------------------------------------------|--------|---------|-------|
| `/v1/admin/health`                                 | GET    | ✅      | DashboardHome, SystemHealth |
| `/v1/admin/readiness`                              | GET    | ❌      | Could supplement SystemHealth on container starts |
| `/v1/auth/login`                                   | POST   | ❌      | No login page; token is read from `localStorage.veda_token` if present, never set |
| `/v1/admin/nodes`                                  | GET    | ❌      | Useful in SystemHealth once Raft lands |
| `/v1/admin/metrics`                                | GET    | ✅      | SystemHealth metrics table |
| `/v1/collections`                                  | GET    | ✅      | List endpoint — summary only |
| `/v1/collections`                                  | POST   | ✅      | Create modal now uses the right shape + types |
| `/v1/collections/{name}`                           | GET    | ✅      | All detail pages |
| `/v1/collections/{name}`                           | DELETE | ✅      | Drop button on CollectionBrowser detail |
| `/v1/collections/{name}/search`                    | POST   | ✅      | Dense-vector only; nested body + explain + truncate_to |
| `/v1/collections/{name}/insert`                    | POST   | ❌      | No raw-insert UI |
| `/v1/collections/{name}/embed-insert`              | POST   | ❌      | Major: text/image → embedding → insert |
| `/v1/collections/{name}/records/{id}`              | GET    | ❌      | Per-record detail page |
| `/v1/collections/{name}/records`                   | GET    | ✅      | Pagination is wired |
| `/v1/collections/{name}/records`                   | DELETE | ❌      | No "delete record" affordance |
| `/v1/collections/{name}/evaluate`                  | GET    | ✅      | EvaluationView |
| `/v1/collections/{name}/rerank`                    | POST   | ❌      | No standalone rerank step in SearchPlayground |
| `/v1/mcp`                                          | POST   | ❌      | Probably not user-facing |
| `/v1/route`                                        | POST   | ❌      | Adaptive query router not exposed |
| `/v1/collections/{name}/process-document`          | POST   | ❌      | OCR + chunk + embed pipeline — flagship feature |
| `/v1/collections/{name}/process-documents`         | POST   | ❌      | Batch variant |

## 2. Search-side features Niriksha doesn't expose

The gateway's `QueryBody` accepts these query types, but only `vector`
is reachable from the UI:

- `sparse_vector` (SPLADE) — server side wired end-to-end.
- `bm25` — full-text against `bm25 = true` string fields.
- `hybrid` — dense + BM25 fused with RRF.
- `multi_vector` — fan-out + fusion across multiple dense fields.
- `colpali` — multi-vector late-interaction.
- `rerank` block — re-rank top-N via cross-encoder.

A first cut would be a tabbed "Query Type" selector in SearchPlayground:
Dense / Sparse / BM25 / Hybrid / Rerank. Each tab sets the right
sub-object in the `query` body.

## 3. Pages with thin functionality

- **VectorAtlas** — does PCA on the first 5000 records pulled with
  `fetchRecords`. No t-SNE / UMAP, no clustering, no density heatmap.
  Currently the only viz available. Method dropdown is locked.
- **EvaluationView** — runs `/evaluate` which returns duplicates +
  field-quality. The "RAG usability / FAQ usability / cross-modal
  alignment" cards explicitly say "not yet implemented".
- **IndexManager** — read-only. No way to rebuild / reindex / change
  metric / change quantization. Server doesn't expose those mutations
  either, so this would need a backend addition first.

## 4. Bugs fixed in this pass

All five of these were broken at runtime or 400'd at the server:

1. `FieldSchema` interface: `field_type` / `dimensions` → real keys
   `type` / `dimension`, plus the previously-dropped `metric` / `model` /
   `filterable`.
2. List endpoint shape: pages were calling `.fields.filter` on the
   summary; replaced with the `CollectionSummary` type.
3. Create-collection body: was sending `field_type` / `dimensions` and
   bad type labels (`integer` / `float` / `boolean`); now sends the
   gateway's `type` / `dimension` + the full type vocabulary.
4. Search request body: was sending `{field, vector, …}`; gateway
   requires `{query: {vector: {field, values}}, …}`.
5. IndexManager's fake "HNSW / RaBitQ-4bit / Cosine" labels inferred
   from dimension are gone — replaced with server-reported values.

## 5. Suggested follow-up priority

1. **Document processing UI** on `CollectionBrowser` detail. The
   `/process-document` flow is Veda's flagship — multipart upload,
   show progress, then surface chunks_inserted / pages.
2. **Embed-insert form** — paste text / pick image, server runs the
   embedder. Removes the "you must hand-craft a 1024-dim JSON array"
   barrier from SearchPlayground too.
3. **Sparse / BM25 / Hybrid search tabs** in SearchPlayground.
4. **Rerank toggle** — already a `RerankConfig` sub-object in the
   search request; add a checkbox + model field.
5. **Adaptive router probe** — small panel: "Type a query, see which
   strategy `/v1/route` recommends".
6. **Drop / delete records** on CollectionBrowser.
