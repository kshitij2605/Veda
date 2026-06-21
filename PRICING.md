# Veda Pricing

Veda is open source and free to self-host. Paid tiers add a commercial license, support,
enterprise features, and (later) managed hosting.

> Indicative pricing for planning — finalize before publishing a public pricing page.
> Contact **sales@veda.example**. The model below intentionally starts light: a solo team
> can run tiers 1–2 today; the managed cloud (tier 3) is added only when revenue justifies
> the operational cost.

---

## Tier 1 — Community  ·  **Free**
Self-hosted, open source. The full database + the Niriksha dashboard.
- Veda engine under AGPL-3.0; Niriksha under FSL-1.1-ALv2 (see [`LICENSING.md`](LICENSING.md)).
- All core capabilities: multimodal ingest (text · image · audio · video), any-document
  pipeline (PDF/DOCX/PPTX/XLSX/scans → OCR → chunk → embed), hybrid + cross-modal search,
  the full index suite, agents/MCP/A2A, AIOps & LLM-observability.
- Community support (GitHub issues/discussions).

**Best for:** individuals, startups, research, and any team comfortable self-hosting under
the AGPL.

## Tier 2 — Enterprise (self-hosted)  ·  **From ~$15k–50k / year**  ·  *Contact us*
Same software you run yourself, plus the things companies need to adopt it in production.
- **Commercial license** — removes the AGPL copyleft / network-source obligation, so you
  can embed Veda in a proprietary product or run it internally without releasing changes.
- **Enterprise add-ons** (closed-source): SSO/SAML & SCIM, fine-grained RBAC, audit logs,
  BYOK / customer-managed encryption, advanced multi-tenancy.
- **Support & SLA:** priority bug fixes, security patches, response-time SLA, upgrade and
  deployment assistance.

**Best for:** companies that can't use AGPL, need enterprise security/compliance features,
or want a support contract. *(This is the primary paid product at launch — high value, low
operational overhead.)*

## Tier 3 — Veda Cloud (managed)  ·  *Roadmap*
Managed Veda, rolled out in stages to keep operations sane:
1. **Bring-Your-Own-Cloud (BYOC) / Hybrid** — Veda runs a thin control plane; the data
   plane runs in **your** AWS/GCP account (your data never leaves your infrastructure).
2. **Cloud-marketplace one-click deploy** (AWS/GCP Marketplace) — pay through your existing
   cloud bill.
3. **Fully-managed serverless** (later) — usage-based, object-storage-backed:
   - Storage **~$0.30 / GB-month**
   - Writes **~$4–5 / million write units**
   - Reads/queries **~$16–20 / million read units**
   - **Free tier:** ~1–2 GB / ~1M vectors, no credit card
   - **Paid minimum:** ~$25–50 / month

**Best for:** teams that want Veda without running it.

## Multimodal / RAG add-ons  ·  *metered, premium*
Veda's differentiator — it replaces a whole RAG **ingestion** stack — maps to high-margin
metered units that pure text-vector stores don't offer:
- **Document processing:** per-page OCR/parse (e.g. $ / 1k pages)
- **Embedding generation:** per-token (text) and **per-image / per-second** (audio · video)
- **Media blob storage:** $ / GB-month for the original files stored alongside their vectors
- A bundled **"Multimodal / RAG" plan** for teams replacing Unstructured + LangChain + a
  vector DB + a separate eval tool with one platform.

---

### Summary

| Tier | Price | Hosting | Key value |
|---|---|---|---|
| **Community** | Free | Self-host | Full OSS engine + dashboard |
| **Enterprise** | ~$15k–50k/yr · contact us | Self-host | Commercial license + SSO/RBAC/audit + support/SLA |
| **Veda Cloud** | Usage-based *(roadmap)* | BYOC → marketplace → serverless | Managed, pay-as-you-go |
| **Multimodal add-ons** | Metered | Any paid tier | OCR / multimodal embedding / media storage |

*Free OSS → paid enterprise license & support → BYOC/marketplace → full serverless cloud
(only when it earns its operational cost).*
