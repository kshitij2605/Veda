import { type ComponentType } from 'react'
import {
  Boxes,
  Cpu,
  Layers,
  GitMerge,
  FileText,
  Server,
  Activity,
  Shield,
  Bot,
  Gauge,
  Waypoints,
  BadgeCheck,
  Radar,
  Globe,
  DatabaseZap,
  Lock,
  SlidersHorizontal,
  ShieldAlert,
  Database,
  Route,
} from 'lucide-react'

type Icon = ComponentType<{ size?: number; strokeWidth?: number; className?: string }>

export interface Feature {
  icon: Icon
  title: string
  blurb: string
  /** Categorical node color (CSS value) for the icon. */
  accent: string
}

export const FEATURES: Feature[] = [
  {
    icon: Boxes,
    accent: 'var(--color-node-2)',
    title: 'Multimodal-native pipeline',
    blurb:
      'Upload a PDF, image, audio, or video — Veda parses it, chunks the text, and routes every modality to its embedder; point it at one unified omni model and a text query retrieves audio, video, and images from a single cross-modal space. Vectors, metadata, and blobs stored together — no external services to stitch.',
  },
  {
    icon: Cpu,
    accent: 'var(--color-node-5)',
    title: 'Pure-Rust + SIMD',
    blurb:
      'API to disk in one Rust binary, with SimSIMD distance kernels (AVX-512, NEON, SVE) in the hot loop — up to ~10× faster than scalar. No GC pauses, no CGo boundary, zero-copy mmap reads.',
  },
  {
    icon: Layers,
    accent: 'var(--color-node-3)',
    title: '10 vector index types',
    blurb:
      'HNSW, DiskANN, IVF-PQ, Flat, three RaBitQ variants, Sq4 + Sq8 scalar quantization — pick the recall, latency and memory trade-off per collection instead of working around one index. A suffix array and scalar filter indexes sit alongside them, and cuVS builds add a GPU CAGRA index that builds a 1M-vector index in 86 seconds.',
  },
  {
    icon: GitMerge,
    accent: 'var(--color-node-1)',
    title: 'Multi-vector fusion',
    blurb:
      'Search text, image, and metadata spaces at once and fuse the results with RRF, weighted sum, or intersection — in a single API call.',
  },
  {
    icon: FileText,
    accent: 'var(--color-node-4)',
    title: 'Built-in document intelligence',
    blurb:
      'Drop in a PDF, an Office file (DOCX / PPTX / XLSX), or a scan — Veda’s OCR pipeline (GLM-OCR + layout analysis) extracts the tables and images, chunks, and embeds it into searchable vectors in one API call. Zero external extraction tooling to stitch.',
  },
  {
    icon: Server,
    accent: 'var(--color-node-6)',
    title: 'Lightweight by default',
    blurb:
      'Three mandatory services — Veda, MinIO, NATS — with no external coordinator, message broker, or metadata store to provision. One compose file brings the whole thing up, and there is nothing else to stand up before you can index your first document.',
  },
  {
    icon: Activity,
    accent: 'var(--color-node-2)',
    title: 'Observability + built-in AutoOps',
    blurb:
      'Metrics, traces, and logs out of the box — Prometheus, Tempo, Loki, Grafana — plus an Ops Advisor that detects issues across 36 rules (latency, disk, CPU, error & log anomalies, SLO burn-rate, recall drift, capacity forecasts, change-points, LLM cost & quality), explains the root cause, alerts Slack/PagerDuty, and can auto-remediate.',
  },
  {
    icon: Shield,
    accent: 'var(--color-node-1)',
    title: 'An app platform, not just an index',
    blurb:
      'JSON-path filters (meta.user.tier = "gold"), per-record row-level security, record TTL with scheduled sweeps, cron jobs, and OLAP-style aggregations — built into the engine.',
  },
  {
    icon: Bot,
    accent: 'var(--color-node-3)',
    title: 'Agents & MCP server',
    blurb:
      'Connect Claude Desktop, Cursor, or any MCP client to your data — and get cross-modal retrieval no text-only vector DB can offer. Built-in agents, an 8-tool catalog, and A2A, all behind Veda’s JWT auth and row-level security.',
  },
  {
    icon: Gauge,
    accent: 'var(--color-node-4)',
    title: 'SLOs & service reliability',
    blurb:
      'Define service-level objectives with error budgets and Google-SRE multi-window burn-rate alerting, run synthetic self-checks against Veda’s own search / embed / health paths, and triage any incident with an AI ops agent that reads your signals and explains the root cause.',
  },
  {
    icon: Waypoints,
    accent: 'var(--color-node-6)',
    title: 'GenAI tracing + cost observability',
    blurb:
      'Agent and LLM calls emit OpenTelemetry GenAI-semantic-convention spans that drop natively into Datadog, Grafana Tempo, Phoenix, and Langfuse — with per-run token, latency, and USD cost metrics, and W3C context propagation that stitches Veda into your existing distributed trace.',
  },
  {
    icon: BadgeCheck,
    accent: 'var(--color-node-5)',
    title: 'Grounded agents with faithfulness scoring',
    blurb:
      'Every agent answer returns citations to the exact retrieved chunks and a 0–1 faithfulness + relevancy score (LLM-judge, embedding fallback) — the #1 RAG quality metric, scorable only because Veda owns the retriever. Plus an eval harness that graduates production traces into regression tests.',
  },
  {
    icon: Radar,
    accent: 'var(--color-node-2)',
    title: 'AIOps: forecasting + incident correlation',
    blurb:
      'Veda forecasts disk-full, memory, and recall-decay ETAs, detects sustained change-points, and semantically clusters related alerts into single incidents — flagging unprecedented ones — while an AI agent drafts the root cause from a cross-signal bundle (logs + queries + traces + resources). Predictive, causal, vector-native AIOps a metrics store without an embedder can’t do.',
  },
  {
    icon: Globe,
    accent: 'var(--color-node-1)',
    title: 'Self-hosted web RAG',
    blurb:
      'Crawl the open web straight into a collection (crawl4ai → clean markdown → chunk → embed) and give agents web_search + fetch_url tools to search → fetch → ground over live pages — all on open-source, self-hosted services. No external API, no data leaving your trust boundary.',
  },
  {
    icon: DatabaseZap,
    accent: 'var(--color-node-3)',
    title: 'Semantic LLM cache & guardrails',
    blurb:
      'Veda turns its vector engine into the LLM layer: a near-duplicate prompt is served from a semantic cache with no LLM call, an unsafe prompt is blocked by embedding-similarity guardrails, and every run lands in a trace store you search and cluster by meaning — observe, optimize, and secure in one engine.',
  },
  {
    icon: Route,
    accent: 'var(--color-node-3)',
    title: 'Answers that span documents',
    blurb:
      'Ask which plant assembles a product and whether that plant is ISO 14001 certified, and no single document holds the answer. Veda’s agent searches, reads the name out of the result, and searches again. On a 2,012-document corpus with Qwen3.5-122B it answered two-document and one-to-many joins correctly in about two searches. No knowledge graph, no query planner, nothing to keep in sync.',
  },
  {
    icon: ShieldAlert,
    accent: 'var(--color-node-2)',
    title: 'Guardrails on what you retrieve',
    blurb:
      'The dangerous instruction is not always in the prompt — it can be sitting in a PDF someone uploaded. Veda screens retrieved chunks at ingest, reusing the embedding it already computed, and drops the ones it flags at the tool-dispatch choke point, so every agent surface is covered by one mechanism. Raw search annotates instead of dropping, so recall stays measurable. A defence-in-depth layer, and the docs are explicit about what it does and does not catch.',
  },
  {
    icon: Database,
    accent: 'var(--color-node-6)',
    title: 'Query your Parquet where it lives',
    blurb:
      'Register a collection over existing Parquet files and search them in place — no ingestion, no second copy, no duplicate storage. Schema and vector dimension are read from the footer and validated up front, filters use the same evaluator as internal collections, and search is an exact scan at recall 1.0.',
  },
  {
    icon: Lock,
    accent: 'var(--color-node-5)',
    title: 'Runs entirely on your infrastructure',
    blurb:
      'The embedder, OCR, reranker, web search, and the LLM endpoint are all self-hostable and swappable, so the whole stack can run air-gapped on your own hardware. No data leaves your trust boundary and nothing phones home — the privacy and sovereignty posture regulated and on-prem deployments need.',
  },
  {
    icon: SlidersHorizontal,
    accent: 'var(--color-node-4)',
    title: 'Self-tuning RAG pipeline',
    blurb:
      'Veda generates a labeled test set from your own corpus, then searches its configuration space — chunker, embedder, index, fusion, rerank — for the best recall / faithfulness / latency trade-off. Apply the winner with one click (reversible), and optionally auto-re-tune every time the collection grows.',
  },
]
