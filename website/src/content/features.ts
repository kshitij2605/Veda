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
    title: 'Nine index algorithms',
    blurb:
      'HNSW, DiskANN, IVF-PQ, Flat, RaBitQ, Sq4 + Sq8 scalar quantization, a suffix array, and scalar filter indexes — pick the right recall/latency/memory tradeoff per collection.',
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
      'Three mandatory services — Veda, MinIO, NATS — with no external coordinator, message broker, or metadata store to provision. docker compose up and you have a vector database in under 30 seconds.',
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
    icon: Lock,
    accent: 'var(--color-node-5)',
    title: 'Runs entirely on your infrastructure',
    blurb:
      'The embedder, OCR, reranker, web search, and the LLM endpoint are all self-hostable and swappable, so the whole stack can run air-gapped on your own hardware. No data leaves your trust boundary and nothing phones home — the privacy and sovereignty posture regulated and on-prem deployments need.',
  },
]
