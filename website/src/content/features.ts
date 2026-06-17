import { type ComponentType } from 'react'
import { Boxes, Cpu, Layers, GitMerge, FileText, Server, Activity, Shield, Bot, Gauge } from 'lucide-react'

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
      'Upload a PDF, image, or video — Veda parses it, chunks the text, routes each modality to the right embedding model, and stores vectors, metadata, and blobs together. No external embedding services to stitch.',
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
      'A complete OCR pipeline (GLM-OCR + FlashParse) plus six chunking strategies turn raw documents into searchable vectors — zero external tooling.',
  },
  {
    icon: Server,
    accent: 'var(--color-node-6)',
    title: 'Lightweight by default',
    blurb:
      'Three mandatory services — Veda, MinIO, NATS — versus the five+ a Milvus cluster needs. docker compose up and you have a vector database in under 30 seconds.',
  },
  {
    icon: Activity,
    accent: 'var(--color-node-2)',
    title: 'Observability + built-in AutoOps',
    blurb:
      'Metrics, traces, and logs out of the box — Prometheus, Tempo, Loki, Grafana — plus an Ops Advisor that detects issues across 28 rules (latency, disk, CPU, error & log anomalies, SLO burn-rate), explains the root cause, alerts Slack/PagerDuty, and can auto-remediate.',
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
]
