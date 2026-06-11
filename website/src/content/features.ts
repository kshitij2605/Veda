import { type ComponentType } from 'react'
import { Boxes, Cpu, Layers, GitMerge, FileText, Server, Activity } from 'lucide-react'

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
    title: 'OpenTelemetry-first observability',
    blurb:
      'Metrics, traces, and logs out of the box — Prometheus, Tempo, Loki, Grafana — with per-phase search timing and a pre-built dashboard.',
  },
]
