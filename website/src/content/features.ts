import { type ComponentType } from 'react'
import { Boxes, Cpu, Layers, GitMerge, FileText, Server, Activity } from 'lucide-react'

type Icon = ComponentType<{ size?: number; strokeWidth?: number; className?: string }>
type Modality = 'text' | 'image' | 'audio' | 'video' | 'doc' | 'mixed' | 'accent'

export interface Feature {
  icon: Icon
  title: string
  blurb: string
  modality: Modality
}

export const FEATURES: Feature[] = [
  {
    icon: Boxes,
    modality: 'image',
    title: 'Multimodal-native pipeline',
    blurb:
      'Upload a PDF, image, or video — Veda parses it, chunks the text, routes each modality to the right embedding model, and stores vectors, metadata, and blobs together. No external embedding services to stitch.',
  },
  {
    icon: Cpu,
    modality: 'text',
    title: 'Pure-Rust + SIMD',
    blurb:
      'API to disk in one Rust binary, with AVX2/FMA distance kernels in the hot loop. No GC pauses, no CGo boundary, zero-copy mmap reads — lower, more predictable p99 latency.',
  },
  {
    icon: Layers,
    modality: 'doc',
    title: 'Seven index algorithms',
    blurb:
      'HNSW, DiskANN, IVF-PQ, Flat, RaBitQ, a suffix array, and scalar indexes — pick the right recall/latency/memory tradeoff per collection, all in one engine.',
  },
  {
    icon: GitMerge,
    modality: 'mixed',
    title: 'Multi-vector fusion',
    blurb:
      'Search text, image, and metadata spaces at once and fuse the results with RRF, weighted sum, or intersection — in a single API call.',
  },
  {
    icon: FileText,
    modality: 'video',
    title: 'Built-in document intelligence',
    blurb:
      'A complete OCR pipeline (GLM-OCR + FlashParse) plus six chunking strategies turn raw documents into searchable vectors — zero external tooling.',
  },
  {
    icon: Server,
    modality: 'audio',
    title: 'Lightweight by default',
    blurb:
      'Three mandatory services — Veda, MinIO, NATS — versus the five+ a Milvus cluster needs. docker compose up and you have a vector database in under 30 seconds.',
  },
  {
    icon: Activity,
    modality: 'accent',
    title: 'OpenTelemetry-first observability',
    blurb:
      'Metrics, traces, and logs out of the box — Prometheus, Tempo, Loki, Grafana — with per-phase search timing and a pre-built dashboard.',
  },
]
