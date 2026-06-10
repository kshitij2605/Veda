import { type ComponentType } from 'react'
import { Upload, ScanText, Scissors, Sparkles, Layers, Search } from 'lucide-react'

type Icon = ComponentType<{ size?: number; strokeWidth?: number }>

export const PIPELINE: { icon: Icon; title: string; caption: string }[] = [
  { icon: Upload, title: 'Upload', caption: 'PDF, image, audio, video, or text — one endpoint.' },
  { icon: ScanText, title: 'OCR', caption: 'GLM-OCR + FlashParse extract text, tables, and figures.' },
  { icon: Scissors, title: 'Chunk', caption: 'Seven strategies, from markdown-recursive to semantic.' },
  { icon: Sparkles, title: 'Embed', caption: 'Each modality auto-routed to the right model.' },
  { icon: Layers, title: 'Index', caption: 'HNSW, DiskANN, IVF-PQ, RaBitQ — your choice.' },
  { icon: Search, title: 'Search', caption: 'Dense, sparse, hybrid, and multi-vector fusion.' },
]
