import api from './client'

export interface ChunkSummary {
  id: string
  type: string
  preview: string
}

export interface ProcessDocumentResponse {
  status: string
  source_file: string
  pages: number
  images_extracted: number
  tables_extracted: number
  records_inserted: number
  record_ids: string[]
  chunks: ChunkSummary[]
}

export type ChunkingStrategyName =
  | 'markdown_recursive'
  | 'fixed_size'
  | 'sentence'
  | 'markdown_h1'
  | 'markdown_h2'
  | 'markdown_h3'
  | 'semantic'
  | 'page'
  | 'hierarchical'
  | 'late_chunking'
  | 'contextual_chunking'
  | 'none'

export interface ChunkPreviewSpec {
  label?: string
  strategy_name: ChunkingStrategyName | string
  chunk_size?: number
  chunk_overlap?: number
}

export interface ChunkPreviewChunk {
  index: number
  text: string
  char_count: number
  start_offset: number
  end_offset: number
  heading_path: string[]
  page_number?: number
}

export interface ChunkPreviewRun {
  label: string
  strategy_name: string
  elapsed_ms: number
  total_chunks: number
  total_chars: number
  avg_chars: number
  min_chars: number
  max_chars: number
  truncated: boolean
  chunks: ChunkPreviewChunk[]
}

export interface ChunkPreviewResponse {
  input_chars: number
  runs: ChunkPreviewRun[]
}

export const chunkPreview = (text: string, runs: ChunkPreviewSpec[]) =>
  api.post<ChunkPreviewResponse>('/chunk/preview', { text, runs })

export interface ProcessDocumentSettings {
  chunking_strategy: ChunkingStrategyName
  chunk_size?: number
  chunk_overlap?: number
  dpi?: number
  skip_captions?: boolean
}

/**
 * Multipart-POST to /v1/collections/{name}/process-document.
 * Returns the same shape the gateway serialises (sync — Phase 6 adds
 * streaming progress).
 */
export async function processDocument(
  collection: string,
  file: File,
  settings: ProcessDocumentSettings,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ProcessDocumentResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('chunking_strategy', settings.chunking_strategy)
  if (settings.chunk_size != null)
    form.append('chunk_size', String(settings.chunk_size))
  if (settings.chunk_overlap != null)
    form.append('chunk_overlap', String(settings.chunk_overlap))
  if (settings.dpi != null) form.append('dpi', String(settings.dpi))
  if (settings.skip_captions) form.append('skip_captions', 'true')

  const { data } = await api.post<ProcessDocumentResponse>(
    `/collections/${collection}/process-document`,
    form,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(e.loaded, e.total)
      },
    },
  )
  return data
}
