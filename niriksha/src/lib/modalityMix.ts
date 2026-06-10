import { deriveModality, type Modality, type SchemaField } from './modality'

export interface CollectionDetail {
  name: string
  fields: SchemaField[]
}

export interface ModalitySlice {
  modality: Modality
  count: number
}

const ORDER: Modality[] = ['text', 'image', 'audio', 'video', 'doc', 'mixed']

/** Count collections by their derived modality, in a stable order. */
export function deriveModalityMix(details: CollectionDetail[]): ModalitySlice[] {
  const counts = new Map<Modality, number>()
  for (const d of details) {
    const m = deriveModality(d.fields)
    counts.set(m, (counts.get(m) ?? 0) + 1)
  }
  return ORDER.filter((m) => counts.has(m)).map((m) => ({ modality: m, count: counts.get(m)! }))
}
