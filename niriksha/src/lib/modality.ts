export type Modality = 'text' | 'image' | 'audio' | 'video' | 'doc' | 'mixed'

export interface SchemaField {
  name: string
  type: string
}

const VECTORY = new Set(['vector', 'sparse_vector', 'multi_vector'])

const RULES: [RegExp, Modality][] = [
  [/imag|img|clip|visual|photo|vision/i, 'image'],
  [/audio|sound|voice|speech|clap/i, 'audio'],
  [/video|motion/i, 'video'],
  [/doc|page|pdf|colpali|ocr/i, 'doc'],
  [/text|bge|e5|title|caption|body|content|embedding/i, 'text'],
]

function classify(name: string): Modality {
  for (const [re, m] of RULES) if (re.test(name)) return m
  return 'text'
}

/** Derive a collection's dominant modality from its vector-ish field names. */
export function deriveModality(fields: SchemaField[]): Modality {
  const found = new Set<Modality>()
  for (const f of fields) if (VECTORY.has(f.type)) found.add(classify(f.name))
  if (found.size === 0) return 'text'
  if (found.size === 1) return [...found][0]
  return 'mixed'
}

export function modalityColorVar(m: Modality): string {
  return `var(--color-modality-${m})`
}

export const MODALITY_LABEL: Record<Modality, string> = {
  text: 'Text',
  image: 'Image',
  audio: 'Audio',
  video: 'Video',
  doc: 'Document',
  mixed: 'Multimodal',
}
