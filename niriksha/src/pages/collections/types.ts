import type { FieldSchema } from '../../api/collections'

export type Density = 'comfortable' | 'compact' | 'dense'

export interface ColumnState {
  /** Field name */
  name: string
  /** Display order (sortable) */
  order: number
  /** Hidden in the grid (still in schema) */
  hidden?: boolean
}

export const VECTOR_TYPES = new Set(['vector', 'sparse_vector', 'multi_vector'])

export function isVectorField(f: FieldSchema): boolean {
  return VECTOR_TYPES.has(f.type)
}

export function isBlobField(f: FieldSchema): boolean {
  return f.type === 'blob'
}

export function isScalarField(f: FieldSchema): boolean {
  return !isVectorField(f) && !isBlobField(f)
}

/** Density → row height + cell padding utility classes. */
export const DENSITY_DIMS: Record<
  Density,
  { rowH: number; cellPx: string; textPx: string }
> = {
  comfortable: { rowH: 36, cellPx: 'px-3 py-2', textPx: 'text-[12.5px]' },
  compact: { rowH: 28, cellPx: 'px-2.5 py-1.5', textPx: 'text-[12px]' },
  dense: { rowH: 22, cellPx: 'px-2 py-0.5', textPx: 'text-[11.5px]' },
}
