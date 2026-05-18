/**
 * Two-component PCA via power iteration. Deterministic-ish seed
 * (sin-based) so re-renders don't shuffle the layout.
 *
 * Reduces an N×D matrix to N 2-d points. ~12 power-iteration passes
 * converges well enough for visualisation. Linear time per pass: O(N*D).
 *
 * Lives here so the atlas page can reuse it without pulling the old
 * VectorAtlas file.
 */
export function pcaTwoDim(vectors: number[][]): [number, number][] {
  const n = vectors.length
  if (n === 0) return []
  const d = vectors[0]!.length
  if (d < 2) return vectors.map(() => [0, 0])

  // 1) Centre each dim.
  const mean = new Float64Array(d)
  for (const v of vectors) {
    for (let i = 0; i < d; i++) mean[i]! += v[i]!
  }
  for (let i = 0; i < d; i++) mean[i]! /= n
  const centered: Float64Array[] = vectors.map((v) => {
    const out = new Float64Array(d)
    for (let i = 0; i < d; i++) out[i] = v[i]! - mean[i]!
    return out
  })

  // 2) Top eigenvector via power iteration on X^T X (computed implicitly).
  const v1 = seedUnit(d, 0)
  powerIterate(v1, centered, d, 12)
  const proj1 = centered.map((row) => dot(row, v1))

  // 3) Second component: orthogonalise + deflate against v1.
  const v2 = seedUnit(d, 1)
  orthogonalize(v2, v1, d)
  powerIterateDeflated(v2, centered, v1, d, 12)
  const proj2 = centered.map((row) => dot(row, v2))

  return proj1.map((x, i) => [x, proj2[i]!])
}

function seedUnit(d: number, salt: number): Float64Array {
  const v = new Float64Array(d)
  let n = 0
  for (let i = 0; i < d; i++) {
    v[i] = Math.sin(i * 12.9898 + 78.233 + salt * 17.5)
    n += v[i]! * v[i]!
  }
  n = Math.sqrt(n)
  if (n > 0) for (let i = 0; i < d; i++) v[i]! /= n
  return v
}

function dot(a: Float64Array | number[], b: Float64Array | number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number)
  return s
}

function normalise(v: Float64Array, d: number) {
  let n = 0
  for (let i = 0; i < d; i++) n += v[i]! * v[i]!
  n = Math.sqrt(n)
  if (n > 0) for (let i = 0; i < d; i++) v[i]! /= n
}

function orthogonalize(v: Float64Array, ref: Float64Array, d: number) {
  const p = dot(v, ref)
  for (let i = 0; i < d; i++) v[i]! -= p * ref[i]!
  normalise(v, d)
}

function powerIterate(
  v: Float64Array,
  rows: Float64Array[],
  d: number,
  iters: number,
) {
  const next = new Float64Array(d)
  for (let it = 0; it < iters; it++) {
    next.fill(0)
    for (const row of rows) {
      const proj = dot(row, v)
      for (let i = 0; i < d; i++) next[i]! += proj * row[i]!
    }
    for (let i = 0; i < d; i++) v[i] = next[i]!
    normalise(v, d)
  }
}

function powerIterateDeflated(
  v: Float64Array,
  rows: Float64Array[],
  v1: Float64Array,
  d: number,
  iters: number,
) {
  const next = new Float64Array(d)
  for (let it = 0; it < iters; it++) {
    next.fill(0)
    for (const row of rows) {
      const proj = dot(row, v) - dot(row, v1) * dot(v, v1)
      for (let i = 0; i < d; i++) next[i]! += proj * (row[i]! - v1[i]! * dot(row, v1))
    }
    for (let i = 0; i < d; i++) v[i] = next[i]!
    orthogonalize(v, v1, d)
  }
}
