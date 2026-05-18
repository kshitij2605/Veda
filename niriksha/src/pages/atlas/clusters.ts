import { llmComplete } from '../../api/collections'
import type { RecordRow } from '../../api/collections'

export interface Cluster {
  id: number
  /** 2-D centroid in canvas coords. */
  cx: number
  cy: number
  size: number
  /** Indices into the original points array. */
  members: number[]
  /** Indices of the records closest to the centroid (for LLM exemplars). */
  exemplars: number[]
  label?: string
}

/**
 * Lloyd's k-means on 2-D coords. k is small (≤16), points typically
 * <5k, so this finishes in <50 ms. Seeds with k-means++.
 *
 * Returns clusters annotated with size, centroid, and the indices of
 * the ~5 records closest to each centroid (used as LLM exemplars
 * downstream).
 */
export function kmeans(points: [number, number][], k: number): Cluster[] {
  if (points.length === 0 || k <= 0) return []
  k = Math.min(k, points.length)

  // k-means++ seeding.
  const seeds: [number, number][] = []
  const rng = mulberry32(123)
  seeds.push(points[Math.floor(rng() * points.length)]!)
  while (seeds.length < k) {
    const dists = points.map((p) => {
      let min = Infinity
      for (const s of seeds) {
        const d = sqDist(p, s)
        if (d < min) min = d
      }
      return min
    })
    const total = dists.reduce((a, b) => a + b, 0)
    let r = rng() * total
    let pick = 0
    for (let i = 0; i < dists.length; i++) {
      r -= dists[i]!
      if (r <= 0) {
        pick = i
        break
      }
    }
    seeds.push(points[pick]!)
  }

  let centroids = seeds.map(([x, y]) => [x, y] as [number, number])
  let assignments = new Array<number>(points.length).fill(0)

  for (let iter = 0; iter < 25; iter++) {
    let changed = false
    // Assign each point to the nearest centroid.
    for (let i = 0; i < points.length; i++) {
      let best = 0
      let bestD = Infinity
      for (let c = 0; c < centroids.length; c++) {
        const d = sqDist(points[i]!, centroids[c]!)
        if (d < bestD) {
          bestD = d
          best = c
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best
        changed = true
      }
    }
    if (!changed && iter > 0) break

    // Recompute centroids.
    const sums = centroids.map(() => [0, 0] as [number, number])
    const counts = new Array<number>(centroids.length).fill(0)
    for (let i = 0; i < points.length; i++) {
      const a = assignments[i]!
      sums[a]![0] += points[i]![0]
      sums[a]![1] += points[i]![1]
      counts[a]! += 1
    }
    for (let c = 0; c < centroids.length; c++) {
      if (counts[c]! > 0) {
        centroids[c] = [sums[c]![0] / counts[c]!, sums[c]![1] / counts[c]!]
      }
    }
  }

  // Build clusters with exemplar indices (top-5 nearest to centroid).
  const clusters: Cluster[] = centroids.map((c, id) => ({
    id,
    cx: c[0],
    cy: c[1],
    size: 0,
    members: [],
    exemplars: [],
  }))
  for (let i = 0; i < points.length; i++) {
    const c = clusters[assignments[i]!]!
    c.members.push(i)
    c.size += 1
  }
  for (const c of clusters) {
    const scored = c.members
      .map((idx) => ({
        idx,
        d: sqDist(points[idx]!, [c.cx, c.cy]),
      }))
      .sort((a, b) => a.d - b.d)
    c.exemplars = scored.slice(0, 5).map((s) => s.idx)
  }
  return clusters.filter((c) => c.size > 0)
}

/**
 * Ask the LLM for a short topic label per cluster. Sends N parallel
 * calls (one per cluster) to keep latency bounded. Each call ships
 * the top-N exemplar contents for the cluster and asks for a 2-4
 * word title.
 */
export async function labelClusters(
  clusters: Cluster[],
  records: RecordRow[],
  textField: string,
): Promise<Cluster[]> {
  const tasks = clusters.map(async (c) => {
    const samples = c.exemplars
      .map((idx) => records[idx]?.fields[textField])
      .filter((v): v is string => typeof v === 'string')
      .map((s) => s.replace(/\s+/g, ' ').trim().slice(0, 220))
    if (samples.length === 0) return { ...c, label: '—' }
    const prompt =
      'Below are several short text excerpts from one cluster of a ' +
      'multimodal RAG document corpus (mix of Japanese and English). ' +
      'Return a 2-4 word topic label that captures what they share. ' +
      'Reply with the label ONLY — no quotes, no explanation, no ' +
      'period.\n\n' +
      samples.map((s, i) => `${i + 1}. ${s}`).join('\n')
    try {
      const { data } = await llmComplete(
        [
          {
            role: 'system',
            content:
              'You produce short, accurate topic labels. Never reply ' +
              'with full sentences. 2-4 words. No quotes.',
          },
          { role: 'user', content: prompt },
        ],
        { temperature: 0.2, max_tokens: 24 },
      )
      const label = data.content
        .trim()
        .replace(/^["'`]+|["'`.]+$/g, '')
        .replace(/\s+/g, ' ')
        .slice(0, 48)
      return { ...c, label: label || '—' }
    } catch {
      return { ...c, label: undefined }
    }
  })
  return Promise.all(tasks)
}

function sqDist(a: [number, number], b: [number, number]): number {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return dx * dx + dy * dy
}

/** Tiny seeded PRNG so cluster seeding is deterministic across renders. */
function mulberry32(seed: number) {
  let a = seed
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
