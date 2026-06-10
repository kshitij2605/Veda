import { describe, it, expect } from 'vitest'
import { projectAsync } from '../project'

/**
 * The UMAP path runs in a Web Worker (jsdom has none), so these tests
 * cover the synchronous PCA + small-sample-fallback branches. The live
 * UMAP projection is verified end-to-end via Playwright on real data.
 */
describe('projectAsync', () => {
  it('projects with PCA to one [x, y] per input vector', async () => {
    const vectors = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [1, 1, 0, 0],
      [0, 1, 1, 0],
    ]
    const coords = await projectAsync('pca', vectors).promise
    expect(coords).toHaveLength(vectors.length)
    for (const c of coords) {
      expect(c).toHaveLength(2)
      expect(Number.isFinite(c[0])).toBe(true)
      expect(Number.isFinite(c[1])).toBe(true)
    }
  })

  it('falls back to PCA when the sample is too small for a UMAP neighbour graph', async () => {
    // method is 'umap' but n < 5 → PCA fallback, so it resolves without
    // constructing a Worker (which jsdom lacks).
    const coords = await projectAsync('umap', [
      [1, 0],
      [0, 1],
      [1, 1],
    ]).promise
    expect(coords).toHaveLength(3)
    expect(coords[0]).toHaveLength(2)
  })
})
