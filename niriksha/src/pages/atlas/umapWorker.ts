import { UMAP } from 'umap-js'

/**
 * UMAP projection worker. Runs the SGD epochs off the main thread so the
 * Atlas stays interactive while a few thousand embeddings are laid out.
 * Posts `progress` as epochs complete, then `done` with the 2-D embedding
 * (or `error` if UMAP throws).
 *
 * Typed against the DOM `self` (the project lib has DOM, not WebWorker);
 * at runtime this module only ever executes inside a dedicated worker,
 * where `onmessage` / `postMessage` resolve to the worker globals.
 */
interface UmapRequest {
  vectors: number[][]
  nNeighbors: number
  minDist: number
}

self.onmessage = (e: MessageEvent) => {
  const { vectors, nNeighbors, minDist } = e.data as UmapRequest
  try {
    const umap = new UMAP({ nComponents: 2, nNeighbors, minDist })
    const total = umap.initializeFit(vectors)
    for (let i = 0; i < total; i++) {
      umap.step()
      if (i % 10 === 0 || i === total - 1) {
        self.postMessage({ type: 'progress', epoch: i + 1, total })
      }
    }
    self.postMessage({ type: 'done', coords: umap.getEmbedding() })
  } catch (err) {
    self.postMessage({
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    })
  }
}
