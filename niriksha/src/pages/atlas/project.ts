import { pcaTwoDim } from './pca'

export type ProjectionMethod = 'pca' | 'umap'

export interface ProjectHandle {
  promise: Promise<[number, number][]>
  /** Terminate an in-flight UMAP worker (no-op for the PCA path). */
  cancel: () => void
}

/**
 * Project D-dimensional vectors to 2-D for the Atlas.
 *
 * PCA runs inline (power iteration — milliseconds). UMAP runs in a Web
 * Worker so its SGD epochs never block the UI; `onProgress(epoch, total)`
 * fires as epochs complete. Samples too small to form a neighbour graph
 * (<5 points) fall back to PCA.
 */
export function projectAsync(
  method: ProjectionMethod,
  vectors: number[][],
  onProgress?: (epoch: number, total: number) => void,
): ProjectHandle {
  if (method === 'pca' || vectors.length < 5) {
    return { promise: Promise.resolve(pcaTwoDim(vectors)), cancel: () => {} }
  }

  const worker = new Worker(new URL('./umapWorker.ts', import.meta.url), {
    type: 'module',
  })
  // umap-js requires nNeighbors < n; clamp to a sane window for small sets.
  const nNeighbors = Math.max(2, Math.min(15, vectors.length - 1))

  const promise = new Promise<[number, number][]>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data as
        | { type: 'progress'; epoch: number; total: number }
        | { type: 'done'; coords: number[][] }
        | { type: 'error'; message: string }
      if (msg.type === 'progress') {
        onProgress?.(msg.epoch, msg.total)
      } else if (msg.type === 'done') {
        resolve(msg.coords as [number, number][])
        worker.terminate()
      } else {
        reject(new Error(msg.message))
        worker.terminate()
      }
    }
    worker.onerror = (e) => {
      reject(new Error(e.message || 'UMAP worker failed'))
      worker.terminate()
    }
    worker.postMessage({ vectors, nNeighbors, minDist: 0.1 })
  })

  return { promise, cancel: () => worker.terminate() }
}
