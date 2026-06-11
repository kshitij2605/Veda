"""turbovec vs Veda RaBitQ head-to-head benchmark.

Run on the build host:
  python3 -m venv /tmp/tvenv && /tmp/tvenv/bin/pip install turbovec numpy
  /tmp/tvenv/bin/python scripts/bench/turbovec_compare.py

Mirrors veda-index/tests/scale_benchmark.rs::vec_of for an IDENTICAL corpus
so numbers are directly comparable (same distribution, same queries).

Veda baseline (d=1536, n=100k, recall@10):
  rabitq (1-bit)         = 0.331
  rabitq+rerank          = 0.749
  rabitq+extrerank       = 0.734
"""

import time
import sys
import numpy as np


def vec_of(seed: int, dim: int) -> np.ndarray:
    """Reference scalar implementation — mirrors scale_benchmark.rs::vec_of.
    Use corpus() for bulk generation; this is kept for validation.
    """
    s = (seed * 2654435761) & 0xFFFFFFFF
    out = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = ((s >> 16) / 32768.0) - 1.0
    return out


def corpus(n: int, dim: int, chunk: int = 10_000) -> np.ndarray:
    """Vectorised LCG corpus — same values as vec_of(), ~12s for n=100k, d=1536.

    Processes rows in chunks to keep peak memory bounded (~chunk * dim * 4 bytes).
    LCG: s_{k+1} = (1103515245 * s_k + 12345) & 0xFFFFFFFF
    Seed per row: s_0[i] = (i * 2654435761) & 0xFFFFFFFF
    """
    a = np.uint32(1103515245)
    c = np.uint32(12345)
    out = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        s = np.arange(start, end, dtype=np.uint32) * np.uint32(2654435761)
        for i in range(dim):
            s = s * a + c
            out[start:end, i] = (s >> np.uint32(16)).astype(np.float32) / 32768.0 - 1.0
    return out


def exact_top_k_batched(mat_norm: np.ndarray, query_rows: np.ndarray, k: int) -> list:
    """Batch exact cosine top-k via a single (N x nq) matmul."""
    # sims[n, q] = cosine(mat_norm[n], query_rows[q])
    sims = mat_norm @ query_rows.T          # (N, nq)
    truths = []
    for q in range(sims.shape[1]):
        col = sims[:, q]
        truths.append(set(np.argpartition(-col, k)[:k].tolist()))
    return truths


def main():
    try:
        from turbovec import TurboQuantIndex
    except ImportError as e:
        print(f"ERROR: could not import turbovec: {e}", file=sys.stderr)
        sys.exit(1)

    N = 100_000
    K = 10
    NQ = 100   # query count

    for dim in (1536,):
        print(f"\n=== turbovec vs Veda RaBitQ  dim={dim}  n={N}  k={K}  nq={NQ} ===")

        # --- Build corpus ---
        t0 = time.time()
        mat = corpus(N, dim)
        corpus_ms = (time.time() - t0) * 1000
        print(f"  corpus generation : {corpus_ms / 1000:.1f} s")

        # L2-normalise for cosine metric (identical to Veda bench)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_norm = np.ascontiguousarray(mat / (norms + 1e-9))

        # Query indices: deterministic spread
        qidx = [(i * 7919) % N for i in range(NQ)]
        Q = mat_norm[qidx]           # (NQ, dim) — batch query matrix

        # --- Exact ground truth (batched matmul, ~2 s) ---
        t0 = time.time()
        truths = exact_top_k_batched(mat_norm, Q, K)
        gt_ms = (time.time() - t0) * 1000
        print(f"  exact ground truth: {gt_ms / 1000:.1f} s")

        # turbovec expects normalised float32 C-contiguous input
        Q_c = np.ascontiguousarray(Q)

        results = {}
        for bits in (2, 4):
            idx = TurboQuantIndex(dim=dim, bit_width=bits)

            # --- Build + warm up ---
            t0 = time.time()
            idx.add(mat_norm)
            idx.prepare()            # amortises SIMD cache init, rotation, Lloyd-Max
            build_ms = (time.time() - t0) * 1000

            # --- Per-query search (measure individual latency) ---
            latencies_ms = []
            recalls = []
            for j in range(NQ):
                q = Q_c[j : j + 1]  # shape (1, dim) — TurboQuantIndex requires 2-D
                t0 = time.time()
                _scores, inds = idx.search(q, K)
                latencies_ms.append((time.time() - t0) * 1e3)
                found = set(inds.ravel().tolist())
                recalls.append(len(found & truths[j]) / K)

            latencies_ms.sort()
            p50 = latencies_ms[len(latencies_ms) // 2]
            p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
            mean_recall = sum(recalls) / len(recalls)
            results[bits] = mean_recall

            print(
                f"  turbovec bits={bits}  build={build_ms / 1000:6.1f} s"
                f"  p50={p50:7.3f} ms  p95={p95:7.3f} ms"
                f"  recall@10={mean_recall:.3f}"
            )

        # --- Comparison table ---
        print()
        print("  ┌────────────────────────────────────┬───────────┐")
        print("  │ Method                             │ recall@10 │")
        print("  ├────────────────────────────────────┼───────────┤")
        print(f"  │ turbovec  2-bit                    │   {results.get(2, '?'):.3f}   │")
        print(f"  │ turbovec  4-bit                    │   {results.get(4, '?'):.3f}   │")
        print("  ├────────────────────────────────────┼───────────┤")
        print("  │ Veda rabitq       (1-bit)          │   0.331   │")
        print("  │ Veda rabitq+rerank                 │   0.749   │")
        print("  │ Veda rabitq+extrerank              │   0.734   │")
        print("  └────────────────────────────────────┴───────────┘")
        print()
        r2 = results.get(2, 0)
        r4 = results.get(4, 0)
        veda_rerank = 0.749
        delta2 = r2 - veda_rerank
        delta4 = r4 - veda_rerank
        print(
            f"  turbovec 2-bit vs Veda +rerank: {delta2:+.3f}  "
            f"({'better' if delta2 > 0 else 'worse'})"
        )
        print(
            f"  turbovec 4-bit vs Veda +rerank: {delta4:+.3f}  "
            f"({'better' if delta4 > 0 else 'worse'})"
        )


if __name__ == "__main__":
    main()
