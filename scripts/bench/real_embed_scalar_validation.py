"""
Real-embedding scalar-quantizer recall@10 validation.

Reads BGE-M3 embeddings (d=1024) from Veda sealed-segment .vec files,
then measures recall@10 for per-dim 8-bit and 4-bit scalar quantizers.

Format of .vec files:
  bytes  0-3:  magic 'VEDA'
  bytes  4-7:  version (u32 LE) = 1
  bytes  8-11: dim    (u32 LE)
  bytes 12-15: count  (u32 LE)
  bytes 16-31: reserved (zeros)
  bytes 32+:   count * dim * f32 LE
"""
import struct
import numpy as np

VEC_FILES = [
    "/home/mac/test/project-mdb/veda-data-e2e/default__rag_anything/shard_0/segment_5302e96c-e60a-45b4-ab46-283931ff657c/vectors/embedding.vec",
    "/home/mac/test/project-mdb/veda-data-e2e/default__rag_anything/shard_1/segment_be375a64-0a56-4b50-b20d-e9e4cc0ce0ad/vectors/embedding.vec",
]

HEADER_SIZE = 32
N_QUERIES = 100
K = 10
RNG_SEED = 42


def load_vec(path):
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)
    magic = header[:4]
    assert magic == b'VEDA', f'bad magic: {magic}'
    dim = struct.unpack_from('<I', header, 8)[0]
    count = struct.unpack_from('<I', header, 12)[0]
    data = np.fromfile(path, dtype=np.float32, count=count * dim, offset=HEADER_SIZE)
    return data.reshape(count, dim)


def cosine_sim_matrix(a, b):
    """a: (m, d), b: (n, d) -> (m, n) cosine similarities."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def scalar_quantize(X, bits):
    """Per-dimension scalar quantization.

    bits=8 -> 256 levels, bits=4 -> 16 levels.
    Returns reconstructed float32 matrix (same shape as X).
    """
    levels = (1 << bits) - 1  # 255 or 15
    vmin = X.min(axis=0)       # (d,)
    vmax = X.max(axis=0)       # (d,)
    scale = vmax - vmin
    scale[scale == 0] = 1.0    # avoid div-by-zero for constant dims

    # Quantize to integer bins [0, levels]
    X_scaled = (X - vmin) / scale * levels
    X_quant = np.round(X_scaled).astype(np.int32)
    X_quant = np.clip(X_quant, 0, levels)

    # Reconstruct
    X_recon = X_quant.astype(np.float32) / levels * scale + vmin
    return X_recon


def recall_at_k(gt_indices, approx_indices, k):
    """gt_indices, approx_indices: (n_queries, k) arrays of neighbor ids."""
    hits = 0
    for i in range(len(gt_indices)):
        gt_set = set(gt_indices[i, :k])
        approx_set = set(approx_indices[i, :k])
        hits += len(gt_set & approx_set)
    return hits / (len(gt_indices) * k)


def top_k_indices(sim_matrix, k, exclude_self=True):
    """sim_matrix: (n_queries, n_corpus) -> (n_queries, k) indices of top-k."""
    if exclude_self:
        np.fill_diagonal(sim_matrix, -np.inf)
    return np.argsort(-sim_matrix, axis=1)[:, :k]


def main():
    print('Loading embeddings ...')
    parts = [load_vec(p) for p in VEC_FILES]
    corpus = np.concatenate(parts, axis=0)
    n, d = corpus.shape
    print(f'Loaded: n={n} vectors, d={d} dimensions (f32)')

    rng = np.random.default_rng(RNG_SEED)
    query_idx = rng.choice(n, size=N_QUERIES, replace=False)
    queries = corpus[query_idx]  # (100, d)

    print(f'Computing exact cosine ground-truth (top-{K}) for {N_QUERIES} queries ...')
    sim_exact = cosine_sim_matrix(queries, corpus)
    # Zero out the query vectors themselves so we don't retrieve self
    for i, qi in enumerate(query_idx):
        sim_exact[i, qi] = -np.inf
    gt = np.argsort(-sim_exact, axis=1)[:, :K]

    print('Quantizing to 8-bit ...')
    corpus_8bit = scalar_quantize(corpus, bits=8)
    queries_8bit = scalar_quantize(queries, bits=8)
    sim_8bit = cosine_sim_matrix(queries_8bit, corpus_8bit)
    for i, qi in enumerate(query_idx):
        sim_8bit[i, qi] = -np.inf
    approx_8bit = np.argsort(-sim_8bit, axis=1)[:, :K]

    print('Quantizing to 4-bit ...')
    corpus_4bit = scalar_quantize(corpus, bits=4)
    queries_4bit = scalar_quantize(queries, bits=4)
    sim_4bit = cosine_sim_matrix(queries_4bit, corpus_4bit)
    for i, qi in enumerate(query_idx):
        sim_4bit[i, qi] = -np.inf
    approx_4bit = np.argsort(-sim_4bit, axis=1)[:, :K]

    r8 = recall_at_k(gt, approx_8bit, K)
    r4 = recall_at_k(gt, approx_4bit, K)

    print()
    print('=== Recall@10 results (real BGE-M3 embeddings, d=%d, n=%d) ===' % (d, n))
    print('  8-bit scalar quant (256 levels, per-dim):  %.4f' % r8)
    print('  4-bit scalar quant ( 16 levels, per-dim):  %.4f' % r4)
    print()
    print('(queries=%d, corpus=%d, k=%d, seed=%d)' % (N_QUERIES, n, K, RNG_SEED))


if __name__ == '__main__':
    main()
