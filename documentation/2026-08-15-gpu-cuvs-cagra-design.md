# Making the CAGRA GPU index build, run, and stay built — Design

**Date:** 2026-08-15
**Status:** Approved for implementation
**Closes:** gap 5 of the 2026-08-12 competitive refresh — "`Cagra` unvalidated on real
hardware. The one E2E that tried it recorded `SKIP`."

## What was actually wrong

The refresh recorded CAGRA as *"closed but never validated on hardware"* and called
validating it *"the cheapest credibility win available"*. Investigation on `iitgpu11`
(4× RTX 6000 Ada, 48 GB each, driver 575.51.03) found something different and worse.

**`gpu_cuvs` does not compile.** Six errors in `crates/veda-index/src/gpu_cuvs.rs`:

| error | location | cause |
|---|---|---|
| `E0599` no method `serialize_to_hnswlib` | `:86` | not in `cuvs::cagra::Index` |
| `E0599` no function `Index::deserialize` | `:106` | not in `cuvs::cagra::Index` |
| `E0277` ×4: not `Send`, not `Sync` | `:142`, `:270` | `*mut cuvsCagraIndex` is a raw pointer; `VectorIndex: Send + Sync` |

`cuvs 26.4`'s `cagra::Index` exposes exactly four items — `new`, `build`, `search`,
`Drop`. It has **no serialization at all**; only `vamana::Index` does. The code was
written against an API that does not exist in the pinned version.

**Nobody could have known.** Two independent reasons:

1. **CI never compiles it.** The workflow has `cargo check (veda-index/gpu_nim)` and no
   equivalent for `gpu_cuvs`.
2. **Seven undocumented environment blockers** stand between a developer and a local
   build. The module's own doc comment lists two prerequisites — CUDA toolkit and driver
   ≥ 535. Both were satisfied on this host and the build still failed seven ways:

   | # | blocker | resolution |
   |---|---|---|
   | 1 | `nvcc` present at `/usr/local/cuda/bin` but not on `PATH` | prepend it |
   | 2 | CMake 3.22.1; `cuvs-sys 26.4.0` needs ≥ 3.30.4 | `pip install 'cmake==3.31.*'` |
   | 3 | `libcuvs` absent; `cuvs-sys` would build RAPIDS from source | `libcuvs-cu12==26.4.0` wheel from `pypi.nvidia.com` (conda is unusable here — `repo.anaconda.com` fails TLS verification behind a self-signed corporate CA) |
   | 4 | `cmake-rs` sets `CMAKE_PREFIX_PATH=""`, discarding it | set nine `<Package>_DIR` variables instead |
   | 5 | bindgen cannot find `libclang` | `pip install libclang`, set `LIBCLANG_PATH` |
   | 6 | clang cannot find `cuda_runtime.h` | `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/local/cuda/include` |
   | 7 | clang cannot find `limits.h` (pip `libclang` ships no builtin headers) | add `-I/usr/lib/gcc/x86_64-linux-gnu/12/include` |

So the `SKIP` in the E2E was the only truthful thing that harness could record, and
`IndexType::Cagra` — wired through `types.rs`, `local_shard.rs`, `search.rs`,
`collections.rs` and the advisor — cannot be served by a binary that can be produced.

## Goals

1. `cargo check -p veda-index --features gpu_cuvs` succeeds.
2. A CAGRA index can be built on GPU, searched, persisted, reloaded, and searched again
   with identical results.
3. CI compiles the feature, so this cannot silently regress.
4. The build recipe is written down where the next person will find it.
5. A **recall-qualified** benchmark on real hardware, published.

## Non-goals

- `gpu_nim` (the external-container path) — untouched.
- Query batching, multi-GPU, or any change to how `IndexType::Cagra` is selected/routed.
- Making CI *run* GPU code. CI compiles; the GPU benchmark runs on `iitgpu11`.

## Design

### Component 1 — thread safety

`VectorIndex` requires `Send + Sync` (`crates/veda-index/src/traits.rs:19`). The cuVS
index is a raw pointer and its `Resources` wraps a CUDA stream. The current code holds
**one** `Resources` and reuses it from `search(&self)`, so concurrent queries would share
a single stream — the trait-bound errors are a symptom of a real hazard, not a missing
marker trait.

Both move behind a `Mutex` inside `CagraIndex`:

```rust
#[cfg(feature = "gpu_cuvs")]
struct CagraInner {
    resources: Resources,
    handle: ffi::CagraHandle,
}

// SAFETY: the Mutex provides *exclusive access*; this impl asserts only
// *thread mobility*. CUDA contexts have been process-wide since CUDA 4,
// so the handle and its stream remain valid when the owning value moves
// between threads. Sync is DERIVED from Mutex<T: Send> — never asserted.
unsafe impl Send for CagraInner {}

pub struct CagraIndex {
    metric: DistanceMetric,
    dimension: u32,
    count: usize,
    #[cfg(feature = "gpu_cuvs")]
    inner: std::sync::Mutex<Option<CagraInner>>,
}
```

**Correction to this design, found during implementation.** This section originally
claimed `Send`/`Sync` would follow from the `Mutex` alone, with **no `unsafe impl`**.
That is not achievable and the claim was wrong. `Mutex<T>` is `Send` only when `T: Send`,
and `T` here transitively contains a raw `*mut cuvsCagraIndex`, which is not. Some
`unsafe impl` is unavoidable; the question is only how narrow it can be made.

What ships is the narrowest form: `Send` on the inner struct only. `Sync` — the property
that actually governs concurrent access, and the one an incorrect assertion turns into a
data race — is still **derived** from `Mutex<T: Send>`, never asserted. So the original
intent holds where it matters; only the literal "no `unsafe impl`" claim does not.

This serialises GPU search across threads. That is honest about the hardware — one shared
device whose work already serialises substantially at the stream level — and the cost is
a number Component 5 measures rather than a guess. If contention proves material, a
dedicated GPU worker thread with query batching is the natural follow-up; nothing here
blocks it.

**Poisoning:** a panic while holding the lock poisons it. Map `PoisonError` to
`VedaError::Index` naming the index as unusable, rather than unwrapping.

### Component 2 — persistence via the C API

> **Revision, 2026-08-15, after attempting the approach below.** The design as first
> written is **impossible**. `cuvs::cagra::Index` is `pub struct Index(ffi::cuvsCagraIndex_t)`
> — the field is private, there is no accessor, and `cuvs` does not re-export its `ffi`
> module (`lib.rs` exports only `ManagedTensor`, `Error`, `Result`, `Resources`). So the
> raw `cuvsCagraIndex_t` cannot be obtained from a safe-wrapper index, and
> `cuvsCagraSerialize` cannot be called on one. `cuvs::Resources` *is*
> `Resources(pub ffi::cuvsResources_t)`, so the resources handle is reachable — but that
> alone is not enough.
>
> **Consequence:** it is all-or-nothing. Either use the safe wrapper and have no
> persistence, or own the index handle ourselves. This module now owns the handle for its
> whole lifecycle via `cuvs-sys` — `cuvsCagraIndexCreate`, `cuvsCagraBuild`,
> `cuvsCagraSearch`, `cuvsCagraSerialize`, `cuvsCagraDeserialize`,
> `cuvsCagraIndexDestroy` — and `cuvs-sys` becomes a direct dependency of `veda-index`
> alongside `cuvs` (which still supplies `Resources` and `ManagedTensor`).
>
> This is more unsafe surface than originally scoped. It is taken because the alternative
> abandons persistence entirely, which would leave `IndexType::Cagra` unusable for any
> segment that must survive a restart — i.e. essentially all of them. The containment
> requirement is unchanged and now matters more: every FFI call lives in one private
> submodule behind a safe type, and no `unsafe` escapes it.


`cuvs 26.4`'s Rust wrapper has no CAGRA serialization, but the C API does and `cuvs-sys`
already generates bindings:

```c
cuvsError_t cuvsCagraSerialize(cuvsResources_t, const char* filename,
                               cuvsCagraIndex_t, bool include_dataset);
cuvsError_t cuvsCagraDeserialize(cuvsResources_t, const char* filename,
                                 cuvsCagraIndex_t);
```

Two private wrappers in this module, each with its invariants stated at the call site and
`cuvsError_t` checked:

```rust
/// # Safety
/// `res` and `idx` must be live handles from this module; `path` must be a
/// valid filesystem path. The `CString` outlives the call.
#[cfg(feature = "gpu_cuvs")]
unsafe fn cagra_serialize(res: &Resources, idx: &CuvsCagraIndex, path: &Path) -> Result<()>;

#[cfg(feature = "gpu_cuvs")]
unsafe fn cagra_deserialize(res: &Resources, path: &Path) -> Result<CuvsCagraIndex>;
```

**The `unsafe` does not escape the module.** `save`/`load` on `VectorIndex` are safe and
do what the trait promises, so a Cagra segment survives a restart.

`include_dataset = true`, so a reloaded index can answer queries without the original
vectors. Note in the doc comment that this makes the file roughly dataset-sized.

`serialize_to_hnswlib` is **removed**, not ported. It was a build-accelerator path that
abandons GPU search, and nothing calls it.

### Component 3 — CI

A new job installs the prebuilt wheels (cached on the lockfile hash) and runs:

```
cargo check -p veda-index --features gpu_cuvs
```

**No GPU required** — compilation and linking are exactly what broke. This is the single
change that stops the failure recurring: without it, the next cuVS API drift is invisible
again.

If the wheel install proves too slow or large for CI, the fallback is a scheduled
(non-blocking) job rather than dropping the check — **silently not compiling the feature
is the status quo that produced this**.

### Component 4 — the build recipe

All seven steps go into the module's doc comment (where a developer hits the error) and a
section in `documentation/` (where someone planning the work looks). The doc comment
currently claims two prerequisites and is wrong by omission; it is replaced.

### Component 5 — benchmark

An `#[ignore]` benchmark in `crates/veda-index/tests/`, run manually on `iitgpu11`,
reporting for a synthetic corpus at two sizes:

- CAGRA build time on GPU
- CAGRA search latency, and the same for Veda's existing HNSW on identical data
- **recall@10 against exact brute-force search, for both**

Latency without recall is meaningless for an ANN index. A number is published only
alongside the recall it was measured at.

## Error handling

- Feature off: every method returns `VedaError::Index` naming `gpu_cuvs`, as today.
- `search` before `build`/`load`: `VedaError::Index`, not a panic.
- Non-zero `cuvsError_t`: `VedaError::Index` including the operation and path.
- Poisoned mutex: `VedaError::Index`, never `unwrap`.
- Dimension or metric mismatch on `load`: error naming both values.

## Testing

All GPU tests are `#[cfg(feature = "gpu_cuvs")]` and `#[ignore]`, since CI has no GPU.

1. **Round trip.** build → search → save → load → search, asserting the two result sets
   are identical. This is the test that would have caught the missing serialization API.
2. **Concurrency.** Several threads searching one `Arc<CagraIndex>` concurrently, results
   asserted correct, plus a `assert_send_sync::<CagraIndex>()` compile-time check.

   **Verified by disabling the mechanism** (2026-08-17), rather than asserted: commenting
   out `unsafe impl Send for CagraInner` fails the build with

   ```text
   error[E0277]: `*mut cuvsCagraIndex` cannot be sent between threads safely
     = note: required for `Mutex<Option<CagraInner>>` to implement `Sync`
    19 | pub trait VectorIndex: Send + Sync {
   ```

   which confirms the two things that matter: the `unsafe impl` is load-bearing, and
   `Sync` is **derived** from `Mutex<T: Send>` rather than asserted anywhere. An incorrect
   `Sync` assertion would be a data race instead of a compile error, so it matters that
   the compiler is the one concluding it. The file was restored afterwards and matches
   `HEAD`.
3. **Recall.** *(This item described a test that did not exist. See "Correction: the
   recall test was fictional" below — it exists now, and finding that it did not is what
   uncovered the recall defect.)* CAGRA results compared against exact search on the same
   data, asserting
   recall@10 above a stated floor — so a silently-wrong index cannot pass.
4. **Feature-off behaviour** unchanged: existing tests keep passing untouched.

Plus the standing gates: `cargo fmt --all --check`, and
`cargo clippy -p veda-index --all-targets -- -D warnings` both with and without the
feature.

## Risks

| Risk | Mitigation |
|---|---|
| The `Mutex` serialises GPU search and becomes a throughput ceiling. | Measured, not assumed (Component 5). A GPU worker thread with batching is the follow-up, and this design does not block it. |
| Raw FFI mishandles a handle or path and corrupts memory. | Contained in two functions; invariants stated; `cuvsError_t` checked; exercised by the round-trip test on real hardware. |
| The CI wheel install is slow or large enough to be unwelcome. | Cache on the lockfile hash; fall back to a scheduled non-blocking job rather than dropping the check. |
| `cuvs` changes its API again. | That is precisely what the CI check exists to catch. |
| The environment recipe rots as versions move. | The recipe names exact versions (`cuvs 26.4`, `libcuvs-cu12==26.4.0`, `cmake==3.31.*`) and CI pins the same wheel. |

## Build recipe

The same seven steps are in the module doc comment of
`crates/veda-index/src/gpu_cuvs.rs`, where a developer hits the error. Every one of these
was a separate, real failure on a host that **already had** CUDA 12.9 and driver 575 —
which is why "install CUDA" was not a sufficient prerequisite and the feature went years
without anyone successfully compiling it.

| # | Blocker | Fix |
|---|---|---|
| 1 | `nvcc` installed but not on `PATH` | `export PATH=/usr/local/cuda/bin:$PATH` |
| 2 | `cuvs-sys 26.4.0` needs CMake ≥ 3.30.4; Ubuntu 22.04 ships 3.22 | `pip install 'cmake==3.31.*'` |
| 3 | Without a prebuilt `libcuvs`, `cuvs-sys` builds RAPIDS from source (hours) | `pip install --extra-index-url https://pypi.nvidia.com 'libcuvs-cu12==26.4.0'` |
| 4 | `cmake-rs` sets `CMAKE_PREFIX_PATH=""` (rust-lang/cmake-rs#111), so the normal way to point at a prebuilt library silently does nothing | Set nine `<Package>_DIR` vars instead: `cuvs_DIR`, `raft_DIR`, `rmm_DIR`, `rapids_logger_DIR`, `nvtx3_DIR`, `CCCL_DIR`, `Thrust_DIR`, `CUB_DIR`, `libcudacxx_DIR` |
| 5 | bindgen needs libclang | `pip install libclang`; `LIBCLANG_PATH=<site-packages>/clang/native` |
| 6 | clang cannot find `cuda_runtime.h` | `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/local/cuda/include` |
| 7 | the pip `libclang` wheel ships no builtin headers, so clang cannot find `limits.h` | add `-I/usr/lib/gcc/x86_64-linux-gnu/<ver>/include` to `BINDGEN_EXTRA_CLANG_ARGS` |

Two further items are needed to **run** rather than compile, and are invisible to
`cargo check`:

8. `RUSTFLAGS="-L native=<site-packages>/libcuvs/lib64 -L native=/usr/local/cuda/lib64"`,
   or the link step cannot find `libcuvs_c`.
9. `LD_LIBRARY_PATH` must include the `lib64` directories of `libcuvs`, `libraft`,
   `librmm` and `rapids_logger`.

Because 8 and 9 are link/runtime-only, **`cargo check --features gpu_cuvs` passes while
`cargo test` still fails**. This is why the CI job runs `cargo build --tests` rather than
`cargo check`; a `check`-based job would have reported success on a feature that could not
link.

On `iitgpu11` the provisioned environment lives at `/home/mac/veda-cuvs-venv` and is
disposable — `rm -rf` it and re-run steps 2, 3 and 5 to rebuild.

## Correction: the recall test was fictional

The Testing section above claimed a recall test that "a silently-wrong index cannot
pass." No such test existed. What existed planted an exact duplicate of the query in the
corpus and asserted it ranked first, at n=1000 / dim=32 — which a badly degraded graph
passes anyway, because the query is itself a node in the graph and trivially reachable.

Six GPU tests were green while CAGRA's measured recall@10 on a clustered 1M corpus was
**0.11**. The feature would have shipped returning fast wrong answers.

Three separate defects were behind that number, each found by measurement rather than
inspection:

| Fix | recall@10 at 1M x 128 |
|---|---|
| as found | 0.1145 |
| `itopk_size` actually set from the caller's `ef` | 0.4000 |
| `search_width` widened (cuVS defaults it to 1) | 0.8050 |
| `graph_degree` 64 -> 128, intermediate 128 -> 256 | **0.9300** |

`cuvsCagraSearchParams` was created and never configured, so every query ran at the
library default of 64 retained candidates regardless of `k`, of the caller's budget, or
of corpus size — a fixed budget explores a shrinking fraction of a growing graph.
`CagraIndex` also never overrode `search_with_ef`, so it was the only graph index in the
workspace silently discarding the caller's traversal budget.

**Two things about the test corpus mattered more than the assertions**, and both are the
real lesson:

1. **Uniform-random vectors cannot express this test.** In 128 dimensions they are all
   nearly equidistant, so the true top-10 is close to arbitrary and *every* graph index
   scores near chance — HNSW measured 0.51 and CAGRA 0.37 on the same uniform data, which
   made a broken index look merely typical. `scale-benchmark-results.md` had already
   recorded this effect (HNSW at 0.38 on uniform, 1.0 on a real clustered corpus). The
   corpus is now clustered.
2. **Dimension mattered.** The test first ran at dim=64 and passed at 0.9641 while the
   benchmark's dim=128 case measured 0.805 — green in the test, failing in the
   configuration the test existed to protect. The test now runs at dim=128 and n=1M,
   because the defect was invisible at 50k: every `ef` setting looked healthy there.

The test also now asserts that recall *improves with `ef`*. That is the property that was
broken, and a floor alone would not have caught it.

### The floor is a recall target, not a smoke alarm

Measured on the test's corpus once it matched the benchmark's shape, at 1M x 128 with the
final build degrees:

| ef | itopk | width | recall@10 |
|---|---|---|---|
| 64 | 64 | 1 | 0.1719 |
| 128 | 128 | 2 | 0.3125 |
| 256 | 256 | 4 | 0.5156 |
| 512 | 512 | 8 | 0.7656 |
| 768 | 512 | 12 | 0.8438 |
| **1024** (default) | 512 | 16 | **0.8750** |

The floor is **0.80** — roughly 5.7 standard errors below the healthy 0.8750, loose enough
not to flake and tight enough to catch every regression this branch fixed. Each of the three
defects lands below it on this corpus: an unset `itopk_size` far below, `search_width` left
at 1 gives 0.1719, and reverting the denser graph puts the same search budget against a
sparser graph at roughly the 0.766 mark. That is the property the earlier, easier corpus
could not provide — at saturation, any floor below 1.0 only detects collapse.

## Measured result

Clustered synthetic corpus (1,000 centroids, sigma=0.10), 128 dimensions, k=10, mean over
200 queries after a discarded warm-up, release build, one RTX 6000 Ada. Ground truth is
exact brute force over the same vectors. **Nothing else was running on the host**, for
reasons given below.

| n | index | build | search (mean) | recall@10 |
|---|---|---|---|---|
| 100k | CAGRA (GPU) | **16.43 s** | 1018.1 us | **1.0000** |
| 100k | HNSW (CPU) | 171.21 s | **549.8 us** | 0.9900 |
| 1M | CAGRA (GPU) | **86.18 s** | **878.4 us** | **0.9300** |
| 1M | HNSW (CPU) | 4184.40 s | 4488.1 us | 0.9130 |

**Does CAGRA beat HNSW on this hardware at comparable recall?** At 1M, yes, on all three
axes: ~49x faster build, ~5.1x faster search, at slightly better recall. At 100k, no, not
on search: HNSW is ~1.85x faster per query, while CAGRA still builds ~10x faster and
edges recall.

The reason is the more useful finding than either ratio. **CAGRA's per-query latency is
flat in corpus size** — 1018 us at 100k, 878 us at 1M — because it is dominated by kernel
launch and host-to-device transfer, not by graph traversal. HNSW's grows with the corpus
(549.8 us to 4488.1 us, 8.2x for 10x the data). So there is a crossover somewhere between
these two sizes, and its position is set by fixed GPU overhead rather than by either
algorithm being better. On small collections, use HNSW.

### What these numbers do not say

- **Per-query latency is the worst case for a GPU.** The `VectorIndex` trait searches one
  query at a time, so every CAGRA search pays launch and transfer overhead that batching
  would amortise. Batched throughput is where CAGRA should win far more decisively, and
  the trait shape cannot express it. The flat-in-n latency above is direct evidence that
  overhead, not traversal, is the cost.
- **The HNSW build baseline is single-threaded.** `HnswIndex::build` is a sequential
  `for id in 0..len` insert loop, so "49x faster build" is against an unparallelised
  baseline, not against HNSW's best achievable.
- **Wall-clock timings carry real error bars; recall does not.** Earlier runs shared the
  host with other work, and the same HNSW 1M build measured 4313 s and 7996 s — an 85%
  swing — while its recall came out at exactly 0.9270 both times. An earlier contended run
  also reported HNSW 100k search at 1247.9 us against the 549.8 us above, i.e. inflated
  2.3x, which briefly made CAGRA look ~1.5x faster at 100k when it is in fact ~1.85x
  slower. Treat build and latency ratios as approximate; treat recall as reliable.
- **Recall is a property of a corpus, not of an index.** This corpus measures 0.9300 and
  the recall test's measures 0.8750. Neither number transfers to real embeddings.

  The test's corpus now uses the **same shape** as this one — 1,000 centroids, two-sided
  noise at 0.10. It previously used 500 centroids with *one-sided* noise at 0.05, which was
  easier in two ways the constants did not reveal (the one-sided noise offset each
  cluster's true centre and cut the effective per-dimension spread to ~0.0144, so the
  earlier "sigma=0.05" wording here was a misnomer). A healthy index **saturated at 1.0**
  on it, which meant the floor could only detect total collapse: a partial regression to
  0.9 would have passed. Matching the shape is what turns the floor into a recall target.

  The test still reads slightly lower than this benchmark (0.8750 vs 0.9300) for two
  reasons: it draws 64 queries rather than 200, so one standard error is ~0.013; and it
  spreads its queries across all 1,000 clusters whereas the benchmark's `i % CLUSTERS` only
  ever reaches the first 200. The test is therefore marginally the harder measurement,
  which is the right direction for a guard.

- **HNSW's 1M baseline is 0.9130, and only that.** Earlier revisions of this document and
  of the module comments quoted 0.936 and 0.9270 as well, from runs that shared the host
  with other work. CAGRA's 0.9300 is better than 0.9130 but *worse* than the other two, so
  the comparison inverted depending on which number a reader picked up. Every recall figure
  now names the corpus and the run it came from; 0.9130 is the clean-host measurement and
  supersedes the others.
- **`itopk_size` has a hard ceiling of 512** — 1024 fails with a bare `CUVS_ERROR` — so
  `k > 512` is rejected up front, and CAGRA cannot trade further latency for recall beyond
  the settings above without denser build parameters again. `ef` therefore **saturates**:
  any `ef >= 1024` produces identical search parameters (`itopk_size = 512`,
  `search_width = 16`), so raising it further buys nothing and reports nothing.

  Note also that the 512 ceiling was established when `search_width` was 1 and
  `graph_degree` 64. CAGRA's internal result buffer scales as
  `itopk_size + search_width * graph_degree`, which those changes took from 576 to 2560, so
  the ceiling is a property of the *combination*, not of `itopk_size` alone. The current
  settings are measured to work; anyone raising the degrees or the width cap again should
  expect to re-establish it.

- **Small indexes do not get these parameters at all.** `intermediate_graph_degree = 256`
  is undefined for a corpus of fewer than 257 vectors, so the build degrees are clamped
  against the corpus size. A segment below that threshold is built with reduced degrees and
  the recall above does not describe it. Sealed segments in this database are routinely that
  small.

- **Serialized indexes carry no record of their build degrees.** An index saved before the
  degrees were raised reloads without complaint and is then searched with the current, wider
  search parameters against a sparser graph — the 0.805 row in the fix table above. Nothing
  detects the mismatch. Rebuild rather than reload where the numbers matter.
