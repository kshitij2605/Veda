//! In-binary GPU-accelerated CAGRA index via NVIDIA cuVS.
//!
//! Companion to [`crate::gpu_nim_client`]. The NIM client offloads to an
//! external container; this module links cuVS directly so the Veda binary
//! itself builds and searches CAGRA on the GPU.
//!
//! Default `cargo build` does not enable `gpu_cuvs` and stays CUDA-free;
//! the no-op `CagraIndex` below still compiles and returns actionable
//! errors when called.
//!
//! ## Why this module owns a raw handle
//!
//! `cuvs::cagra::Index` is `pub struct Index(ffi::cuvsCagraIndex_t)` — the
//! field is private, there is no accessor, and `cuvs` does not re-export
//! its `ffi` module. The CAGRA **C** API has serialization
//! (`cuvsCagraSerialize` / `cuvsCagraDeserialize`) but the Rust wrapper
//! exposes none of it: `cagra::Index` has only `new`, `build`, `search`
//! and `Drop`. Persisting an index therefore requires owning the handle
//! ourselves, which is what the private [`ffi`] submodule below does.
//!
//! **All `unsafe` in this file lives in that submodule.** Nothing outside
//! it may contain an `unsafe` block.
//!
//! ## Build prerequisites (`--features veda-index/gpu_cuvs`)
//!
//! This feature needs considerably more than a CUDA toolkit. Every item
//! below was a real, separate failure on a host that already had CUDA 12.9
//! and driver 575 installed:
//!
//! 1. **`nvcc` on `PATH`** — often installed at `/usr/local/cuda/bin` but
//!    not exported. `export PATH=/usr/local/cuda/bin:$PATH`.
//! 2. **CMake ≥ 3.30.4** — `cuvs-sys 26.4.0` requires it; Ubuntu 22.04
//!    ships 3.22. `pip install 'cmake==3.31.*'`.
//! 3. **A prebuilt `libcuvs`** — otherwise `cuvs-sys` builds RAPIDS from
//!    source, which takes hours. `pip install --extra-index-url
//!    https://pypi.nvidia.com 'libcuvs-cu12==26.4.0'`.
//! 4. **Nine `<Package>_DIR` variables** — `cmake-rs` sets
//!    `CMAKE_PREFIX_PATH=""` before invoking CMake
//!    (rust-lang/cmake-rs#111), so the usual way of pointing at a prebuilt
//!    library does not work. Set `cuvs_DIR`, `raft_DIR`, `rmm_DIR`,
//!    `rapids_logger_DIR`, `nvtx3_DIR`, `CCCL_DIR`, `Thrust_DIR`,
//!    `CUB_DIR` and `libcudacxx_DIR` to the corresponding
//!    `.../lib64/cmake/<pkg>` directories instead.
//! 5. **`LIBCLANG_PATH`** — bindgen needs libclang. `pip install libclang`
//!    and point at `site-packages/clang/native`.
//! 6. **CUDA headers for bindgen** —
//!    `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/local/cuda/include`, or clang cannot
//!    find `cuda_runtime.h`.
//! 7. **Compiler builtin headers** — the pip `libclang` wheel ships no
//!    builtin headers, so add `-I/usr/lib/gcc/x86_64-linux-gnu/<ver>/include`
//!    to `BINDGEN_EXTRA_CLANG_ARGS` or clang cannot find `limits.h`.
//!
//! To **run** (not merely compile), `LD_LIBRARY_PATH` must include the
//! `lib64` directories of `libcuvs`, `libraft`, `librmm` and
//! `rapids_logger`.
//!
//! ## Running the GPU tests
//!
//! ```text
//! cargo test --release -p veda-index --features gpu_cuvs --lib \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is required, not preferred. These tests share one
//! device, and the recall test builds a **1,000,000 x 128** index; run in
//! parallel with the others it fails intermittently on device contention
//! while passing in isolation. `--release` matters too — the recall test
//! brute-forces its own ground truth over that corpus.
//!
//! Budget roughly 4-5 minutes and ~2 GB of host memory for the recall
//! test alone: the corpus and its noise buffer are held simultaneously,
//! `build` stages another copy, and ground truth is 64 full passes over
//! 1M vectors.
//!
//! ## A note on serialized indexes
//!
//! `save`/`load` record the vector count but **not** the build degrees.
//! An index serialized before the degrees were raised will load without
//! complaint and then be searched with the current, wider search
//! parameters against a sparser graph — the 0.805 row in the design
//! doc's fix table. Nothing detects this, so the recall figures here
//! apply only to indexes built after that change. Rebuild rather than
//! reload if the numbers matter.
//!
//! See `documentation/2026-08-15-gpu-cuvs-cagra-design.md` for the full
//! recipe and the measured benchmark.

use std::path::Path;

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::traits::{SearchResult, VectorIndex};

#[cfg(feature = "gpu_cuvs")]
use cuvs::{ManagedTensor, Resources};
#[cfg(feature = "gpu_cuvs")]
use ndarray::{Array2, s};

/// Traversal budget used when a caller does not supply one.
///
/// CAGRA needs a far larger budget than HNSW for the same recall.
///
/// Measured on the recall test's 1M x 128 corpus (1,000 centroids,
/// two-sided noise at 0.10 — the same shape the benchmark uses), with the
/// final build degrees: recall@10 was 0.1719 at ef=64, 0.3125 at 128,
/// 0.5156 at 256, 0.7656 at 512, 0.8438 at 768 and **0.8750 at 1024**.
/// The benchmark's own corpus measures 0.9300 at the same default, against
/// HNSW's 0.9130 there.
///
/// The test reads lower than the benchmark despite matching its shape,
/// for two reasons worth knowing: it draws 64 queries rather than 200 (so
/// one standard error is ~0.013), and it spreads those queries across all
/// 1,000 clusters where the benchmark's `i % CLUSTERS` only ever reaches
/// the first 200. The test is therefore slightly the harder measurement.
///
/// Recall is a property of a corpus, not of an index, so every figure in
/// this module names the corpus it came from. Earlier revisions of these
/// comments quoted three mutually inconsistent HNSW baselines (0.936,
/// 0.9270, 0.9130) drawn from runs that shared the host with other work.
/// **0.9130 is the clean-host number** and the only one to compare
/// against.
///
/// Small corpora are not a useful guide here: at 50k, ef=512 already
/// reached recall 1.0 and every value looked fine. The budget has to be
/// chosen against the scale where the graph is actually hard to
/// traverse.
///
/// Defaulting lower would mean shipping an index that silently returns
/// wrong answers, which is the state this module was found in. The
/// latency this costs is measured and published in the benchmark rather
/// than avoided by quietly lowering the default.
#[cfg(feature = "gpu_cuvs")]
const DEFAULT_EF: usize = 1024;

/// Bounds on `itopk_size`.
///
/// The ceiling is empirical, not from the docs: `itopk_size = 1024` makes
/// `cuvsCagraSearch` fail with a bare `CUVS_ERROR` on this build. 512 is
/// the largest value that works.
#[cfg(feature = "gpu_cuvs")]
const ITOPK_MIN: usize = 32;
#[cfg(feature = "gpu_cuvs")]
const ITOPK_MAX: usize = 512;

/// Output graph degree, and the working degree used while pruning it.
///
/// cuVS defaults these to 64 / 128. Search-side tuning alone plateaued
/// below HNSW at scale — at 1M x 128 dims, with both search knobs at
/// their ceilings, recall@10 was 0.805 on the benchmark corpus against
/// HNSW's 0.9130 there — because a
/// sparser graph has fewer edges to traverse no matter how wide the
/// frontier. A denser graph costs build time and index memory, both of
/// which CAGRA has in hand: its build measured ~49x faster than HNSW's
/// on the same corpus.
///
/// These are upper bounds, not absolutes: a k-NN graph cannot have a
/// degree at or above its own vertex count, so [`graph_degrees_for`]
/// clamps them against the corpus size.
#[cfg(feature = "gpu_cuvs")]
const GRAPH_DEGREE: usize = 128;
#[cfg(feature = "gpu_cuvs")]
const INTERMEDIATE_GRAPH_DEGREE: usize = 256;

/// Clamp the build degrees against the corpus size.
///
/// `INTERMEDIATE_GRAPH_DEGREE` is 256, which is undefined for a corpus of
/// fewer than 257 vectors — and this module's own tests build corpora of
/// 200 and 500. It is not a test-only concern: a vector database seals
/// small segments routinely, so any segment below ~257 vectors reaches
/// this path in production. Rather than depend on whatever cuVS does with
/// an over-large degree (it is not documented, and "silently clamps" and
/// "fails with a bare CUVS_ERROR" are both plausible), state the clamp
/// here.
///
/// Note the consequence: the recall figures quoted throughout this module
/// were measured at the full degrees, so they do **not** describe a small
/// index built with reduced ones.
#[cfg(feature = "gpu_cuvs")]
fn graph_degrees_for(n: usize) -> (usize, usize) {
    let intermediate = INTERMEDIATE_GRAPH_DEGREE.min(n.saturating_sub(1)).max(1);
    let graph = GRAPH_DEGREE.min(intermediate);
    (graph, intermediate)
}

/// Number of graph nodes expanded per traversal iteration.
///
/// cuVS defaults this to 1, which expands a single frontier node per
/// step. `itopk_size` alone was not enough to make CAGRA correct at
/// scale: with `itopk_size` pinned at its 512 ceiling, recall@10 on a
/// clustered 1M x 128 corpus was 0.40 at width 1, 0.8391 at width 8 and
/// 0.9641 at width 16 — those three at the *original* build degrees, which
/// is why they differ from the `DEFAULT_EF` figures above. It is derived
/// from the same `ef` budget so callers still have a single dial to turn.
///
/// Capped at 16 because that is where CAGRA reached HNSW's recall on the
/// measured corpus; wider costs latency for accuracy already in hand.
#[cfg(feature = "gpu_cuvs")]
fn search_width_for(ef: usize) -> usize {
    (ef / 64).clamp(1, 16)
}

/// Map a caller's `ef` budget onto a legal `itopk_size`.
///
/// Rounded to a **power of two**, not to cuVS's documented multiple of
/// 32. cuVS rounds the value down internally: `itopk_size = 96` produced
/// recall bit-identical to 64, so a 32-granularity knob silently lies
/// about the budget it is applying. Powers of two are the granularity
/// the library actually honours.
///
/// Note that `ef` saturates: any `ef` at or above 1024 yields the same
/// `itopk_size = 512` (the hard ceiling) and the same `search_width = 16`
/// (the cap), so raising `ef` past 1024 buys nothing.
#[cfg(feature = "gpu_cuvs")]
fn itopk_for(k: usize, ef: usize) -> usize {
    // Clamp BEFORE rounding, not after. `next_power_of_two` panics on
    // overflow in debug and wraps to 0 in release, so rounding first would
    // turn a large `ef` into `itopk_size = 0`: cuVS writes nothing, the
    // caller receives the zero-filled neighbour buffer as though it were a
    // result set, and we are back to fast wrong answers reported as
    // success. In debug it is worse — the panic fires while the index
    // mutex is held, poisoning it and disabling the index for the rest of
    // the process.
    ef.min(ITOPK_MAX)
        .max(k.min(ITOPK_MAX))
        .max(ITOPK_MIN)
        .next_power_of_two()
        .min(ITOPK_MAX)
}

/// Raw cuVS CAGRA FFI.
///
/// Isolated so that every `unsafe` in this file is in one place, each with
/// its invariants stated at the call site. See the module docs for why
/// owning the handle is necessary.
#[cfg(feature = "gpu_cuvs")]
mod ffi {
    use std::path::Path;

    use cuvs::{ManagedTensor, Resources};
    use veda_core::error::{Result, VedaError};

    /// Map a `cuvsError_t` to a `Result`, naming the operation.
    ///
    /// Note the C enum is `CUVS_ERROR = 0`, `CUVS_SUCCESS = 1` — the
    /// opposite of the usual C convention, so this must compare against
    /// `CUVS_SUCCESS` explicitly rather than testing for zero.
    pub fn check(code: cuvs_sys::cuvsError_t, op: &str) -> Result<()> {
        if code == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
            return Ok(());
        }
        // `cuvsError_t` carries no detail — every failure is the same
        // opaque `CUVS_ERROR`. The real message lives in cuVS's
        // thread-local last-error slot, so fetch it: "cuvsCagraBuild
        // failed: CUVS_ERROR" tells an operator nothing they can act on.
        Err(VedaError::Index(format!(
            "cuvs {op} failed: {code:?}: {}",
            last_error_text()
        )))
    }

    /// cuVS's thread-local last-error message, or a placeholder.
    fn last_error_text() -> String {
        // SAFETY: `cuvsGetLastErrorText` returns either NULL or a pointer
        // to a NUL-terminated string owned by cuVS and valid until the
        // next cuVS call on this thread. It is copied out immediately.
        let raw = unsafe { cuvs_sys::cuvsGetLastErrorText() };
        if raw.is_null() {
            return "<no detail from cuvsGetLastErrorText>".to_owned();
        }
        // SAFETY: non-null and NUL-terminated per the contract above.
        unsafe { std::ffi::CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }

    fn c_path(path: &Path) -> Result<std::ffi::CString> {
        std::ffi::CString::new(path.to_string_lossy().as_ref())
            .map_err(|e| VedaError::Index(format!("gpu_cuvs: path contains a NUL byte: {e}")))
    }

    /// An owned `cuvsCagraIndex_t`, destroyed exactly once on drop.
    pub struct CagraHandle(cuvs_sys::cuvsCagraIndex_t);

    impl CagraHandle {
        pub fn new() -> Result<Self> {
            let mut raw: cuvs_sys::cuvsCagraIndex_t = std::ptr::null_mut();
            // SAFETY: `raw` is a valid out-pointer for the duration of the
            // call; cuVS either writes a fresh handle into it or returns a
            // non-success code, which `check` turns into an error.
            check(
                unsafe { cuvs_sys::cuvsCagraIndexCreate(&mut raw) },
                "cuvsCagraIndexCreate",
            )?;
            if raw.is_null() {
                return Err(VedaError::Index(
                    "gpu_cuvs: cuvsCagraIndexCreate reported success but returned a null handle"
                        .into(),
                ));
            }
            Ok(Self(raw))
        }

        pub fn raw(&self) -> cuvs_sys::cuvsCagraIndex_t {
            self.0
        }

        /// Rows in the index, per cuVS. Used to restore `len()` after a
        /// load, where the count is not otherwise recoverable.
        pub fn size(&self) -> Result<usize> {
            let mut n: i64 = 0;
            // SAFETY: `self.0` is a live handle; `n` is a valid out-pointer.
            check(
                unsafe { cuvs_sys::cuvsCagraIndexGetSize(self.0, &mut n) },
                "cuvsCagraIndexGetSize",
            )?;
            Ok(n.max(0) as usize)
        }
    }

    impl Drop for CagraHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` came from `cuvsCagraIndexCreate`. The
                // handle is destroyed exactly once because `CagraHandle`
                // is neither `Clone` nor `Copy` and owns it.
                let _ = unsafe { cuvs_sys::cuvsCagraIndexDestroy(self.0) };
            }
        }
    }

    /// Build `handle` over `dataset` on the GPU with the given metric.
    ///
    /// The metric must be set on the params struct: cuVS defaults to
    /// `L2Expanded`, so building a cosine index without setting it would
    /// silently produce a graph for the wrong distance function and return
    /// plausible-looking but wrong neighbours.
    pub fn build(
        res: &Resources,
        dataset: &ManagedTensor,
        handle: &CagraHandle,
        metric: cuvs_sys::cuvsDistanceType,
        graph_degree: usize,
        intermediate_graph_degree: usize,
    ) -> Result<()> {
        let mut params: cuvs_sys::cuvsCagraIndexParams_t = std::ptr::null_mut();
        // SAFETY: valid out-pointer; the object is destroyed below on both
        // the success and failure paths.
        check(
            unsafe { cuvs_sys::cuvsCagraIndexParamsCreate(&mut params) },
            "cuvsCagraIndexParamsCreate",
        )?;
        // SAFETY: `params` is a live, just-created struct we exclusively
        // own until it is destroyed below.
        unsafe {
            (*params).metric = metric;
            (*params).graph_degree = graph_degree;
            (*params).intermediate_graph_degree = intermediate_graph_degree;
        };
        // SAFETY: `res.0` is a live resources handle; `params` was just
        // created; `dataset` outlives the call; `handle.raw()` is live.
        let rc = unsafe { cuvs_sys::cuvsCagraBuild(res.0, params, dataset.as_ptr(), handle.raw()) };
        // SAFETY: `params` came from ParamsCreate and is destroyed once,
        // here, before any early return.
        let _ = unsafe { cuvs_sys::cuvsCagraIndexParamsDestroy(params) };
        check(rc, "cuvsCagraBuild")
    }

    /// Search `handle`, writing into the caller's device tensors.
    pub fn search(
        res: &Resources,
        handle: &CagraHandle,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
        itopk: usize,
        search_width: usize,
    ) -> Result<()> {
        let mut params: cuvs_sys::cuvsCagraSearchParams_t = std::ptr::null_mut();
        // SAFETY: valid out-pointer; destroyed below on every path.
        check(
            unsafe { cuvs_sys::cuvsCagraSearchParamsCreate(&mut params) },
            "cuvsCagraSearchParamsCreate",
        )?;
        // `itopk_size` is the number of intermediate candidates retained
        // during traversal — CAGRA's accuracy/speed knob, the analogue of
        // HNSW's `ef_search`. Leaving it at the library default (64) for
        // every query regardless of corpus size is what made recall decay
        // with scale: a fixed candidate budget explores a shrinking
        // fraction of a growing graph. Measured recall@10 on a clustered
        // corpus was 0.57 at 100k and 0.11 at 1M before this was set.
        //
        // SAFETY: `params` was just created by ParamsCreate above and is
        // a live, uniquely-owned params object; nothing else aliases it.
        unsafe {
            (*params).itopk_size = itopk;
            (*params).search_width = search_width;
        }
        let filter = cuvs_sys::cuvsFilter {
            addr: 0,
            type_: cuvs_sys::cuvsFilterType::NO_FILTER,
        };
        // SAFETY: all handles are live for the duration of the call and
        // the three tensors outlive it; `filter` is a by-value POD.
        let rc = unsafe {
            cuvs_sys::cuvsCagraSearch(
                res.0,
                params,
                handle.raw(),
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                filter,
            )
        };
        // SAFETY: `params` came from ParamsCreate and is destroyed once.
        let _ = unsafe { cuvs_sys::cuvsCagraSearchParamsDestroy(params) };
        check(rc, "cuvsCagraSearch")
    }

    /// Write `handle` to `path`.
    ///
    /// `include_dataset = true` so a reloaded index can answer queries
    /// without the original vectors. That makes the file roughly
    /// dataset-sized.
    pub fn serialize(res: &Resources, handle: &CagraHandle, path: &Path) -> Result<()> {
        let c = c_path(path)?;
        // SAFETY: `res.0` and `handle.raw()` are live; `c` is
        // NUL-terminated and outlives the call.
        check(
            unsafe { cuvs_sys::cuvsCagraSerialize(res.0, c.as_ptr(), handle.raw(), true) },
            "cuvsCagraSerialize",
        )
    }

    /// Fill an already-created empty `handle` from `path`.
    pub fn deserialize(res: &Resources, path: &Path, handle: &CagraHandle) -> Result<()> {
        let c = c_path(path)?;
        // SAFETY: as `serialize`; cuVS populates the existing handle.
        check(
            unsafe { cuvs_sys::cuvsCagraDeserialize(res.0, c.as_ptr(), handle.raw()) },
            "cuvsCagraDeserialize",
        )
    }
}

/// The cuVS resources and index handle, held together because every GPU
/// operation needs both.
#[cfg(feature = "gpu_cuvs")]
struct CagraInner {
    resources: Resources,
    handle: ffi::CagraHandle,
}

// SAFETY: Two separate claims are needed to share this across threads, and
// only one of them requires an assertion.
//
// *Exclusive access* is provided by the `Mutex<Option<CagraInner>>` in
// `CagraIndex`: no two threads ever touch the handle or the stream
// concurrently, so there is no data race to reason about.
//
// *Thread mobility* is what this `impl` asserts: that the handle and its
// stream remain valid when used from a thread other than the one that
// created them. That holds because CUDA contexts have been process-wide
// (not thread-affine) since CUDA 4, and a CUDA stream may be used from any
// thread; `cuvsResources_t` and `cuvsCagraIndex_t` are opaque handles over
// device allocations and a stream, with no thread-local state.
//
// `Sync` is deliberately NOT asserted — it is derived from `Mutex<T>` given
// `T: Send`, which is exactly the guarantee we want: safe to move between
// threads, never usable from two at once.
#[cfg(feature = "gpu_cuvs")]
unsafe impl Send for CagraInner {}

/// GPU CAGRA index.
///
/// With `gpu_cuvs` on, this owns a `cuvs::Resources` and a CAGRA handle
/// behind a `Mutex`. [`VectorIndex`] requires `Send + Sync`, and both come
/// from that mutex plus the narrow `unsafe impl Send for CagraInner` above:
/// the mutex guarantees no two threads ever touch the GPU handle at once,
/// and the `impl` asserts only that the handle may travel between threads,
/// which CUDA's process-wide context model permits.
///
/// Searches therefore serialise. That is honest about the hardware — one
/// shared device whose work already serialises substantially at the stream
/// level — and what it costs is measured in
/// `crates/veda-index/tests/cagra_gpu_bench.rs` rather than guessed at.
///
/// Without the feature, every method returns a clear `VedaError::Index`.
pub struct CagraIndex {
    metric: DistanceMetric,
    dimension: u32,
    count: usize,
    #[cfg(feature = "gpu_cuvs")]
    inner: std::sync::Mutex<Option<CagraInner>>,
}

impl CagraIndex {
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            metric,
            dimension,
            count: 0,
            #[cfg(feature = "gpu_cuvs")]
            inner: std::sync::Mutex::new(None),
        }
    }

    /// Lock the index, mapping a poisoned mutex to an error rather than
    /// panicking. A poisoned lock means a previous panic left the GPU
    /// handle in an unknown state, so the index is not reusable.
    #[cfg(feature = "gpu_cuvs")]
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<CagraInner>>> {
        self.inner.lock().map_err(|_| {
            VedaError::Index(
                "gpu_cuvs: index mutex poisoned by an earlier panic; this index is unusable".into(),
            )
        })
    }

    /// Map Veda's metric onto cuVS's, rejecting the ones CAGRA has no
    /// equivalent for rather than silently building the wrong graph.
    ///
    /// `Euclidean` maps to `L2Expanded`, which returns **squared**
    /// distances. `L2SqrtExpanded` would be the natural choice — it matches
    /// `veda_core::simd::euclidean_distance`'s scale directly — but CAGRA
    /// rejects it: `cuvsCagraBuild` fails with a bare `CUVS_ERROR` and no
    /// detail. Verified on cuvs 26.4 by building the same dataset with each
    /// variant.
    ///
    /// The square root is therefore applied to returned distances in
    /// [`Self::rescale_distance`] instead, so GPU and CPU distances stay
    /// comparable. Skipping that would silently break any threshold tuned
    /// against the CPU path.
    #[cfg(feature = "gpu_cuvs")]
    fn cuvs_metric(metric: DistanceMetric) -> Result<cuvs_sys::cuvsDistanceType> {
        match metric {
            DistanceMetric::Euclidean => Ok(cuvs_sys::cuvsDistanceType::L2Expanded),
            DistanceMetric::Cosine => Ok(cuvs_sys::cuvsDistanceType::CosineExpanded),
            DistanceMetric::DotProduct => Ok(cuvs_sys::cuvsDistanceType::InnerProduct),
            other => Err(VedaError::Index(format!(
                "gpu_cuvs: CAGRA has no equivalent for {other:?}; \
                 supported metrics are Euclidean, Cosine and DotProduct"
            ))),
        }
    }

    /// Put a cuVS distance back on the scale Veda's CPU kernels use.
    ///
    /// CAGRA can only be built with `L2Expanded` (see [`Self::cuvs_metric`]),
    /// which yields squared L2. Every other supported metric is already on
    /// the same scale as its `veda_core::simd` counterpart.
    #[cfg(feature = "gpu_cuvs")]
    fn rescale_distance(metric: DistanceMetric, d: f32) -> f32 {
        match metric {
            // Guard against tiny negatives from floating-point error in
            // the expanded form, which would produce NaN.
            DistanceMetric::Euclidean => d.max(0.0).sqrt(),
            _ => d,
        }
    }
}

/// Whether this binary was compiled with the `gpu_cuvs` feature, i.e.
/// whether the CAGRA GPU index can actually build and search. Callers
/// (e.g. the gateway's create-collection validation) use this to reject
/// a `cagra` index request up-front with a clear error instead of
/// accepting it and silently producing an unsearchable collection.
pub const fn gpu_cuvs_available() -> bool {
    cfg!(feature = "gpu_cuvs")
}

#[allow(dead_code)]
fn feature_off_error(op: &str) -> VedaError {
    VedaError::Index(format!(
        "gpu_cuvs: {op} called but the gpu_cuvs cargo feature is off; \
         rebuild with --features veda-index/gpu_cuvs and ensure the CUDA \
         toolkit is on the build PATH"
    ))
}

#[cfg(feature = "gpu_cuvs")]
impl CagraIndex {
    /// The single implementation of GPU search.
    ///
    /// Inherent rather than a trait method so that both `search` and
    /// `search_with_ef` delegate *down* to one body instead of to each
    /// other — see the note on [`VectorIndex::search_with_ef`].
    fn search_inner(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        let dim = self.dimension as usize;
        if query.len() != dim {
            return Err(VedaError::Index(format!(
                "gpu_cuvs: query dim {} != index dim {dim}",
                query.len()
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        // CAGRA cannot return more results than it retains, and the
        // retention ceiling is a hard library limit. Say so, rather than
        // letting cuVS fail with a bare CUVS_ERROR carrying no detail.
        if k > ITOPK_MAX {
            return Err(VedaError::Index(format!(
                "gpu_cuvs: k={k} exceeds CAGRA's maximum retained candidates ({ITOPK_MAX}); \
                 use a CPU index for larger k"
            )));
        }

        let guard = self.lock()?;
        let inner = guard
            .as_ref()
            .ok_or_else(|| VedaError::Index("gpu_cuvs: search before build or load".into()))?;
        let res = &inner.resources;

        let query_host: Array2<f32> = Array2::from_shape_vec((1, dim), query.to_vec())
            .map_err(|e| VedaError::Index(format!("gpu_cuvs: query reshape failed: {e}")))?;
        let queries_dev = ManagedTensor::from(&query_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs query to_device failed: {e:?}")))?;

        let mut neighbors_host: Array2<u32> = Array2::zeros((1, k));
        let neighbors_dev = ManagedTensor::from(&neighbors_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs neighbors to_device failed: {e:?}")))?;

        let mut distances_host: Array2<f32> = Array2::zeros((1, k));
        let distances_dev = ManagedTensor::from(&distances_host)
            .to_device(res)
            .map_err(|e| VedaError::Index(format!("cuvs distances to_device failed: {e:?}")))?;

        ffi::search(
            res,
            &inner.handle,
            &queries_dev,
            &neighbors_dev,
            &distances_dev,
            itopk_for(k, ef),
            search_width_for(ef),
        )?;

        neighbors_dev
            .to_host(res, &mut neighbors_host)
            .map_err(|e| VedaError::Index(format!("cuvs neighbors to_host failed: {e:?}")))?;
        distances_dev
            .to_host(res, &mut distances_host)
            .map_err(|e| VedaError::Index(format!("cuvs distances to_host failed: {e:?}")))?;

        let neighbors_row = neighbors_host.slice(s![0, ..]);
        let distances_row = distances_host.slice(s![0, ..]);
        let mut results = Vec::with_capacity(k);
        for (id, dist) in neighbors_row.iter().zip(distances_row.iter()) {
            results.push(SearchResult {
                id: *id,
                distance: Self::rescale_distance(self.metric, *dist),
            });
        }
        Ok(results)
    }
}

impl VectorIndex for CagraIndex {
    #[cfg(feature = "gpu_cuvs")]
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let cuvs_metric = Self::cuvs_metric(self.metric)?;
        let n = vectors.len();
        if n == 0 {
            return Err(VedaError::Index(
                "gpu_cuvs: cannot build over empty vector set".into(),
            ));
        }
        let dim = self.dimension as usize;
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VedaError::Index(format!(
                    "gpu_cuvs: vector {i} has dim {} but index dim is {dim}",
                    v.len()
                )));
            }
        }
        let mut flat = Vec::with_capacity(n * dim);
        for v in vectors {
            flat.extend_from_slice(v);
        }
        let dataset: Array2<f32> = Array2::from_shape_vec((n, dim), flat)
            .map_err(|e| VedaError::Index(format!("gpu_cuvs: ndarray reshape failed: {e}")))?;

        let res = Resources::new()
            .map_err(|e| VedaError::Index(format!("cuvs Resources init failed: {e:?}")))?;
        let handle = ffi::CagraHandle::new()?;
        let tensor = ManagedTensor::from(&dataset);
        let (graph_degree, intermediate_graph_degree) = graph_degrees_for(n);
        ffi::build(
            &res,
            &tensor,
            &handle,
            cuvs_metric,
            graph_degree,
            intermediate_graph_degree,
        )?;

        *self.lock()? = Some(CagraInner {
            resources: res,
            handle,
        });
        self.count = n;
        Ok(())
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn build(&mut self, _vectors: &[&[f32]]) -> Result<()> {
        Err(feature_off_error("build"))
    }

    #[cfg(feature = "gpu_cuvs")]
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_inner(query, k, DEFAULT_EF)
    }

    /// CAGRA's traversal budget, plumbed from `ef` onto the cuVS knobs.
    ///
    /// Overriding this is not optional for a graph index: the trait's
    /// default implementation discards `ef` and calls `search`, so before
    /// this existed CAGRA ignored the caller's budget entirely and always
    /// ran at the cuVS default of 64 candidates — including on a 1M-vector
    /// corpus, where that yielded recall@10 = 0.11.
    ///
    /// Both this and [`VectorIndex::search`] delegate to the inherent
    /// [`CagraIndex::search_inner`] rather than to each other. That is
    /// deliberate: if `search` delegated here and this override were ever
    /// removed or fell out of the `cfg`, the trait default would delegate
    /// straight back to `search` and recurse until the stack died. A
    /// stack overflow in an `#[ignore]`d 1M-vector GPU test reads as
    /// device trouble, not as a missing override.
    #[cfg(feature = "gpu_cuvs")]
    fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        self.search_inner(query, k, ef)
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn search(&self, _query: &[f32], _k: usize) -> Result<Vec<SearchResult>> {
        Err(feature_off_error("search"))
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        _filter: &RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        // cuVS exposes a filtered search via `cuvsFilter`, but the bitset
        // format is cuvs-specific. For now, delegate to the unfiltered
        // search; callers post-filter against the returned ids. Wiring the
        // cuvs bitset is a follow-up.
        self.search(query, k)
    }

    fn insert(&mut self, _id: u32, _vector: &[f32]) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: CAGRA is a batch-build index — incremental insert not supported".into(),
        ))
    }

    fn delete(&mut self, _id: u32) -> Result<()> {
        Err(VedaError::Index(
            "gpu_cuvs: CAGRA delete not supported (immutable graph)".into(),
        ))
    }

    fn len(&self) -> usize {
        self.count
    }

    #[cfg(feature = "gpu_cuvs")]
    fn save(&self, path: &Path) -> Result<()> {
        let guard = self.lock()?;
        let inner = guard
            .as_ref()
            .ok_or_else(|| VedaError::Index("gpu_cuvs: save before build or load".into()))?;
        ffi::serialize(&inner.resources, &inner.handle, path)
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn save(&self, path: &Path) -> Result<()> {
        let _ = path;
        Err(feature_off_error("save"))
    }

    #[cfg(feature = "gpu_cuvs")]
    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        // Reject an unsupported metric here too: a file written for one
        // metric must not be reopened claiming another.
        Self::cuvs_metric(metric)?;
        let res = Resources::new()
            .map_err(|e| VedaError::Index(format!("cuvs Resources init failed: {e:?}")))?;
        let handle = ffi::CagraHandle::new()?;
        ffi::deserialize(&res, path, &handle)?;
        // `count` is not carried in the Rust API, so recover it from the
        // index itself rather than reporting a wrong `len()`.
        let count = handle.size().unwrap_or(0);
        Ok(Self {
            metric,
            dimension,
            count,
            inner: std::sync::Mutex::new(Some(CagraInner {
                resources: res,
                handle,
            })),
        })
    }

    #[cfg(not(feature = "gpu_cuvs"))]
    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let _ = (path, metric, dimension);
        Err(feature_off_error("load"))
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "gpu_cuvs"))]
    fn feature_off_returns_clear_error() {
        let mut idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.build(&[]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }

    #[test]
    #[cfg(not(feature = "gpu_cuvs"))]
    fn search_off_returns_clear_error() {
        let idx = CagraIndex::new(DistanceMetric::Cosine, 16);
        let err = idx.search(&[0.0; 16], 5).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gpu_cuvs"), "got: {msg}");
    }

    /// `VectorIndex` requires `Send + Sync`. This is a compile-time
    /// assertion: it fails to build if the `Mutex` is removed, which is
    /// the entire reason the mutex exists.
    #[test]
    #[cfg(feature = "gpu_cuvs")]
    fn cagra_index_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CagraIndex>();
    }

    #[cfg(feature = "gpu_cuvs")]
    fn deterministic_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / 16_777_216.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    /// `itopk_size` must never come back as 0, and must never be below `k`.
    ///
    /// `itopk_size = 0` is the dangerous case: cuVS writes nothing, and the
    /// caller receives the zero-filled neighbour buffer as though it were a
    /// result set. An earlier version rounded to a power of two before
    /// clamping, so `ef = usize::MAX` wrapped to exactly that.
    #[test]
    #[cfg(feature = "gpu_cuvs")]
    fn itopk_is_never_zero_and_never_below_k() {
        for &ef in &[0usize, 1, 31, 32, 96, 512, 1024, usize::MAX / 2, usize::MAX] {
            for &k in &[1usize, 10, 511, 512] {
                let itopk = itopk_for(k, ef);
                assert!(itopk > 0, "itopk_for({k}, {ef}) = 0");
                assert!(itopk >= k, "itopk_for({k}, {ef}) = {itopk} < k");
                assert!(itopk <= ITOPK_MAX, "itopk_for({k}, {ef}) = {itopk} > max");
                assert!(itopk.is_power_of_two(), "itopk_for({k}, {ef}) = {itopk}");
            }
        }
        assert!(search_width_for(usize::MAX) <= 16);
        assert!(search_width_for(0) >= 1);
    }

    /// Build degrees must stay below the corpus size.
    ///
    /// A k-NN graph cannot have degree >= its own vertex count, and this
    /// module's own tests build corpora of 200 and 500 against an
    /// intermediate degree of 256.
    #[test]
    #[cfg(feature = "gpu_cuvs")]
    fn graph_degrees_stay_under_the_corpus_size() {
        for &n in &[2usize, 10, 200, 256, 257, 500, 1_000, 1_000_000] {
            let (graph, intermediate) = graph_degrees_for(n);
            assert!(
                graph >= 1 && intermediate >= 1,
                "n={n} produced a zero degree"
            );
            assert!(
                graph <= intermediate,
                "n={n}: graph {graph} > intermediate {intermediate}"
            );
            assert!(intermediate < n, "n={n}: intermediate {intermediate} >= n");
            assert!(intermediate <= INTERMEDIATE_GRAPH_DEGREE);
            assert!(graph <= GRAPH_DEGREE);
        }
        // Large corpora get the full, measured degrees.
        assert_eq!(
            graph_degrees_for(1_000_000),
            (GRAPH_DEGREE, INTERMEDIATE_GRAPH_DEGREE)
        );
    }

    /// Corpus shape, matched to the benchmark's so the two are comparable.
    #[cfg(feature = "gpu_cuvs")]
    const TEST_CLUSTERS: usize = 1_000;
    #[cfg(feature = "gpu_cuvs")]
    const TEST_CLUSTER_SIGMA: f32 = 0.10;

    /// Vectors drawn from `clusters` centroids — the shape real embeddings
    /// have.
    ///
    /// Uniform-random vectors in high dimension are all nearly equidistant,
    /// so the true top-k is close to arbitrary and *every* graph index scores
    /// near chance on them. A recall assertion over uniform data therefore
    /// cannot distinguish a healthy index from a broken one. Centroids come
    /// from a fixed seed so a query set and the corpus share structure.
    ///
    /// **Deliberately identical to `tests/cagra_gpu_bench.rs::generate`**:
    /// 1,000 centroids, signed values in `[-1, 1)`, two-sided noise at
    /// `TEST_CLUSTER_SIGMA`. An earlier version used 500 centroids with
    /// one-sided noise at 0.05, which was easier in two ways that the
    /// constants did not reveal — the one-sided noise offset each cluster's
    /// true centre and cut the effective per-dimension spread to ~0.0144.
    /// A healthy index then **saturated at 1.0** on it, so the floor could
    /// only detect collapse and not a partial regression, and the test
    /// scored 0.9641 where the benchmark it was meant to protect measured
    /// 0.805. Matching the benchmark is what makes the floor a recall
    /// target rather than a smoke alarm.
    #[cfg(feature = "gpu_cuvs")]
    fn clustered_vectors(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        let centroids = deterministic_vectors(clusters, dim, 0xC0FF_EE01);
        let noise = deterministic_vectors(n, dim, seed);
        (0..n)
            .map(|i| {
                // Spread the query set across the whole centroid set. With
                // `n < clusters` — which is the query set — `i % clusters`
                // only ever touches the first `n` centroids, so with 64
                // queries over 1,000 clusters 94% of the corpus could never
                // be a target and the measurement covered one corner of it.
                let c_idx = if n >= clusters {
                    i % clusters
                } else {
                    i * clusters / n
                };
                let c = &centroids[c_idx];
                (0..dim)
                    .map(|d| signed(c[d]) + signed(noise[i][d]) * TEST_CLUSTER_SIGMA)
                    .collect()
            })
            .collect()
    }

    /// Map `deterministic_vectors`' `[0, 1)` output onto `[-1, 1)`.
    ///
    /// Without this the noise is one-sided, which offsets every cluster's
    /// true centre from its centroid and shrinks the effective spread by
    /// ~3.5x — making the corpus easier than the benchmark's in a way that
    /// is invisible from the constants.
    #[cfg(feature = "gpu_cuvs")]
    fn signed(v: f32) -> f32 {
        v * 2.0 - 1.0
    }

    /// Exact top-k by brute force, the ground truth recall is measured against.
    #[cfg(feature = "gpu_cuvs")]
    fn exact_top_k(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i as u32, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// CAGRA must actually find the true nearest neighbours.
    ///
    /// This is the test the design doc claimed existed and did not. The
    /// pre-existing `build_and_search_finds_the_planted_neighbour` plants an
    /// exact duplicate of the query and asserts it ranks first — which a
    /// badly degraded graph still passes, because the query is itself a node
    /// in the graph. It stayed green while measured recall@10 on a clustered
    /// 1M corpus was 0.11.
    ///
    /// The floor is deliberately well below what a healthy CAGRA achieves:
    /// this guards against collapse, not against ordinary ANN approximation.
    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn recall_at_10_is_not_silently_terrible() {
        // Scale matters: the defect this guards was invisible at 50k
        // (recall 0.99) and severe at 1M (0.40). A cheap small-corpus
        // recall test would have passed throughout.
        const N: usize = 1_000_000;
        // Must match the dimension the benchmark measures. At dim=64 this
        // test passed at 0.9641 while the benchmark's dim=128 case measured
        // 0.805 — green here, failing there. A test that does not cover
        // the configuration it exists to protect is worse than no test,
        // because it also supplies false confidence.
        const DIM: usize = 128;
        const K: usize = 10;
        // Set from measurement, with margin, and it is a real recall target
        // rather than a collapse detector.
        //
        // A healthy index measures **0.8750** here. The floor sits ~5.7
        // standard errors below that (64 queries x k=10 gives 640 samples,
        // so one standard error is ~0.013), which is loose enough not to
        // flake and tight enough to catch every regression this branch
        // fixed. Measured on this corpus: reverting the denser graph lands
        // near 0.766 (the ef=512 row of the sweep below is the same search
        // budget against a sparser graph), leaving `search_width` at 1
        // lands at 0.1719, and the original unset `itopk_size` lower still.
        // All are caught.
        //
        // This is only meaningful because the corpus matches the
        // benchmark's. On the previous, easier corpus a healthy index
        // saturated at 1.0000, and any floor below it could only detect
        // total collapse — a partial regression to 0.9 would have passed.
        const FLOOR: f64 = 0.80;

        let data = clustered_vectors(N, DIM, TEST_CLUSTERS, 0x0BAD_F00D);
        let queries = clustered_vectors(64, DIM, TEST_CLUSTERS, 0x1234_ABCD);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut idx = CagraIndex::new(DistanceMetric::Euclidean, DIM as u32);
        idx.build(&refs).expect("GPU build");

        let truths: Vec<std::collections::HashSet<u32>> = queries
            .iter()
            .map(|q| exact_top_k(&data, q, K).into_iter().collect())
            .collect();

        let measure = |ef: Option<usize>| -> f64 {
            let mut total = 0.0f64;
            for (q, truth) in queries.iter().zip(truths.iter()) {
                let hits = match ef {
                    Some(e) => idx.search_with_ef(q, K, e).expect("GPU search"),
                    None => idx.search(q, K).expect("GPU search"),
                };
                let found = hits.iter().filter(|r| truth.contains(&r.id)).count();
                total += found as f64 / K as f64;
            }
            total / queries.len() as f64
        };

        for ef in [64usize, 128, 256, 512, 768, 1024] {
            println!(
                "ef={ef:<5} itopk={:<5} width={:<3} recall@{K}={:.4}",
                itopk_for(K, ef),
                search_width_for(ef),
                measure(Some(ef))
            );
        }

        let recall = measure(None);

        assert!(
            recall >= FLOOR,
            "recall@{K} = {recall:.4} on {N} clustered vectors, below the {FLOOR} floor. \
             CAGRA is returning fast wrong answers."
        );

        // A larger traversal budget must actually produce better recall.
        // This is the assertion that would have caught the original
        // defect: `search_with_ef` was not overridden, so `ef` was
        // discarded and every query ran at the library default no matter
        // what the caller asked for. Recall was then flat in `ef` — which
        // is what this fails on.
        let low = measure(Some(ITOPK_MIN));
        assert!(
            recall > low,
            "recall@{K} did not improve with the traversal budget \
             (ef={ITOPK_MIN} gives {low:.4}, default ef={DEFAULT_EF} gives {recall:.4}). \
             `ef` is not reaching cuVS."
        );
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn ffi_handle_creates_and_drops() {
        // Exercises cuvsCagraIndexCreate/Destroy against the real driver.
        // A leak or double-free surfaces here rather than at some
        // unrelated later point.
        for _ in 0..100 {
            let h = ffi::CagraHandle::new().expect("cuvsCagraIndexCreate");
            assert!(!h.raw().is_null());
        }
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn build_and_search_finds_the_planted_neighbour() {
        let dim = 32usize;
        let n = 1000usize;
        let mut data = deterministic_vectors(n, dim, 0x1234_5678);
        let query = data[500].clone();
        // Plant an exact duplicate so the correct answer is known
        // independently of whatever the index does.
        data[500] = query.clone();

        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
        idx.build(&refs).expect("GPU build");
        assert_eq!(idx.len(), n);

        let hits = idx.search(&query, 5).expect("GPU search");
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].id, 500, "planted exact duplicate must rank first");
        assert!(
            hits[0].distance < 1e-3,
            "exact duplicate distance should be ~0, got {}",
            hits[0].distance
        );
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn save_then_load_returns_identical_results() {
        let dim = 32usize;
        let n = 500usize;
        let data = deterministic_vectors(n, dim, 0xABCD_EF01);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
        idx.build(&refs).expect("GPU build");
        let before = idx.search(&data[7], 10).expect("search before save");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cagra.idx");
        idx.save(&path).expect("save");
        assert!(path.exists(), "save must write a file");

        let reloaded =
            CagraIndex::load(&path, DistanceMetric::Euclidean, dim as u32).expect("load");
        let after = reloaded.search(&data[7], 10).expect("search after load");

        // This is the test that would have caught the original defect:
        // the previous code called serialize_to_hnswlib/deserialize, which
        // do not exist in cuvs 26.4.
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.id, a.id, "id differs across the persistence boundary");
            assert!(
                (b.distance - a.distance).abs() < 1e-4,
                "distance differs across persistence: {} vs {}",
                b.distance,
                a.distance
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn concurrent_searches_are_safe_and_correct() {
        use std::sync::Arc;

        let dim = 32usize;
        let n = 500usize;
        let data = deterministic_vectors(n, dim, 0x5A5A_1234);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
        idx.build(&refs).expect("GPU build");

        let expected = idx.search(&data[3], 5).expect("baseline search");
        let shared = Arc::new(idx);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let s = Arc::clone(&shared);
                let q = data[3].clone();
                std::thread::spawn(move || s.search(&q, 5).expect("threaded search"))
            })
            .collect();

        for h in handles {
            let got = h.join().expect("worker thread panicked");
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_eq!(g.id, e.id, "concurrent search returned different ids");
            }
        }
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn gpu_euclidean_distances_match_the_cpu_kernel() {
        // CAGRA can only be built with L2Expanded, which returns SQUARED
        // distances, so `rescale_distance` takes the square root. Without
        // that, GPU distances would be on a different scale from
        // `veda_core::simd::euclidean_distance` and every threshold tuned
        // on the CPU path would silently misbehave on GPU.
        //
        // This test fails if the sqrt is removed — squared distances for
        // these vectors are far outside the tolerance below.
        let dim = 16usize;
        let n = 200usize;
        let data = deterministic_vectors(n, dim, 0x7777_1111);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
        idx.build(&refs).expect("GPU build");

        let query = &data[11];
        let hits = idx.search(query, 5).expect("GPU search");
        for h in &hits {
            let cpu = veda_core::simd::euclidean_distance(query, &data[h.id as usize]);
            assert!(
                (h.distance - cpu).abs() < 1e-2,
                "GPU distance {} is not on the CPU scale {} for id {}",
                h.distance,
                cpu,
                h.id
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu_cuvs")]
    #[ignore]
    fn unsupported_metric_is_rejected_not_silently_wrong() {
        // Hamming has no CAGRA equivalent. Building with it must fail
        // loudly rather than defaulting to L2 and returning plausible but
        // wrong neighbours.
        let dim = 16usize;
        let data = deterministic_vectors(64, dim, 0x0F0F_0F0F);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut idx = CagraIndex::new(DistanceMetric::Hamming, dim as u32);
        let err = idx.build(&refs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Hamming"), "error must name the metric: {msg}");
    }
}
