# CAGRA GPU Index Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `veda-index`'s CAGRA GPU index compile, run, persist, and stay guarded by CI — then publish a recall-qualified benchmark on real hardware.

**Architecture:** `crates/veda-index/src/gpu_cuvs.rs` owns the `cuvsCagraIndex_t` handle directly through `cuvs-sys`, because `cuvs`'s safe wrapper hides the pointer and therefore cannot serialize. All FFI lives in one private submodule behind a safe owned type; the index and its `Resources` sit behind a `Mutex` so `CagraIndex` is `Send + Sync` without asserting anything unsafe about cuVS internals.

**Tech Stack:** Rust 2024, `cuvs 26.4` (for `Resources` and `ManagedTensor`), `cuvs-sys 26.4` (for the CAGRA C API), CUDA 12.9, 4× RTX 6000 Ada on `gpu-host-a`.

**Spec:** `documentation/2026-08-15-gpu-cuvs-cagra-design.md` (`4a4b538`, revised `7d7a6e8`).

## Global Constraints

- Toolchain pinned to **1.96.0**. Do not change it.
- **Source of truth is remote:** `gpu-host-a.internal.example:/home/mac/test/project-mdb`.
  Edit locally, `scp` across, build and test over SSH.
- **Every cargo command that touches `gpu_cuvs` needs this exact environment.** Copy it
  verbatim; it is seven separate blockers' worth of setup and none of it is optional:

```bash
V=/home/mac/veda-cuvs-venv/lib/python3.12/site-packages
export PATH=/home/mac/veda-cuvs-venv/bin:/usr/local/cuda/bin:$PATH
export PROTOC=/home/mac/bin/protoc
export CUDAToolkit_ROOT=/usr/local/cuda
export LIBCLANG_PATH=$V/clang/native
export BINDGEN_EXTRA_CLANG_ARGS='-I/usr/local/cuda/include -I/usr/lib/gcc/x86_64-linux-gnu/12/include'
export cuvs_DIR=$V/libcuvs/lib64/cmake/cuvs
export raft_DIR=$V/libraft/lib64/cmake/raft
export rmm_DIR=$V/librmm/lib64/cmake/rmm
export rapids_logger_DIR=$V/rapids_logger/lib64/cmake/rapids_logger
export nvtx3_DIR=$V/librmm/lib64/cmake/nvtx3
export CCCL_DIR=$V/libraft/lib64/rapids/cmake/cccl
export Thrust_DIR=$V/libraft/lib64/rapids/cmake/thrust
export CUB_DIR=$V/libraft/lib64/rapids/cmake/cub
export libcudacxx_DIR=$V/libraft/lib64/rapids/cmake/libcudacxx
export LD_LIBRARY_PATH=$V/libcuvs/lib64:$V/libraft/lib64:$V/librmm/lib64:$V/rapids_logger/lib64:$LD_LIBRARY_PATH
```

  `LD_LIBRARY_PATH` is needed to **run** (not merely compile) anything linking `libcuvs_c`.
- **Run every cargo command in the FOREGROUND.** Note the harness moves any command past
  ~10 minutes to the background automatically — that is not your doing and not a failure.
  If it happens, wait for the completion notification and read the output; do **not**
  abandon the task or re-run blindly.
- **Never** `pkill`/`kill -9` cargo, rustc, or any process you did not start. This
  previously wedged the host into a reboot.
- **Never** run an unscoped `cargo test`/`cargo build`. Scope with `-p veda-index`.
- **Never** `git clean`, `git checkout -- .`, or `git stash` — a human works in this
  checkout concurrently. `git add` explicit paths only.
- Commit messages: conventional prefix, **never** mention Claude, AI, or `Co-Authored-By`.
- Branch `fix/gpu-cuvs-cagra` is checked out. Do not switch branches.
- **All GPU tests are `#[cfg(feature = "gpu_cuvs")]` and `#[ignore]`** — CI has no GPU.
  Run them explicitly with `-- --ignored`.
- **No `unsafe` outside the `ffi` submodule.** That containment is the whole safety story.
- **The default (feature-off) build must keep working unchanged.** Every existing
  feature-off test passes untouched.

---

### Task 1: The FFI submodule — owned handle, create and destroy

**Files:**
- Modify: `crates/veda-index/Cargo.toml`
- Modify: `crates/veda-index/src/gpu_cuvs.rs`

**Interfaces:**
- Produces, used by every later task:
  - `mod ffi` (private submodule inside `gpu_cuvs.rs`)
  - `ffi::CagraHandle` — owns a `cuvsCagraIndex_t`, destroys it on `Drop`
  - `ffi::CagraHandle::new() -> Result<CagraHandle>`
  - `ffi::CagraHandle::raw(&self) -> cuvs_sys::cuvsCagraIndex_t`
  - `ffi::check(code: cuvs_sys::cuvsError_t, op: &str) -> Result<()>`

- [ ] **Step 1: Add the dependency**

In `crates/veda-index/Cargo.toml`, beside the existing optional `cuvs`:

```toml
cuvs-sys = { version = "26.4", optional = true }
```

and extend the feature:

```toml
gpu_cuvs = ["dep:cuvs", "dep:cuvs-sys", "dep:ndarray"]
```

The version must match `cuvs`'s own `cuvs-sys` so the two resolve to one crate and the
`DLManagedTensor` / `cuvsResources_t` types unify. Confirm with:

`cargo tree -p veda-index --features gpu_cuvs -i cuvs-sys` — expect a single version.

- [ ] **Step 2: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-index/src/gpu_cuvs.rs`:

```rust
#[test]
#[ignore]
#[cfg(feature = "gpu_cuvs")]
fn ffi_handle_creates_and_drops() {
    // Exercises cuvsCagraIndexCreate/Destroy against the real driver.
    // If this leaks or double-frees, the GPU driver reports it here rather
    // than at some unrelated later point.
    for _ in 0..100 {
        let h = ffi::CagraHandle::new().expect("cuvsCagraIndexCreate");
        assert!(!h.raw().is_null());
    }
}
```

- [ ] **Step 3: Run it to verify it fails**

```bash
<env block from Global Constraints>
cargo test -p veda-index --features gpu_cuvs --lib gpu_cuvs -- --ignored
```

Expected: compile error — `ffi` does not exist.

- [ ] **Step 4: Implement the submodule**

Add near the top of `gpu_cuvs.rs`:

```rust
/// Raw cuVS CAGRA FFI.
///
/// `cuvs::cagra::Index` hides its `cuvsCagraIndex_t` and `cuvs` does not
/// re-export its `ffi` module, so a safe-wrapper index cannot be
/// serialized. This module therefore owns the handle for its whole
/// lifecycle. **All `unsafe` in this file lives here**; nothing outside
/// this submodule may contain an `unsafe` block.
#[cfg(feature = "gpu_cuvs")]
mod ffi {
    use std::path::Path;

    use veda_core::error::{Result, VedaError};

    /// Map a `cuvsError_t` to a `Result`, naming the operation.
    pub fn check(code: cuvs_sys::cuvsError_t, op: &str) -> Result<()> {
        if code == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
            Ok(())
        } else {
            Err(VedaError::Index(format!("cuvs {op} failed: {code:?}")))
        }
    }

    /// An owned `cuvsCagraIndex_t`, destroyed on drop.
    pub struct CagraHandle(cuvs_sys::cuvsCagraIndex_t);

    impl CagraHandle {
        pub fn new() -> Result<Self> {
            let mut raw: cuvs_sys::cuvsCagraIndex_t = std::ptr::null_mut();
            // SAFETY: `raw` is a valid out-pointer for the duration of the
            // call; cuVS writes a fresh handle into it or returns an error.
            check(unsafe { cuvs_sys::cuvsCagraIndexCreate(&mut raw) }, "cuvsCagraIndexCreate")?;
            if raw.is_null() {
                return Err(VedaError::Index(
                    "cuvsCagraIndexCreate returned success with a null handle".into(),
                ));
            }
            Ok(Self(raw))
        }

        pub fn raw(&self) -> cuvs_sys::cuvsCagraIndex_t {
            self.0
        }
    }

    impl Drop for CagraHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` came from `cuvsCagraIndexCreate` and is
                // destroyed exactly once, here, because `CagraHandle` is not
                // `Clone` and owns the handle.
                unsafe { cuvs_sys::cuvsCagraIndexDestroy(self.0) };
            }
        }
    }
}
```

If the generated enum variant is not `cuvsError_t::CUVS_SUCCESS`, read
`target/debug/build/cuvs-sys-*/out/cuvs_bindings.rs` for the real name and use it.
**Report any such correction.**

- [ ] **Step 5: Run it to verify it passes**

Same command. Expected: PASS.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-index --all-targets -- -D warnings
cargo clippy -p veda-index --all-targets --features gpu_cuvs -- -D warnings
git add crates/veda-index/Cargo.toml Cargo.lock crates/veda-index/src/gpu_cuvs.rs
git commit -m "feat(gpu): own the CAGRA index handle through cuvs-sys"
```

---

### Task 2: Build and search through the owned handle

**Files:**
- Modify: `crates/veda-index/src/gpu_cuvs.rs`

**Interfaces:**
- Consumes: `ffi::{CagraHandle, check}` (Task 1).
- Produces:
  - `ffi::build(res: &Resources, dataset: &ManagedTensor, handle: &CagraHandle) -> Result<()>`
  - `ffi::search(res: &Resources, handle: &CagraHandle, queries: &ManagedTensor, neighbors: &ManagedTensor, distances: &ManagedTensor) -> Result<()>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[ignore]
#[cfg(feature = "gpu_cuvs")]
fn gpu_build_then_search_finds_the_planted_neighbour() {
    // 1000 random vectors plus one planted duplicate of the query, so the
    // nearest neighbour is known independently of the index.
    let dim = 32usize;
    let n = 1000usize;
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        ((rng >> 40) as f32) / 16_777_216.0
    };
    let mut data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
    let query: Vec<f32> = (0..dim).map(|_| next()).collect();
    data[500] = query.clone();

    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).expect("GPU build");
    let hits = idx.search(&query, 5).expect("GPU search");

    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].id, 500, "planted exact duplicate must rank first");
    assert!(hits[0].distance < 1e-3, "exact duplicate distance: {}", hits[0].distance);
}
```

- [ ] **Step 2: Run it to verify it fails**

Expected: compile error, or a failure from the still-unported `build`/`search`.

- [ ] **Step 3: Implement `ffi::build` and `ffi::search`**

Add to the `ffi` submodule:

```rust
    use cuvs::{ManagedTensor, Resources};

    pub fn build(res: &Resources, dataset: &ManagedTensor, handle: &CagraHandle) -> Result<()> {
        let mut params: cuvs_sys::cuvsCagraIndexParams_t = std::ptr::null_mut();
        // SAFETY: valid out-pointer; destroyed below on every path.
        check(unsafe { cuvs_sys::cuvsCagraIndexParamsCreate(&mut params) },
              "cuvsCagraIndexParamsCreate")?;
        // SAFETY: `res.0` is a live resources handle, `params` was just
        // created, `dataset` outlives the call, `handle.raw()` is live.
        let rc = unsafe {
            cuvs_sys::cuvsCagraBuild(res.0, params, dataset.as_ptr(), handle.raw())
        };
        // SAFETY: `params` came from ParamsCreate and is destroyed once.
        unsafe { cuvs_sys::cuvsCagraIndexParamsDestroy(params) };
        check(rc, "cuvsCagraBuild")
    }
```

`search` follows the same shape with `cuvsCagraSearchParamsCreate` /
`cuvsCagraSearchParamsDestroy` and a `NO_FILTER` `cuvsFilter`:

```rust
        let filter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };
```

**Destroy params on the error path too** — return only after destroying, as `build` does
above. A leaked params object per query is a real leak on a hot path.

- [ ] **Step 4: Rewrite `VectorIndex::build` and `search` to use them**

Keep the existing tensor construction (`Array2`, `ManagedTensor`) and the existing
neighbour/distance readback; replace only the `CuvsCagraIndex::build` /
`.search(...)` calls with `ffi::build` / `ffi::search`, and store a `ffi::CagraHandle`
instead of a `CuvsCagraIndex`. Remove the `use cuvs::cagra::...` imports.

Metric: `cuvsCagraIndexParams` defaults to L2. If `self.metric` is not `Euclidean`,
return a clear `VedaError::Index` saying which metrics the GPU path supports rather than
silently searching with the wrong one. **Report what you find about metric support.**

- [ ] **Step 5: Run it to verify it passes**

Expected: PASS.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-index --all-targets --features gpu_cuvs -- -D warnings
git add crates/veda-index/src/gpu_cuvs.rs
git commit -m "feat(gpu): build and search CAGRA through the owned handle"
```

---

### Task 3: Persistence

**Files:**
- Modify: `crates/veda-index/src/gpu_cuvs.rs`

**Interfaces:**
- Consumes: `ffi::{CagraHandle, check}`.
- Produces:
  - `ffi::serialize(res: &Resources, handle: &CagraHandle, path: &Path) -> Result<()>`
  - `ffi::deserialize(res: &Resources, path: &Path, handle: &CagraHandle) -> Result<()>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[ignore]
#[cfg(feature = "gpu_cuvs")]
fn save_then_load_returns_identical_results() {
    let dim = 32usize;
    let n = 500usize;
    let mut rng = 0xABCD_EF01u64;
    let mut next = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        ((rng >> 40) as f32) / 16_777_216.0
    };
    let data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

    let mut idx = CagraIndex::new(DistanceMetric::Euclidean, dim as u32);
    idx.build(&refs).expect("GPU build");
    let before = idx.search(&data[7], 10).expect("search before save");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cagra.idx");
    idx.save(&path).expect("save");
    assert!(path.exists(), "save must write a file");

    let reloaded = CagraIndex::load(&path, DistanceMetric::Euclidean, dim as u32)
        .expect("load");
    let after = reloaded.search(&data[7], 10).expect("search after load");

    // This is the test that would have caught the missing serialization
    // API: the original code called methods cuvs 26.4 does not have.
    assert_eq!(before.len(), after.len());
    for (b, a) in before.iter().zip(after.iter()) {
        assert_eq!(b.id, a.id, "id differs across the persistence boundary");
        assert!((b.distance - a.distance).abs() < 1e-4,
                "distance differs: {} vs {}", b.distance, a.distance);
    }
}
```

`tempfile` must be a dev-dependency of `veda-index`; check `Cargo.toml` and add it under
`[dev-dependencies]` only if absent.

- [ ] **Step 2: Run it to verify it fails**

Expected: failure — `save`/`load` currently return the feature-off error or call the
removed hnswlib functions.

- [ ] **Step 3: Implement**

```rust
    pub fn serialize(res: &Resources, handle: &CagraHandle, path: &Path) -> Result<()> {
        let c = std::ffi::CString::new(path.to_string_lossy().as_ref())
            .map_err(|e| VedaError::Index(format!("path contains a NUL byte: {e}")))?;
        // SAFETY: `res.0` and `handle.raw()` are live; `c` is NUL-terminated
        // and outlives the call. `include_dataset = true` so a reloaded
        // index can answer queries without the original vectors.
        check(unsafe { cuvs_sys::cuvsCagraSerialize(res.0, c.as_ptr(), handle.raw(), true) },
              "cuvsCagraSerialize")
    }
```

`deserialize` mirrors it, taking an already-created empty `CagraHandle` and filling it via
`cuvsCagraDeserialize`.

- [ ] **Step 4: Wire `VectorIndex::save` / `load`**

`save` locks the mutex, errors clearly if there is no built index, and calls
`ffi::serialize`. `load` creates `Resources`, creates an empty handle, calls
`ffi::deserialize`, and returns a `CagraIndex` with the given metric and dimension.

`load` cannot recover `count` from the file via the Rust API, so use
`cuvsCagraIndexGetSize` (it exists in the bindings) to populate it. If that proves
unreliable, set `count` to 0 and **say so in your report** rather than inventing a value.

**Delete `serialize_to_hnswlib` and `deserialize_from_hnswlib` entirely.** They are a
build-accelerator path that abandons GPU search, nothing calls them, and they reference
methods that do not exist.

- [ ] **Step 5: Run it to verify it passes**

Expected: PASS. Also re-run Task 2's test — it must still pass.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-index --all-targets --features gpu_cuvs -- -D warnings
git add crates/veda-index/src/gpu_cuvs.rs crates/veda-index/Cargo.toml
git commit -m "feat(gpu): persist and reload CAGRA indexes via the cuVS C API"
```

---

### Task 4: Thread safety

**Files:**
- Modify: `crates/veda-index/src/gpu_cuvs.rs`

**Interfaces:**
- Produces: `CagraIndex` satisfying `VectorIndex: Send + Sync` with no `unsafe impl`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[ignore]
#[cfg(feature = "gpu_cuvs")]
fn concurrent_searches_are_safe_and_correct() {
    use std::sync::Arc;
    let dim = 32usize;
    let n = 500usize;
    let mut rng = 0x5A5A_1234u64;
    let mut next = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        ((rng >> 40) as f32) / 16_777_216.0
    };
    let data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
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
        let got = h.join().expect("thread panicked");
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g.id, e.id, "concurrent search returned different ids");
        }
    }
}

/// `VectorIndex` requires `Send + Sync`; this fails to COMPILE if the
/// mutex is removed, which is the point.
#[test]
#[cfg(feature = "gpu_cuvs")]
fn cagra_index_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CagraIndex>();
}
```

Note the second test is **not** `#[ignore]` — it is a compile-time assertion and needs no
GPU.

- [ ] **Step 2: Run it to verify it fails**

Expected: the `E0277` errors from the spec — `*mut cuvsCagraIndex` is neither `Send` nor
`Sync`.

- [ ] **Step 3: Implement the mutex**

```rust
#[cfg(feature = "gpu_cuvs")]
struct CagraInner {
    resources: Resources,
    handle: ffi::CagraHandle,
}

pub struct CagraIndex {
    metric: DistanceMetric,
    dimension: u32,
    count: usize,
    #[cfg(feature = "gpu_cuvs")]
    inner: std::sync::Mutex<Option<CagraInner>>,
}
```

`Send`/`Sync` follow from `Mutex`. **Do not write `unsafe impl Send`/`Sync`** — an
incorrect assertion there is a data race, not a test failure.

Add a helper that maps a poisoned lock to an error rather than unwrapping:

```rust
    #[cfg(feature = "gpu_cuvs")]
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<CagraInner>>> {
        self.inner.lock().map_err(|_| {
            VedaError::Index(
                "gpu_cuvs: index mutex poisoned by an earlier panic; index unusable".into(),
            )
        })
    }
```

Every method that touches the index goes through `lock()`.

- [ ] **Step 4: Run it to verify it passes**

Expected: PASS, including Tasks 2 and 3's tests, which must still pass.

- [ ] **Step 5: Verify the feature-off build is untouched**

```bash
cargo test -p veda-index --lib gpu_cuvs
```

Expected: the existing feature-off tests pass unchanged.

- [ ] **Step 6: Gates and commit**

```bash
cargo fmt --all --check
cargo clippy -p veda-index --all-targets -- -D warnings
cargo clippy -p veda-index --all-targets --features gpu_cuvs -- -D warnings
git add crates/veda-index/src/gpu_cuvs.rs
git commit -m "fix(gpu): make the CAGRA index Send + Sync behind a mutex"
```

---

### Task 5: CI coverage

**Files:**
- Modify: the workflow file under `.github/workflows/` that defines
  `cargo check (veda-index/gpu_nim)` — read it and follow its structure exactly.

This is the task that stops the failure recurring. Without it, `gpu_cuvs` goes back to
being compiled by nobody.

- [ ] **Step 1: Read the existing gpu_nim job**

Find it and note how it names itself, which runner it uses, and how it caches. The new job
mirrors it.

- [ ] **Step 2: Add the job**

A `cargo check (veda-index/gpu_cuvs)` job that:

1. installs `protobuf-compiler` (as the other jobs do),
2. installs the toolchain into a venv:
   `pip install 'cmake==3.31.*' libclang` and
   `pip install --extra-index-url https://pypi.nvidia.com 'libcuvs-cu12==26.4.0'`,
3. exports the same environment block as this plan's Global Constraints, with paths
   adjusted to the runner's venv location,
4. runs `cargo check -p veda-index --features gpu_cuvs`.

Cache the pip downloads on the hash of the workflow file so the ~2 GB install is paid once
per change rather than per run.

**No GPU is required** — compilation and linking are exactly what broke.

- [ ] **Step 3: Validate the workflow file parses**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/<file>'))"`
Expected: no output. If `pyyaml` is unavailable, say so and validate by inspection.

You cannot run GitHub Actions locally; the real proof is CI on the PR. **Say plainly in
your report that this job is unverified until CI runs it.**

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/
git commit -m "ci: compile the gpu_cuvs feature so it cannot silently break"
```

---

### Task 6: The build recipe

**Files:**
- Modify: `crates/veda-index/src/gpu_cuvs.rs` (module doc comment)
- Modify: `documentation/2026-08-15-gpu-cuvs-cagra-design.md` (append)

The module's current doc comment lists two prerequisites — CUDA toolkit and driver ≥ 535.
Both were satisfied on the build host and the build still failed seven distinct ways. It
is wrong by omission and gets replaced.

- [ ] **Step 1: Replace the doc comment's build section**

State all seven steps with exact versions, and why each is needed — particularly that
`cmake-rs` sets `CMAKE_PREFIX_PATH=""`, which is why nine `<Package>_DIR` variables are
required instead, and that `LD_LIBRARY_PATH` is needed to *run* rather than compile.

- [ ] **Step 2: Append the same recipe to the design doc** under a "Build recipe" heading,
  so someone planning work finds it without reading source.

- [ ] **Step 3: Commit**

```bash
git add crates/veda-index/src/gpu_cuvs.rs documentation/2026-08-15-gpu-cuvs-cagra-design.md
git commit -m "docs(gpu): document the seven-step gpu_cuvs build recipe"
```

---

### Task 7: The recall-qualified benchmark

**Files:**
- Create: `crates/veda-index/tests/cagra_gpu_bench.rs`
- Modify: `documentation/2026-08-15-gpu-cuvs-cagra-design.md` (append measured results)

**Latency without recall is meaningless for an ANN index.** No number is published
without the recall it was measured at.

- [ ] **Step 1: Write the benchmark**

`#[ignore]` throughout, matching `crates/veda-index/tests/ann_benchmark.rs`. For 100,000
and 1,000,000 vectors at 128 dimensions, with a fixed xorshift seed:

- CAGRA GPU build time
- CAGRA search latency, mean over 200 queries at k=10
- the same two figures for Veda's existing HNSW on identical data
- **recall@10 for both, against exact brute-force search over the same vectors**

Print one line per case in a single grep-able format, e.g.
`cagra: n=100000 dim=128 build=..s search=..us recall@10=0.xx`.

- [ ] **Step 2: Run it and record every number**

```bash
<env block>
cargo test -p veda-index --features gpu_cuvs --test cagra_gpu_bench -- --ignored --nocapture
```

If the 1M case exhausts GPU memory, report the failure and the memory figure rather than
silently reducing the size — a 48 GB card failing at 1M × 128 would itself be the finding.

- [ ] **Step 3: Append a "Measured result" section to the design doc**

Every figure verbatim, plus one sentence stating plainly whether CAGRA beats HNSW on this
hardware **at comparable recall**. If it does not, say so — the point is a true number,
not a flattering one.

- [ ] **Step 4: Commit**

```bash
git add crates/veda-index/tests/cagra_gpu_bench.rs documentation/2026-08-15-gpu-cuvs-cagra-design.md
git commit -m "test(gpu): recall-qualified CAGRA benchmark on real hardware"
```

---

## Plan self-review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `cargo check --features gpu_cuvs` succeeds | 1–4 |
| Own the handle via `cuvs-sys` (pointer is private) | 1 |
| Build + search on GPU | 2 |
| Persist, reload, identical results | 3 |
| `Mutex`, no `unsafe impl Send/Sync`, poison handled | 4 |
| `unsafe` confined to the `ffi` submodule | 1–3 |
| `serialize_to_hnswlib` removed, not ported | 3 |
| CI compiles the feature | 5 |
| Seven-step recipe documented | 6 |
| Recall-qualified benchmark published | 7 |
| Feature-off build unchanged | 4 (explicit step) |

**Type consistency:** `ffi::CagraHandle` with `new`/`raw`/`Drop` is introduced in Task 1
and used unchanged in 2 and 3. `ffi::check(code, op)` is consistent throughout.
`ffi::build`/`search`/`serialize`/`deserialize` all take `&Resources` first and are
consistent between their defining task and their callers. `CagraInner { resources, handle }`
appears only in Task 4 and matches the handle type from Task 1.

**Two read-then-decide points, both requiring a report rather than a guess:** the
`cuvsError_t` success-variant name (Task 1) and CAGRA's metric support (Task 2). Both
state what to do if the plan's assumption is wrong.
