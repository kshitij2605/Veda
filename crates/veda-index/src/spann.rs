//! SPANN-style object-storage-native vector index.
//!
//! Every other index in this crate is *resident*: `build` takes the whole
//! corpus, `save`/`load` move the whole structure through a local path,
//! and search walks memory. That is why the 2026-08-12 refresh lists an
//! "object-storage-native index tier" as an open gap while MinIO has
//! shipped since day one — MinIO holds **blobs** (images, PDFs, media),
//! and the *index* is still built and searched locally.
//!
//! # What "object-storage-native" means here
//!
//! Two levels, split by size:
//!
//! - **Resident:** `nlist` centroids, `nlist × dim` floats. For 4,096
//!   lists at 768 dimensions that is ~12 MB, and it does **not grow with
//!   the corpus**.
//! - **Remote:** one posting list per centroid, each its own object.
//!   A query finds its `nprobe` nearest centroids and fetches **only
//!   those objects**, by byte range.
//!
//! So resident memory is `O(nlist × dim)`, not `O(n × dim)`, and bytes
//! read per query are `O(nprobe / nlist)` of the corpus rather than all
//! of it. Those two properties are the whole point, so both are asserted
//! in the tests rather than left as prose.
//!
//! # Why this is not [`crate::ivfpq`]
//!
//! `IvfPqIndex` already has k-means centroids and inverted lists, and at
//! a glance looks like the same shape. It keeps `vectors: Vec<Vec<f32>>`
//! — the entire corpus, resident — and serialises the lot through one
//! bincode blob. Retrofitting it would mean changing what every existing
//! IVF-PQ deployment loads. This is a separate index that reuses the
//! *idea* and none of the residency.
//!
//! # Layout
//!
//! ```text
//! {prefix}/manifest.bin     centroids, list offsets, config   (fetched once)
//! {prefix}/lists.bin        every posting list, concatenated  (fetched by range)
//! ```
//!
//! One object holds all lists rather than one object per list. Object
//! stores charge per request and round-trip latency dominates at this
//! size, so `nprobe` ranged reads into a single object beat `nprobe`
//! separate GETs; and a bulk build writes one object instead of
//! thousands. The manifest carries each list's `(offset, len)`, which is
//! what makes the ranged read possible.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::traits::SearchResult;

/// Bytes per stored record in a posting list: `u32` id + `dim` × `f32`.
fn entry_bytes(dim: usize) -> usize {
    4 + dim * 4
}

/// One decoded posting-list entry: the vector's id and its components.
pub type Entry = (u32, Vec<f32>);

/// A decoded posting list. `Arc` so a cache hit hands out a handle
/// rather than cloning the list into every caller.
pub type PostingList = Arc<Vec<Entry>>;

/// Largest lists object written, and therefore the ceiling on how much of the
/// index is held in memory while building.
///
/// The first version concatenated every posting list into ONE object, which
/// meant materialising a second copy of the whole corpus in RAM before the
/// `put` — a 100 GB index needed 100 GB of buffer to publish. Parts cap that
/// at this size regardless of corpus size, while keeping the object count
/// small (`index_size / 64 MB`) so a query still issues one ranged read per
/// probed list rather than hunting through thousands of tiny objects.
const PART_BYTES: usize = 64 * 1024 * 1024;

/// Where one centroid's posting list sits: which part object, and where inside
/// it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
struct ListSpan {
    /// Index of the `lists.{part}.bin` object holding this list. A list is
    /// never split across parts, so one ranged read always suffices.
    part: u32,
    offset: u64,
    len: u64,
    count: u32,
}

/// The small, always-resident half of the index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Manifest {
    centroids: Vec<Vec<f32>>,
    spans: Vec<ListSpan>,
    dimension: u32,
    metric: DistanceMetric,
    total_vectors: u64,
    /// How many `lists.{n}.bin` objects the index was written as.
    part_count: u32,
    /// Format version. bincode is not self-describing, so an explicit
    /// version is the only way a future layout change can be detected
    /// rather than mis-parsed into plausible garbage.
    version: u16,
}

/// Bumped to 2 when posting lists moved from one concatenated object to
/// size-capped parts, so a v1 index is refused rather than read with `part`
/// interpreted as whatever byte happened to be there.
const MANIFEST_VERSION: u16 = 2;

/// Counters proving the index actually reads a small fraction of itself.
///
/// Without these the central claim is unfalsifiable: a query that
/// silently fell back to reading everything would still return correct
/// results, just expensively — the failure mode that is invisible in a
/// recall number. Cheap enough to leave on in production.
#[derive(Debug, Default)]
pub struct IoStats {
    pub range_reads: AtomicU64,
    pub bytes_fetched: AtomicU64,
    pub cache_hits: AtomicU64,
}

impl IoStats {
    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.range_reads.load(Ordering::Relaxed),
            self.bytes_fetched.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
        )
    }
}

/// The minimal object-store surface this index needs.
///
/// Deliberately narrower than `veda_storage::blob_store::BlobStore`:
/// `veda-index` does not depend on `veda-storage` (the dependency runs
/// the other way), and an index needs exactly three operations. A thin
/// adapter in `veda-storage` implements this over `BlobStore`.
pub trait RangeStore: Send + Sync {
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;
    fn get(&self, key: &str) -> Result<Vec<u8>>;
    fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>>;
}

/// An object-storage-native IVF index.
///
/// Construct with [`SpannIndex::build_into`] to write one, or
/// [`SpannIndex::open`] to attach to an existing one. `open` fetches the
/// manifest and nothing else, so attaching to a 100 GB index costs a few
/// megabytes and is independent of corpus size.
pub struct SpannIndex {
    manifest: Manifest,
    store: Arc<dyn RangeStore>,
    prefix: String,
    nprobe: usize,
    /// Decoded posting lists kept after fetching, keyed by centroid.
    /// Bounded by `cache_limit` because an unbounded cache would
    /// reconstruct exactly the residency this index exists to avoid.
    cache: Mutex<HashMap<usize, PostingList>>,
    cache_limit: usize,
    pub stats: IoStats,
}

/// The object store this process serves SPANN indexes from.
///
/// A process-global rather than a parameter because
/// [`crate::traits::VectorIndex`] is the interface the rest of the
/// engine builds and loads indexes through, and its `load(path, metric,
/// dimension)` is a static with nowhere to thread a handle. Threading a
/// store through every index constructor to serve one index type would
/// touch far more than it buys.
///
/// A `RwLock` rather than a `OnceLock` because the gateway swaps the
/// process blob store after startup when `VEDA_BLOB_BACKEND` names an
/// object store (`AppState::load_blob_store_from_env`) — a `OnceLock`
/// would pin the local-filesystem store installed during construction.
static SPANN_STORE: std::sync::RwLock<Option<Arc<dyn RangeStore>>> = std::sync::RwLock::new(None);

/// Install the store SPANN indexes read and write. Called by the server
/// at startup and again whenever the blob backend is reconfigured.
pub fn register_store(store: Arc<dyn RangeStore>) {
    *SPANN_STORE.write().unwrap() = Some(store);
}

/// The registered store, or `None` if nothing has been installed.
pub fn registered_store() -> Option<Arc<dyn RangeStore>> {
    SPANN_STORE.read().unwrap().clone()
}

fn require_store() -> Result<Arc<dyn RangeStore>> {
    registered_store().ok_or_else(|| {
        VedaError::Index(
            "no object store is registered for SPANN indexes; call \
             veda_index::spann::register_store at startup"
                .into(),
        )
    })
}

/// Adapter making any [`veda_storage::blob_store::BlobStore`] usable as
/// a [`RangeStore`].
///
/// This is what connects the index to real infrastructure: wrap the
/// process-wide `ObjectBlobStore` and the posting lists live in MinIO or
/// S3, fetched by HTTP range. Wrap a `LocalBlobStore` and the same index
/// runs against the filesystem, which is what the tests and the
/// single-node path use.
pub struct BlobRangeStore(pub std::sync::Arc<dyn veda_storage::blob_store::BlobStore>);

impl RangeStore for BlobRangeStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.0.put(key, data)
    }
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.0.get(key)
    }
    fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.0.get_range(key, offset, len)
    }
}

/// Hand-written because `Arc<dyn RangeStore>` is not `Debug`, and because
/// dumping centroids and cached lists would bury the fields anyone
/// actually wants when a test fails.
impl std::fmt::Debug for SpannIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpannIndex")
            .field("prefix", &self.prefix)
            .field("vectors", &self.manifest.total_vectors)
            .field("dimension", &self.manifest.dimension)
            .field("nlist", &self.manifest.centroids.len())
            .field("nprobe", &self.nprobe)
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

/// Serialises posting lists into size-capped part objects, flushing each part
/// as it fills.
///
/// Shared by the in-memory and streaming builds so the on-disk layout has
/// exactly one implementation: the streaming path's whole claim is that it
/// produces the *same* index using less memory, and two copies of this logic
/// would let that quietly stop being true.
struct PartWriter<'a> {
    store: &'a dyn RangeStore,
    prefix: &'a str,
    part_bytes: usize,
    buf: Vec<u8>,
    part_index: u32,
    spans: Vec<ListSpan>,
}

impl<'a> PartWriter<'a> {
    fn new(store: &'a dyn RangeStore, prefix: &'a str, part_bytes: usize, nlist: usize) -> Self {
        let part_bytes = part_bytes.max(1);
        Self {
            store,
            prefix,
            part_bytes,
            buf: Vec::with_capacity(part_bytes.min(1 << 20)),
            part_index: 0,
            spans: Vec::with_capacity(nlist),
        }
    }

    /// Append one complete posting list, in centroid order.
    fn push_list(&mut self, entries: &[(u32, &[f32])]) -> Result<()> {
        let dim = entries.first().map(|(_, v)| v.len()).unwrap_or(0);
        let list_len = entries.len() * entry_bytes(dim);
        // A list is never split across parts, so one ranged read always
        // suffices at query time. Flush first when this list would push the
        // part over the cap — unless the part is empty, in which case a single
        // oversized list gets its own part rather than looping.
        if !self.buf.is_empty() && self.buf.len() + list_len > self.part_bytes {
            self.store.put(
                &SpannIndex::part_key(self.prefix, self.part_index),
                &self.buf,
            )?;
            self.buf.clear();
            self.part_index += 1;
        }
        let offset = self.buf.len() as u64;
        for (id, v) in entries {
            self.buf.extend_from_slice(&id.to_le_bytes());
            for &c in *v {
                self.buf.extend_from_slice(&c.to_le_bytes());
            }
        }
        self.spans.push(ListSpan {
            part: self.part_index,
            offset,
            len: self.buf.len() as u64 - offset,
            count: entries.len() as u32,
        });
        Ok(())
    }

    /// Write the final part — including the all-empty-lists case, where
    /// nothing was ever flushed — and hand back the manifest's span table.
    fn finish(self) -> Result<(Vec<ListSpan>, u32)> {
        self.store.put(
            &SpannIndex::part_key(self.prefix, self.part_index),
            &self.buf,
        )?;
        Ok((self.spans, self.part_index + 1))
    }
}

/// A corpus the streaming build can read more than once.
///
/// The point of the trait is the `for_each` shape: a caller backed by segment
/// storage can hand over one vector at a time and never materialise the
/// corpus, which is what keeps [`SpannIndex::build_into_streaming`] off
/// `O(n * dim)`. Re-readable rather than an `Iterator` because the build makes
/// several passes and each must see every vector with the same id.
pub trait VectorSource {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Vector width. Ragged input is rejected by the build, not here.
    fn dimension(&self) -> usize;
    /// Invoke `f` once per vector, ids ascending from 0, every pass.
    fn for_each(&self, f: &mut dyn FnMut(u32, &[f32]) -> Result<()>) -> Result<()>;
}

/// A [`VectorSource`] over a slice already in memory.
///
/// Exists so the streaming path can be tested against the in-memory one, and
/// so a caller that *does* have the whole batch can still use it.
pub struct SliceSource<'a> {
    vectors: &'a [&'a [f32]],
}

impl<'a> SliceSource<'a> {
    pub fn new(vectors: &'a [&'a [f32]]) -> Self {
        Self { vectors }
    }
}

impl VectorSource for SliceSource<'_> {
    fn len(&self) -> usize {
        self.vectors.len()
    }
    fn dimension(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }
    fn for_each(&self, f: &mut dyn FnMut(u32, &[f32]) -> Result<()>) -> Result<()> {
        for (i, v) in self.vectors.iter().enumerate() {
            f(i as u32, v)?;
        }
        Ok(())
    }
}

/// What a streaming build actually did.
///
/// `peak_buffered_bytes` is the number the whole exercise turns on. Without it
/// the memory claim is unfalsifiable: a build that quietly buffered the entire
/// corpus would produce a correct index and a correct recall figure, and look
/// identical from the outside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingBuildStats {
    /// Passes over the source: one to sample, one to count, one per group.
    pub passes: u32,
    /// Centroid groups the build was split into. 1 means it all fit at once.
    pub groups: u32,
    /// High-water mark of posting-list bytes held in memory at any moment.
    pub peak_buffered_bytes: usize,
}

impl SpannIndex {
    fn manifest_key(prefix: &str) -> String {
        format!("{prefix}/manifest.bin")
    }
    fn part_key(prefix: &str, part: u32) -> String {
        format!("{prefix}/lists.{part}.bin")
    }

    /// Build an index over `vectors` and write it to `store` under `prefix`.
    ///
    /// `nlist` is the number of centroids. The rule of thumb is
    /// `~sqrt(n)`; the caller chooses because the right value depends on
    /// how much resident memory the deployment will tolerate.
    pub fn build_into(
        store: Arc<dyn RangeStore>,
        prefix: &str,
        vectors: &[&[f32]],
        nlist: usize,
        metric: DistanceMetric,
    ) -> Result<Self> {
        Self::build_into_with_part_size(store, prefix, vectors, nlist, metric, PART_BYTES)
    }

    /// [`Self::build_into`] with an explicit part-size cap.
    ///
    /// The cap is a parameter rather than only a constant so the streaming
    /// behaviour is testable: at the 64 MB production value, any corpus small
    /// enough to build in a unit test fits in a single part, and a test that
    /// cannot produce a second part cannot prove parts are flushed rather than
    /// accumulated.
    pub fn build_into_with_part_size(
        store: Arc<dyn RangeStore>,
        prefix: &str,
        vectors: &[&[f32]],
        nlist: usize,
        metric: DistanceMetric,
        part_bytes: usize,
    ) -> Result<Self> {
        let part_bytes = part_bytes.max(1);
        if vectors.is_empty() {
            return Err(VedaError::Index("cannot build SPANN over 0 vectors".into()));
        }
        let dim = vectors[0].len();
        if let Some(bad) = vectors.iter().position(|v| v.len() != dim) {
            return Err(VedaError::Index(format!(
                "vector {bad} has dimension {}, expected {dim}",
                vectors[bad].len()
            )));
        }
        let k = nlist.clamp(1, vectors.len());

        let centroids = kmeans(vectors, k, 25, metric);

        // Group ids by nearest centroid.
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); centroids.len()];
        for (id, v) in vectors.iter().enumerate() {
            buckets[nearest_centroid(v, &centroids, metric)].push(id as u32);
        }

        // Serialise the lists into size-capped parts, flushing each as it
        // fills, so peak build memory is bounded by `PART_BYTES` rather than
        // by the corpus. Full f32 vectors are stored, not PQ codes: this
        // index's claim is about *where the bytes live*, and mixing in a lossy
        // encoding would confound a recall measurement with a quantisation one.
        let mut writer = PartWriter::new(store.as_ref(), prefix, part_bytes, centroids.len());
        for bucket in &buckets {
            let entries: Vec<(u32, &[f32])> = bucket
                .iter()
                .map(|&id| (id, vectors[id as usize]))
                .collect();
            writer.push_list(&entries)?;
        }
        let (spans, part_count) = writer.finish()?;

        let manifest = Manifest {
            centroids,
            spans,
            dimension: dim as u32,
            metric,
            total_vectors: vectors.len() as u64,
            part_count,
            version: MANIFEST_VERSION,
        };

        // The manifest is written LAST, after every part is durable. It is the
        // only object `open` reads to find the rest, so publishing it first
        // would make a half-written index look complete.
        let encoded = bincode::serialize(&manifest)
            .map_err(|e| VedaError::Index(format!("manifest serialize failed: {e}")))?;
        store.put(&Self::manifest_key(prefix), &encoded)?;

        Ok(Self::attach(manifest, store, prefix))
    }

    /// Build from a [`VectorSource`] without ever holding the corpus in memory.
    ///
    /// [`Self::build_into`] needs `&[&[f32]]`, so the caller must materialise
    /// every vector first — `O(n * dim)` before the build even starts. That is
    /// the wrong shape for an index whose entire premise is that the data does
    /// not fit in memory. This path keeps peak usage near `budget_bytes` by
    /// trading memory for passes over the source:
    ///
    /// 1. **Sample** — take a strided sample that fits the budget and run
    ///    k-means on it. When the budget admits the whole corpus the sample
    ///    *is* the corpus, and the result is identical to [`Self::build_into`].
    /// 2. **Count** — one pass assigning every vector to its nearest centroid,
    ///    keeping only per-centroid counts (`O(nlist)`). This is what makes the
    ///    memory bound exact rather than a guess: the groups below are cut from
    ///    real list sizes.
    /// 3. **Group** — split the centroids into contiguous runs whose lists fit
    ///    the budget, and make one pass per run, buffering only that run's
    ///    lists before writing them.
    ///
    /// Centroid runs are contiguous and processed in order, so lists reach the
    /// [`PartWriter`] in exactly the order the in-memory build would produce —
    /// the on-disk layout is byte-identical, not merely equivalent.
    ///
    /// The cost is `2 + groups` passes over the source and `groups` times the
    /// assignment work. A single posting list larger than the budget still gets
    /// buffered whole, since a list is never split across parts; `nlist` should
    /// be large enough that no one cluster dominates.
    pub fn build_into_streaming(
        store: Arc<dyn RangeStore>,
        prefix: &str,
        source: &dyn VectorSource,
        nlist: usize,
        metric: DistanceMetric,
        budget_bytes: usize,
    ) -> Result<(Self, StreamingBuildStats)> {
        Self::build_into_streaming_with_part_size(
            store,
            prefix,
            source,
            nlist,
            metric,
            budget_bytes,
            PART_BYTES,
        )
    }

    /// [`Self::build_into_streaming`] with an explicit part-size cap, for the
    /// same reason [`Self::build_into_with_part_size`] has one.
    #[allow(clippy::too_many_arguments)]
    pub fn build_into_streaming_with_part_size(
        store: Arc<dyn RangeStore>,
        prefix: &str,
        source: &dyn VectorSource,
        nlist: usize,
        metric: DistanceMetric,
        budget_bytes: usize,
        part_bytes: usize,
    ) -> Result<(Self, StreamingBuildStats)> {
        let n = source.len();
        if n == 0 {
            return Err(VedaError::Index("cannot build SPANN over 0 vectors".into()));
        }
        let dim = source.dimension();
        if dim == 0 {
            return Err(VedaError::Index(
                "cannot build SPANN over 0-dimensional vectors".into(),
            ));
        }
        // A budget below one entry would divide to a zero-size sample and
        // zero-size groups; one entry is the smallest coherent unit.
        let budget = budget_bytes.max(entry_bytes(dim));

        // --- pass 1: strided sample -> centroids -------------------------
        let sample_cap = (budget / entry_bytes(dim)).clamp(1, n);
        let stride = n.div_ceil(sample_cap);
        let mut sample: Vec<Vec<f32>> = Vec::with_capacity(sample_cap);
        source.for_each(&mut |id, v| {
            if v.len() != dim {
                return Err(VedaError::Index(format!(
                    "vector {id} has dimension {}, expected {dim}",
                    v.len()
                )));
            }
            if (id as usize).is_multiple_of(stride) {
                sample.push(v.to_vec());
            }
            Ok(())
        })?;
        let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
        // Clamped against the sample, not `n`: k-means cannot produce more
        // distinct centroids than it has points to seed from.
        let k = nlist.clamp(1, sample_refs.len());
        let centroids = kmeans(&sample_refs, k, 25, metric);
        drop(sample_refs);
        drop(sample);

        // --- pass 2: exact list sizes, O(nlist) memory --------------------
        let mut counts = vec![0usize; centroids.len()];
        source.for_each(&mut |_, v| {
            counts[nearest_centroid(v, &centroids, metric)] += 1;
            Ok(())
        })?;

        // Cut the centroids into contiguous runs that fit the budget.
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut lo = 0usize;
        let mut acc = 0usize;
        for (c, &count) in counts.iter().enumerate() {
            let bytes = count * entry_bytes(dim);
            if c > lo && acc + bytes > budget {
                groups.push((lo, c));
                lo = c;
                acc = 0;
            }
            acc += bytes;
        }
        groups.push((lo, centroids.len()));

        // --- one pass per group ------------------------------------------
        let mut writer = PartWriter::new(store.as_ref(), prefix, part_bytes, centroids.len());
        let mut peak_buffered_bytes = 0usize;
        for &(lo, hi) in &groups {
            let mut bufs: Vec<Vec<(u32, Vec<f32>)>> =
                (lo..hi).map(|c| Vec::with_capacity(counts[c])).collect();
            let mut buffered = 0usize;
            source.for_each(&mut |id, v| {
                let c = nearest_centroid(v, &centroids, metric);
                if c >= lo && c < hi {
                    bufs[c - lo].push((id, v.to_vec()));
                    buffered += entry_bytes(dim);
                }
                Ok(())
            })?;
            peak_buffered_bytes = peak_buffered_bytes.max(buffered);
            for b in &bufs {
                let entries: Vec<(u32, &[f32])> =
                    b.iter().map(|(id, v)| (*id, v.as_slice())).collect();
                writer.push_list(&entries)?;
            }
        }
        let (spans, part_count) = writer.finish()?;

        let manifest = Manifest {
            centroids,
            spans,
            dimension: dim as u32,
            metric,
            total_vectors: n as u64,
            part_count,
            version: MANIFEST_VERSION,
        };
        // Written last, for the same reason as in the in-memory build: it is
        // the only object `open` reads to find the rest.
        let encoded = bincode::serialize(&manifest)
            .map_err(|e| VedaError::Index(format!("manifest serialize failed: {e}")))?;
        store.put(&Self::manifest_key(prefix), &encoded)?;

        let stats = StreamingBuildStats {
            passes: 2 + groups.len() as u32,
            groups: groups.len() as u32,
            peak_buffered_bytes,
        };
        Ok((Self::attach(manifest, store, prefix), stats))
    }

    /// Attach to an index already in the store, fetching only the manifest.
    pub fn open(store: Arc<dyn RangeStore>, prefix: &str) -> Result<Self> {
        let raw = store.get(&Self::manifest_key(prefix))?;
        let manifest: Manifest = bincode::deserialize(&raw)
            .map_err(|e| VedaError::Index(format!("manifest deserialize failed: {e}")))?;
        if manifest.version != MANIFEST_VERSION {
            return Err(VedaError::Index(format!(
                "SPANN manifest version {} is not supported (expected {MANIFEST_VERSION})",
                manifest.version
            )));
        }
        Ok(Self::attach(manifest, store, prefix))
    }

    fn attach(manifest: Manifest, store: Arc<dyn RangeStore>, prefix: &str) -> Self {
        let nlist = manifest.centroids.len();
        Self {
            manifest,
            store,
            prefix: prefix.to_owned(),
            // A tenth of the lists, so a hot working set is cached while
            // the cache can never hold the whole index.
            cache_limit: (nlist / 10).max(1),
            nprobe: 8.min(nlist),
            cache: Mutex::new(HashMap::new()),
            stats: IoStats::default(),
        }
    }

    /// Set how many centroids a query probes. Higher = better recall,
    /// proportionally more bytes fetched.
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe.clamp(1, self.manifest.centroids.len());
        self
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }
    pub fn len(&self) -> usize {
        self.manifest.total_vectors as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dimension(&self) -> u32 {
        self.manifest.dimension
    }
    pub fn metric(&self) -> DistanceMetric {
        self.manifest.metric
    }
    pub fn nlist(&self) -> usize {
        self.manifest.centroids.len()
    }

    /// Bytes held resident: the centroids and the span table only.
    ///
    /// Exposed so the residency claim is testable rather than asserted;
    /// [`Self::total_index_bytes`] is what it is compared against.
    pub fn resident_bytes(&self) -> usize {
        let c = self.manifest.centroids.len() * self.manifest.dimension as usize * 4;
        let s = self.manifest.spans.len() * std::mem::size_of::<ListSpan>();
        c + s
    }

    /// Bytes the full index occupies in the store.
    pub fn total_index_bytes(&self) -> usize {
        self.manifest.total_vectors as usize * entry_bytes(self.manifest.dimension as usize)
    }

    /// Fetch and decode one posting list, consulting the cache first.
    fn load_list(&self, cid: usize) -> Result<PostingList> {
        if let Some(hit) = self.cache.lock().unwrap().get(&cid) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit.clone());
        }
        let span = self.manifest.spans[cid];
        if span.len == 0 {
            return Ok(Arc::new(Vec::new()));
        }

        // The ranged read: only this list's bytes cross the wire. A list never
        // spans two parts, so this stays exactly one request.
        let raw = self.store.get_range(
            &Self::part_key(&self.prefix, span.part),
            span.offset,
            span.len as usize,
        )?;
        self.stats.range_reads.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_fetched
            .fetch_add(raw.len() as u64, Ordering::Relaxed);

        let dim = self.manifest.dimension as usize;
        let stride = entry_bytes(dim);
        if raw.len() % stride != 0 {
            return Err(VedaError::Index(format!(
                "posting list {cid} is {} bytes, not a multiple of the {stride}-byte entry stride",
                raw.len()
            )));
        }
        let mut out = Vec::with_capacity(raw.len() / stride);
        for chunk in raw.chunks_exact(stride) {
            let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                let b = 4 + d * 4;
                v.push(f32::from_le_bytes([
                    chunk[b],
                    chunk[b + 1],
                    chunk[b + 2],
                    chunk[b + 3],
                ]));
            }
            out.push((id, v));
        }

        let list = Arc::new(out);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= self.cache_limit {
            // Evict an arbitrary entry. Not LRU: keeping recency would
            // need a second structure, and the bound — never holding the
            // whole index — is what actually matters here. Worth
            // revisiting if a workload shows thrashing.
            if let Some(&victim) = cache.keys().next() {
                cache.remove(&victim);
            }
        }
        cache.insert(cid, list.clone());
        Ok(list)
    }

    /// Search for the `k` nearest neighbours, reading only the posting
    /// lists of the `nprobe` closest centroids.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_nprobe(query, k, self.nprobe)
    }

    /// Search at a caller-supplied probe budget, leaving the index's own
    /// `nprobe` untouched.
    ///
    /// This is what backs `VectorIndex::search_with_ef`: the recall /
    /// bytes-fetched trade-off is per query, and mutating shared index
    /// state to express it would make concurrent queries interfere.
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let nprobe = nprobe.clamp(1, self.manifest.centroids.len());
        if query.len() != self.manifest.dimension as usize {
            return Err(VedaError::Index(format!(
                "query has dimension {}, index has {}",
                query.len(),
                self.manifest.dimension
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let metric = self.manifest.metric;
        let mut probes: Vec<(usize, f32)> = self
            .manifest
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, metric)))
            .collect();
        probes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        probes.truncate(nprobe);

        let mut hits: Vec<SearchResult> = Vec::new();
        for (cid, _) in probes {
            for (id, v) in self.load_list(cid)?.iter() {
                hits.push(SearchResult {
                    id: *id,
                    distance: compute_distance(query, v, metric),
                });
            }
        }
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }
}

/// [`crate::traits::VectorIndex`] adapter, so SPANN is selectable as
/// `IndexType::Spann` alongside every other index.
///
/// # Where the corpus lives, and when
///
/// Serving is the property this index exists for: once built, a query
/// touches `nlist` resident centroids and fetches only the posting lists
/// it probes.
///
/// Building has two entry points. [`crate::traits::VectorIndex::build`]
/// receives the whole batch in memory, because that is the shape
/// `SearchExecutor::build_indexes` hands every index — peak build memory is
/// then `O(n * dim)`, same as any other index type. [`Self::build_from_source`]
/// takes a [`VectorSource`] instead and holds only a budgeted slice of the
/// corpus at a time, producing a byte-identical index. What remains before the
/// tier is cheap at ingest as well as at query is a `VectorSource` backed by
/// segment storage and a `build_indexes` path that prefers it; until then the
/// streaming build is reachable but not the default.
pub struct SpannVectorIndex {
    inner: Option<SpannIndex>,
    prefix: String,
    metric: DistanceMetric,
    dimension: u32,
    nlist: Option<usize>,
    nprobe: Option<usize>,
    /// Logical deletes, filtered at search. Matches the convention the
    /// other build-once indexes use (see `RaBitQIndex`): sealed segments
    /// build once and never mutate, and the segment's delete bitmap is
    /// the real authority.
    deleted: std::collections::HashSet<u32>,
}

impl std::fmt::Debug for SpannVectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpannVectorIndex")
            .field("prefix", &self.prefix)
            .field("built", &self.inner.is_some())
            .field("deleted", &self.deleted.len())
            .finish_non_exhaustive()
    }
}

impl SpannVectorIndex {
    /// A SPANN index that will publish under `prefix` when built.
    ///
    /// The caller supplies the prefix so it can be **deterministic** —
    /// `build_indexes` derives it from segment id and field name. A
    /// generated-per-build prefix would leave the previous build's
    /// objects orphaned in the bucket on every restart, since vector
    /// indexes are rebuilt at recovery rather than persisted.
    pub fn new(prefix: impl Into<String>, metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            inner: None,
            prefix: prefix.into(),
            metric,
            dimension,
            nlist: None,
            nprobe: None,
            deleted: std::collections::HashSet::new(),
        }
    }

    /// Override the centroid count. Defaults to `~sqrt(n)`, clamped to
    /// [64, 4096] — enough lists that a probe reads a small slice, few
    /// enough that the resident half stays small.
    pub fn with_nlist(mut self, nlist: usize) -> Self {
        self.nlist = Some(nlist);
        self
    }

    /// Override the probe count. Defaults to 8.
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    fn default_nlist(n: usize) -> usize {
        ((n as f64).sqrt() as usize).clamp(64, 4096).min(n.max(1))
    }

    /// The underlying index, for callers that want [`IoStats`].
    pub fn inner(&self) -> Option<&SpannIndex> {
        self.inner.as_ref()
    }

    fn built(&self) -> Result<&SpannIndex> {
        self.inner
            .as_ref()
            .ok_or_else(|| VedaError::Index("SPANN index has not been built".into()))
    }

    /// Search, then drop logically-deleted ids.
    ///
    /// Oversamples so deletions cannot shorten the result set: asking the
    /// index for exactly `k` and then filtering would silently return
    /// fewer than `k` live hits.
    fn search_live(
        &self,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let idx = self.built()?;
        let want = if self.deleted.is_empty() {
            k
        } else {
            k.saturating_add(self.deleted.len()).min(idx.len().max(k))
        };
        let mut hits = match nprobe {
            Some(p) if p != idx.nprobe() => idx.search_with_nprobe(query, want, p)?,
            _ => idx.search(query, want)?,
        };
        if !self.deleted.is_empty() {
            hits.retain(|r| !self.deleted.contains(&r.id));
        }
        hits.truncate(k);
        Ok(hits)
    }
}

impl SpannVectorIndex {
    /// Build from a [`VectorSource`], holding roughly `budget_bytes` of the
    /// corpus at a time instead of all of it.
    ///
    /// Same index as [`crate::traits::VectorIndex::build`] would produce —
    /// byte-identical when the budget admits the whole corpus — but reachable
    /// without materialising the batch first. Returns the build's
    /// [`StreamingBuildStats`] so a caller can record what the memory
    /// actually was rather than assume it.
    pub fn build_from_source(
        &mut self,
        source: &dyn VectorSource,
        budget_bytes: usize,
    ) -> Result<StreamingBuildStats> {
        let store = require_store()?;
        let nlist = self
            .nlist
            .unwrap_or_else(|| Self::default_nlist(source.len()));
        let (mut idx, stats) = SpannIndex::build_into_streaming(
            store,
            &self.prefix,
            source,
            nlist,
            self.metric,
            budget_bytes,
        )?;
        if let Some(p) = self.nprobe {
            idx = idx.with_nprobe(p);
        }
        self.dimension = idx.dimension();
        self.inner = Some(idx);
        Ok(stats)
    }
}

impl crate::traits::VectorIndex for SpannVectorIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let store = require_store()?;
        let nlist = self
            .nlist
            .unwrap_or_else(|| Self::default_nlist(vectors.len()));
        let mut idx = SpannIndex::build_into(store, &self.prefix, vectors, nlist, self.metric)?;
        if let Some(p) = self.nprobe {
            idx = idx.with_nprobe(p);
        }
        self.dimension = idx.dimension();
        self.inner = Some(idx);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_live(query, k, None)
    }

    /// `ef` is the IVF `nprobe` for this index — the per-query candidate
    /// budget, exactly as the trait describes it.
    fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        self.search_live(query, k, Some(ef.max(1)))
    }

    fn insert(&mut self, id: u32, _vector: &[f32]) -> Result<()> {
        // Same posture as the other build-once indexes: appending would
        // mean re-clustering and rewriting a posting list in the object
        // store. Sealed segments build once and do not mutate, so this
        // only ever un-deletes.
        self.deleted.remove(&id);
        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(|i| i.len().saturating_sub(self.deleted.len()))
            .unwrap_or(0)
    }

    /// Write a pointer, not the index.
    ///
    /// The index already lives in the object store; copying it to a
    /// local path would defeat the entire point. What is persisted is
    /// the prefix needed to find it again.
    fn save(&self, path: &std::path::Path) -> Result<()> {
        let pointer = serde_json::json!({
            "spann_prefix": self.prefix,
            "metric": self.metric,
            "dimension": self.dimension,
        });
        std::fs::write(path, serde_json::to_vec(&pointer)?)?;
        Ok(())
    }

    fn load(path: &std::path::Path, metric: DistanceMetric, dimension: u32) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let pointer: serde_json::Value = serde_json::from_slice(&raw)?;
        let prefix = pointer["spann_prefix"]
            .as_str()
            .ok_or_else(|| VedaError::Index("SPANN pointer file has no spann_prefix".into()))?
            .to_owned();
        let store = require_store()?;
        let inner = SpannIndex::open(store, &prefix)?;
        Ok(Self {
            prefix,
            metric,
            dimension: if dimension == 0 {
                inner.dimension()
            } else {
                dimension
            },
            nlist: None,
            nprobe: None,
            deleted: std::collections::HashSet::new(),
            inner: Some(inner),
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

/// Nearest centroid by index.
fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>], metric: DistanceMetric) -> usize {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = compute_distance(v, c, metric);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Lloyd's k-means with deterministic strided seeding.
///
/// Seeded by stride rather than at random so a rebuild of the same
/// corpus produces the same index — an index whose layout shifts between
/// builds cannot be diffed, cached, or incrementally published to an
/// object store.
fn kmeans(data: &[&[f32]], k: usize, iters: usize, metric: DistanceMetric) -> Vec<Vec<f32>> {
    let dim = data[0].len();
    let stride = (data.len() / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| data[(i * stride) % data.len()].to_vec())
        .collect();

    for _ in 0..iters {
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for v in data {
            let c = nearest_centroid(v, &centroids, metric);
            counts[c] += 1;
            for (j, &x) in v.iter().enumerate() {
                sums[c][j] += x as f64;
            }
        }
        let mut moved = false;
        for c in 0..k {
            if counts[c] == 0 {
                continue; // keep an empty cluster's old position
            }
            for j in 0..dim {
                let next = (sums[c][j] / counts[c] as f64) as f32;
                if (next - centroids[c][j]).abs() > 1e-6 {
                    moved = true;
                }
                centroids[c][j] = next;
            }
        }
        if !moved {
            break;
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    /// In-memory `RangeStore` that counts what actually crosses the wire.
    #[derive(Default)]
    struct MemStore {
        objects: Mutex<Map<String, Vec<u8>>>,
        whole_gets: AtomicU64,
    }

    impl RangeStore for MemStore {
        fn put(&self, key: &str, data: &[u8]) -> Result<()> {
            self.objects
                .lock()
                .unwrap()
                .insert(key.to_owned(), data.to_vec());
            Ok(())
        }
        fn get(&self, key: &str) -> Result<Vec<u8>> {
            self.whole_gets.fetch_add(1, Ordering::Relaxed);
            self.objects
                .lock()
                .unwrap()
                .get(key)
                .cloned()
                .ok_or_else(|| VedaError::Index(format!("missing {key}")))
        }
        fn get_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
            let objects = self.objects.lock().unwrap();
            let data = objects
                .get(key)
                .ok_or_else(|| VedaError::Index(format!("missing {key}")))?;
            let start = (offset as usize).min(data.len());
            let end = (start + len).min(data.len());
            Ok(data[start..end].to_vec())
        }
    }

    /// Serialises every test that touches [`SPANN_STORE`].
    ///
    /// `cargo test` runs tests in parallel threads within one process,
    /// and the store is process-global by design, so a test that clears
    /// it would otherwise pull the store out from under a concurrent
    /// test that had just registered one. Guarding is cheaper than
    /// chasing the intermittent failure later.
    static STORE_GUARD: Mutex<()> = Mutex::new(());

    /// Take the guard, tolerating a previous test's panic having
    /// poisoned it — the lock protects ordering, not invariants.
    fn store_guard() -> std::sync::MutexGuard<'static, ()> {
        STORE_GUARD.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// A clustered corpus, the shape that mis-ranked indexes before (see
    /// `documentation/2026-08-18-corpus-shape-index-ranking.md`).
    fn clustered(n: usize, dim: usize, clusters: usize) -> Vec<Vec<f32>> {
        let mut rng = Lcg(0xC0FFEE);
        let centres: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % clusters];
                (0..dim).map(|d| c[d] + rng.next_f32() * 0.10).collect()
            })
            .collect()
    }

    fn brute_force(data: &[Vec<f32>], q: &[f32], k: usize, m: DistanceMetric) -> Vec<u32> {
        let mut all: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, compute_distance(q, v, m)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(k);
        all.into_iter().map(|(i, _)| i).collect()
    }

    #[test]
    fn recall_is_high_on_a_clustered_corpus_at_768d() {
        // 768d because that is what the product embeds at. A 128d-only
        // guard is exactly how the DiskANN fix shipped broken.
        let dim = 768;
        let data = clustered(2_000, dim, 10);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        let idx = SpannIndex::build_into(store, "idx", &refs, 40, DistanceMetric::Cosine)
            .unwrap()
            .with_nprobe(8);

        let mut hit = 0usize;
        let mut total = 0usize;
        for q in data.iter().take(50) {
            let truth = brute_force(&data, q, 10, DistanceMetric::Cosine);
            let got = idx.search(q, 10).unwrap();
            for r in &got {
                if truth.contains(&r.id) {
                    hit += 1;
                }
            }
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        assert!(recall >= 0.90, "recall@10 = {recall:.3}, expected >= 0.90");
    }

    #[test]
    fn a_query_reads_only_a_small_fraction_of_the_index() {
        // The central claim. A query that quietly fell back to reading
        // everything would still pass the recall test above, so this is
        // the assertion that makes "object-storage-native" falsifiable.
        let dim = 128;
        let data = clustered(4_000, dim, 20);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        let idx = SpannIndex::build_into(store, "idx", &refs, 64, DistanceMetric::Cosine)
            .unwrap()
            .with_nprobe(4);

        idx.search(&data[0], 10).unwrap();
        let (reads, bytes, _) = idx.stats.snapshot();
        let total = idx.total_index_bytes();

        assert!(
            reads <= 4,
            "expected at most nprobe=4 ranged reads, got {reads}"
        );
        assert!(
            (bytes as f64) < total as f64 * 0.25,
            "one query fetched {bytes} of {total} bytes ({:.1}%); \
             the point of this index is that it does not read the corpus",
            100.0 * bytes as f64 / total as f64
        );
    }

    #[test]
    fn resident_memory_does_not_grow_with_the_corpus() {
        // Doubling the corpus at fixed nlist must not move resident
        // bytes at all: residency is O(nlist x dim), which is what makes
        // the tier viable on a node that cannot hold the index.
        let dim = 128;
        let build = |n: usize| {
            let data = clustered(n, dim, 10);
            let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
            let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
            let idx =
                SpannIndex::build_into(store, "idx", &refs, 32, DistanceMetric::Cosine).unwrap();
            (idx.resident_bytes(), idx.total_index_bytes())
        };
        let (small_res, small_total) = build(1_000);
        let (big_res, big_total) = build(4_000);

        assert_eq!(
            small_res, big_res,
            "resident bytes must be independent of corpus size"
        );
        assert!(big_total > small_total * 3, "sanity: the corpus did grow");
        assert!(
            big_res * 4 < big_total,
            "resident {big_res} should be a small fraction of {big_total}"
        );
    }

    #[test]
    fn opening_an_index_fetches_the_manifest_and_nothing_else() {
        let dim = 64;
        let data = clustered(1_000, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();
        SpannIndex::build_into(dynstore.clone(), "idx", &refs, 16, DistanceMetric::Cosine).unwrap();

        store.whole_gets.store(0, Ordering::Relaxed);
        let reopened = SpannIndex::open(dynstore, "idx").unwrap();

        assert_eq!(
            store.whole_gets.load(Ordering::Relaxed),
            1,
            "open must read exactly one object: the manifest"
        );
        assert_eq!(reopened.len(), 1_000);
        assert_eq!(reopened.dimension(), dim as u32);
        let (reads, bytes, _) = reopened.stats.snapshot();
        assert_eq!((reads, bytes), (0, 0), "open must fetch no posting list");
    }

    #[test]
    fn a_reopened_index_returns_the_same_results() {
        let dim = 64;
        let data = clustered(800, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        let built = SpannIndex::build_into(store.clone(), "idx", &refs, 16, DistanceMetric::Cosine)
            .unwrap()
            .with_nprobe(4);
        let opened = SpannIndex::open(store, "idx").unwrap().with_nprobe(4);

        for q in data.iter().take(10) {
            let a = built.search(q, 10).unwrap();
            let b = opened.search(q, 10).unwrap();
            assert_eq!(
                a.iter().map(|r| r.id).collect::<Vec<_>>(),
                b.iter().map(|r| r.id).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_cache_is_bounded_so_it_cannot_become_the_whole_index() {
        // An unbounded cache would silently rebuild the residency this
        // index exists to avoid, and every other test here would still
        // pass.
        let dim = 32;
        let data = clustered(2_000, dim, 20);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        let idx = SpannIndex::build_into(store, "idx", &refs, 50, DistanceMetric::Cosine)
            .unwrap()
            .with_nprobe(50); // probe everything, every query

        for q in data.iter().take(30) {
            idx.search(q, 5).unwrap();
        }
        let cached = idx.cache.lock().unwrap().len();
        assert!(
            cached <= idx.cache_limit,
            "cache holds {cached} lists, limit is {}",
            idx.cache_limit
        );
        assert!(
            idx.cache_limit < idx.nlist(),
            "the cache must never be able to hold every list"
        );
    }

    #[test]
    fn higher_nprobe_trades_bytes_for_recall() {
        let dim = 128;
        let data = clustered(2_000, dim, 16);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let measure = |nprobe: usize| {
            let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
            let idx = SpannIndex::build_into(store, "idx", &refs, 32, DistanceMetric::Cosine)
                .unwrap()
                .with_nprobe(nprobe);
            let mut hit = 0;
            let mut total = 0;
            for q in data.iter().take(30) {
                let truth = brute_force(&data, q, 10, DistanceMetric::Cosine);
                for r in idx.search(q, 10).unwrap() {
                    if truth.contains(&r.id) {
                        hit += 1;
                    }
                }
                total += truth.len();
            }
            let (_, bytes, _) = idx.stats.snapshot();
            (hit as f64 / total as f64, bytes)
        };

        let (r1, b1) = measure(1);
        let (r8, b8) = measure(8);
        assert!(
            r8 > r1,
            "nprobe=8 recall {r8:.3} should beat nprobe=1 {r1:.3}"
        );
        assert!(b8 > b1, "and it should cost more bytes: {b8} vs {b1}");
    }

    #[test]
    fn lists_are_written_as_size_capped_parts() {
        // The build must not hold a second copy of the corpus. Asserting the
        // largest object written is the observable proxy: the old
        // one-object-per-index layout buffered the whole thing before its
        // single `put`, so a corpus larger than the cap proves parts are being
        // flushed rather than accumulated.
        //
        // `MemStore` records every object, so the largest is the high-water
        // mark of the build's buffer.
        let dim = 64;
        let stride = entry_bytes(dim); // 4 + 64*4 = 260 bytes
        let n = 4_000; // 1.04 MB of lists
        const TEST_PART_BYTES: usize = 200 * 1024; // ~6 parts
        let data = clustered(n, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();

        let idx = SpannIndex::build_into_with_part_size(
            dynstore,
            "parts",
            &refs,
            64,
            DistanceMetric::Cosine,
            TEST_PART_BYTES,
        )
        .unwrap();

        let objects = store.objects.lock().unwrap();
        let list_objects: Vec<(&String, usize)> = objects
            .iter()
            .filter(|(k, _)| k.contains("lists."))
            .map(|(k, v)| (k, v.len()))
            .collect();
        assert!(
            list_objects.len() >= 2,
            "expected the corpus to span several parts, got {:?}",
            list_objects.iter().map(|(k, _)| k).collect::<Vec<_>>()
        );
        let total: usize = list_objects.iter().map(|(_, n)| n).sum();
        assert_eq!(
            total,
            n * stride,
            "every vector must be written exactly once across the parts"
        );
        // No part exceeds the cap, except where a single oversized list needs
        // its own — which cannot happen at this cluster size.
        for (k, len) in &list_objects {
            assert!(*len <= TEST_PART_BYTES, "part {k} is {len} bytes");
        }
        assert_eq!(idx.len(), n);
    }

    #[test]
    fn a_list_never_straddles_two_parts() {
        // The whole point of parts is that a probe stays one ranged read. If a
        // list were split, `load_list` would silently return a truncated list
        // and recall would drop with nothing failing.
        let dim = 64;
        let data = clustered(3_000, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();
        let idx = SpannIndex::build_into_with_part_size(
            dynstore,
            "straddle",
            &refs,
            32,
            DistanceMetric::Cosine,
            100 * 1024,
        )
        .unwrap();

        let objects = store.objects.lock().unwrap();
        for (i, span) in idx.manifest.spans.iter().enumerate() {
            let key = SpannIndex::part_key("straddle", span.part);
            let part_len = objects.get(&key).map(|v| v.len()).unwrap_or(0);
            assert!(
                span.offset + span.len <= part_len as u64,
                "list {i} runs past the end of {key}: offset {} + len {} > {part_len}",
                span.offset,
                span.len
            );
        }
    }

    #[test]
    fn a_manifest_from_a_future_version_is_refused() {
        // bincode is not self-describing, so without the explicit
        // version check a changed layout would deserialize into
        // plausible garbage rather than failing.
        let dim = 16;
        let data = clustered(100, dim, 4);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();
        SpannIndex::build_into(dynstore.clone(), "idx", &refs, 8, DistanceMetric::Cosine).unwrap();

        let key = SpannIndex::manifest_key("idx");
        let mut m: Manifest = bincode::deserialize(&store.get(&key).unwrap()).unwrap();
        m.version = MANIFEST_VERSION + 1;
        store.put(&key, &bincode::serialize(&m).unwrap()).unwrap();

        let err = SpannIndex::open(dynstore, "idx").unwrap_err();
        assert!(
            format!("{err}").contains("not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn it_runs_against_a_real_blob_store_through_the_adapter() {
        // The unit tests above use an in-memory store, which proves the
        // algorithm and not the plumbing. This drives the same index
        // through BlobRangeStore over a real LocalBlobStore, so the
        // ranged-read path that S3/MinIO uses is actually exercised —
        // the same code path, a different backend.
        use veda_storage::blob_store::LocalBlobStore;

        let dir = tempfile::tempdir().unwrap();
        let blobs: std::sync::Arc<dyn veda_storage::blob_store::BlobStore> =
            std::sync::Arc::new(LocalBlobStore::new(dir.path()).unwrap());
        let store: Arc<dyn RangeStore> = Arc::new(BlobRangeStore(blobs));

        let dim = 128;
        let data = clustered(1_500, dim, 12);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let idx =
            SpannIndex::build_into(store.clone(), "spann/c1", &refs, 32, DistanceMetric::Cosine)
                .unwrap()
                .with_nprobe(6);
        let reopened = SpannIndex::open(store, "spann/c1").unwrap().with_nprobe(6);

        for q in data.iter().take(10) {
            assert_eq!(
                idx.search(q, 10)
                    .unwrap()
                    .iter()
                    .map(|r| r.id)
                    .collect::<Vec<_>>(),
                reopened
                    .search(q, 10)
                    .unwrap()
                    .iter()
                    .map(|r| r.id)
                    .collect::<Vec<_>>(),
            );
        }
        // Per query, not cumulative. `IoStats` counts for the lifetime of
        // the index, so ten queries at nprobe=6 of nlist=32 legitimately
        // total ~190% of the index — comparing that against the index
        // size measures nothing but the query count.
        let (reads, bytes, _) = reopened.stats.snapshot();
        assert!(reads > 0, "the adapter must actually issue ranged reads");
        let per_query = bytes as f64 / 10.0;
        let total = reopened.total_index_bytes() as f64;
        assert!(
            per_query < total * 0.35,
            "each query fetched {per_query:.0} of {total:.0} bytes ({:.1}%); \
             nprobe=6 of nlist=32 should land near 19%",
            100.0 * per_query / total
        );
    }

    /// Prints the numbers behind the claims. Not a pass/fail gate — the
    /// assertions above are — but a `PASS` with no figures is how
    /// unverifiable performance claims get written, so the figures are
    /// produced by something runnable.
    ///
    /// `cargo test -p veda-index --release --lib spann::tests::report -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn report_the_object_storage_tradeoff() {
        for &(n, dim, nlist) in &[(10_000usize, 768usize, 100usize), (50_000, 768, 224)] {
            let data = clustered(n, dim, 20);
            let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
            println!("\n=== n={n} dim={dim} nlist={nlist} ===");
            for &nprobe in &[1usize, 4, 8, 16] {
                let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
                let idx =
                    SpannIndex::build_into(store, "idx", &refs, nlist, DistanceMetric::Cosine)
                        .unwrap()
                        .with_nprobe(nprobe);
                const QUERIES: usize = 50;
                let mut hit = 0usize;
                let mut total = 0usize;
                // Ground truth is computed OUTSIDE the timed section: a
                // brute-force scan of the whole corpus costs far more
                // than the index search it is grading, and including it
                // would report the baseline's latency as the index's.
                let truths: Vec<Vec<u32>> = data
                    .iter()
                    .take(QUERIES)
                    .map(|q| brute_force(&data, q, 10, DistanceMetric::Cosine))
                    .collect();

                let t0 = std::time::Instant::now();
                for q in data.iter().take(QUERIES) {
                    std::hint::black_box(idx.search(q, 10).unwrap());
                }
                let secs = t0.elapsed().as_secs_f64();

                for (q, truth) in data.iter().take(QUERIES).zip(&truths) {
                    for r in idx.search(q, 10).unwrap() {
                        if truth.contains(&r.id) {
                            hit += 1;
                        }
                    }
                    total += truth.len();
                }

                // IoStats are cumulative over the index's lifetime, so
                // everything below is divided by the query count. The
                // raw counter over 50 queries reads as ">100% of the
                // index", which is a statement about the loop, not the
                // index.
                let (reads, bytes, hits) = idx.stats.snapshot();
                let queries_run = (QUERIES * 2) as f64; // timed pass + recall pass
                let per_query_bytes = bytes as f64 / queries_run;
                println!(
                    "nprobe={nprobe:2}  recall@10={:.3}  reads/query={:.1}  \
                     fetched/query={:.2} MB ({:.2}% of index)  cache_hits={hits:4}  \
                     resident={:.2} MB of {:.1} MB ({:.2}%)  {:.2} ms/query",
                    hit as f64 / total as f64,
                    reads as f64 / queries_run,
                    per_query_bytes / 1e6,
                    100.0 * per_query_bytes / idx.total_index_bytes() as f64,
                    idx.resident_bytes() as f64 / 1e6,
                    idx.total_index_bytes() as f64 / 1e6,
                    100.0 * idx.resident_bytes() as f64 / idx.total_index_bytes() as f64,
                    1000.0 * secs / QUERIES as f64,
                );
            }
        }
    }

    #[test]
    fn the_vector_index_adapter_builds_searches_and_round_trips() {
        let _g = store_guard();
        use crate::traits::VectorIndex;
        use veda_storage::blob_store::LocalBlobStore;

        let dir = tempfile::tempdir().unwrap();
        let blobs: std::sync::Arc<dyn veda_storage::blob_store::BlobStore> =
            std::sync::Arc::new(LocalBlobStore::new(dir.path()).unwrap());
        register_store(Arc::new(BlobRangeStore(blobs)));

        let dim = 64;
        let data = clustered(1_200, dim, 10);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let mut idx = SpannVectorIndex::new("seg-1/embedding", DistanceMetric::Cosine, dim as u32)
            .with_nlist(24)
            .with_nprobe(6);
        idx.build(&refs).unwrap();

        assert_eq!(idx.len(), 1_200);
        assert_eq!(idx.dimension(), dim as u32);
        let hits = idx.search(&data[5], 10).unwrap();
        assert_eq!(hits.len(), 10);
        assert_eq!(hits[0].id, 5, "a vector's nearest neighbour is itself");

        // save() writes a pointer, not the index: the whole point is
        // that the index does not come back to local disk.
        let ptr = dir.path().join("spann.idx");
        idx.save(&ptr).unwrap();
        let bytes = std::fs::metadata(&ptr).unwrap().len();
        assert!(
            bytes < 512,
            "pointer file is {bytes} bytes; it should not hold the index"
        );

        let reloaded = SpannVectorIndex::load(&ptr, DistanceMetric::Cosine, dim as u32).unwrap();
        assert_eq!(
            reloaded
                .search(&data[5], 10)
                .unwrap()
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>(),
            hits.iter().map(|r| r.id).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn deleted_ids_are_filtered_without_shortening_the_result_set() {
        let _g = store_guard();
        use crate::traits::VectorIndex;
        use veda_storage::blob_store::LocalBlobStore;

        let dir = tempfile::tempdir().unwrap();
        let blobs: std::sync::Arc<dyn veda_storage::blob_store::BlobStore> =
            std::sync::Arc::new(LocalBlobStore::new(dir.path()).unwrap());
        register_store(Arc::new(BlobRangeStore(blobs)));

        let dim = 32;
        let data = clustered(600, dim, 6);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut idx = SpannVectorIndex::new("seg-del/f", DistanceMetric::Cosine, dim as u32)
            .with_nlist(12)
            .with_nprobe(12);
        idx.build(&refs).unwrap();

        let before = idx.search(&data[0], 5).unwrap();
        let victim = before[1].id;
        idx.delete(victim).unwrap();

        let after = idx.search(&data[0], 5).unwrap();
        assert!(
            !after.iter().any(|r| r.id == victim),
            "deleted id came back"
        );
        assert_eq!(
            after.len(),
            5,
            "oversampling must keep the result set full after a delete"
        );
        assert_eq!(idx.len(), 599);
    }

    #[test]
    fn search_with_ef_changes_the_probe_budget_without_mutating_the_index() {
        let _g = store_guard();
        use crate::traits::VectorIndex;
        let dim = 64;
        let data = clustered(2_000, dim, 16);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        register_store(store);

        let mut idx = SpannVectorIndex::new("seg-ef/f", DistanceMetric::Cosine, dim as u32)
            .with_nlist(32)
            .with_nprobe(1);
        idx.build(&refs).unwrap();

        let (_, narrow, _) = idx.inner().unwrap().stats.snapshot();
        idx.search_with_ef(&data[0], 10, 16).unwrap();
        let (_, wide, _) = idx.inner().unwrap().stats.snapshot();
        assert!(
            wide > narrow,
            "ef=16 must fetch more than the default nprobe=1"
        );

        // The index's own nprobe is untouched, so a concurrent query at
        // the default budget is unaffected.
        assert_eq!(idx.inner().unwrap().nprobe(), 1);
    }

    #[test]
    fn building_without_a_registered_store_fails_with_a_clear_message() {
        // Fails loudly rather than silently falling back to a local
        // index, which would look like it worked and quietly reinstate
        // the residency this tier exists to remove.
        let _g = store_guard();
        let saved = registered_store();
        *SPANN_STORE.write().unwrap() = None;
        use crate::traits::VectorIndex;
        let v = vec![0.1f32; 8];
        let refs: Vec<&[f32]> = vec![v.as_slice()];
        let mut idx = SpannVectorIndex::new("nostore/f", DistanceMetric::Cosine, 8);
        let err = idx.build(&refs).unwrap_err();
        // Restore before asserting, so a failure here does not leave the
        // global cleared for whatever runs next.
        if let Some(s) = saved {
            register_store(s);
        }
        assert!(
            format!("{err}").contains("no object store is registered"),
            "got: {err}"
        );
    }

    #[test]
    fn building_over_nothing_or_ragged_input_is_refused() {
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        assert!(
            SpannIndex::build_into(store.clone(), "e", &[], 4, DistanceMetric::Cosine).is_err()
        );

        let a = vec![0.0f32; 8];
        let b = vec![0.0f32; 4];
        let ragged: Vec<&[f32]> = vec![a.as_slice(), b.as_slice()];
        let err =
            SpannIndex::build_into(store, "r", &ragged, 2, DistanceMetric::Cosine).unwrap_err();
        assert!(format!("{err}").contains("dimension"), "got: {err}");
    }

    /// Every object an index wrote, so two builds can be compared whole.
    fn dump(store: &MemStore, prefix: &str) -> Vec<(String, Vec<u8>)> {
        let objects = store.objects.lock().unwrap();
        let mut out: Vec<(String, Vec<u8>)> = objects
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{prefix}/")))
            .map(|(k, v)| (k[prefix.len() + 1..].to_owned(), v.clone()))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_generous_budget_reproduces_the_in_memory_index_byte_for_byte() {
        // The streaming path's claim is that it builds the *same* index with
        // less memory. Equivalent recall would not show that — a different but
        // similarly good layout would pass. Identical bytes is the real check.
        let dim = 64;
        let data = clustered(600, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();

        SpannIndex::build_into(dynstore.clone(), "mem", &refs, 24, DistanceMetric::Cosine).unwrap();
        let (_, stats) = SpannIndex::build_into_streaming(
            dynstore,
            "stream",
            &SliceSource::new(&refs),
            24,
            DistanceMetric::Cosine,
            usize::MAX / 4,
        )
        .unwrap();

        assert_eq!(stats.groups, 1, "a budget this large should not split");
        assert_eq!(dump(&store, "mem"), dump(&store, "stream"));
    }

    #[test]
    fn a_tight_budget_splits_into_groups_and_stays_inside_it() {
        let dim = 64;
        let n = 600;
        let data = clustered(n, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        let dynstore: Arc<dyn RangeStore> = store.clone();

        let corpus_bytes = n * entry_bytes(dim);
        let budget = corpus_bytes / 8;
        let (idx, stats) = SpannIndex::build_into_streaming(
            dynstore,
            "idx",
            &SliceSource::new(&refs),
            24,
            DistanceMetric::Cosine,
            budget,
        )
        .unwrap();

        assert!(
            stats.groups > 1,
            "a budget of 1/8 the corpus must split; got {} group(s)",
            stats.groups
        );
        assert_eq!(
            stats.passes,
            2 + stats.groups,
            "one sample, one count, one per group"
        );

        // A single posting list is never split across parts, so a cluster
        // larger than the budget is buffered whole. That is the one documented
        // way to exceed it — so the bound is checked against it rather than
        // waved away.
        let biggest_list = idx
            .manifest
            .spans
            .iter()
            .map(|s| s.len as usize)
            .max()
            .unwrap();
        let ceiling = budget.max(biggest_list);
        assert!(
            stats.peak_buffered_bytes <= ceiling,
            "peak {} exceeded ceiling {} (budget {budget}, largest list {biggest_list})",
            stats.peak_buffered_bytes,
            ceiling
        );
        assert_eq!(idx.len(), n, "every vector must still be indexed");
    }

    #[test]
    fn the_streaming_build_never_buffers_the_whole_corpus() {
        // The reason the path exists. A build that quietly materialised
        // everything would still return a correct index and correct recall,
        // so this is the only assertion that can catch it.
        let dim = 128;
        let n = 1_200;
        let data = clustered(n, dim, 12);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());

        let corpus_bytes = n * entry_bytes(dim);
        let (_, stats) = SpannIndex::build_into_streaming(
            store,
            "idx",
            &SliceSource::new(&refs),
            40,
            DistanceMetric::Cosine,
            corpus_bytes / 10,
        )
        .unwrap();

        assert!(
            stats.peak_buffered_bytes * 4 < corpus_bytes,
            "peak {} is not meaningfully below the corpus {corpus_bytes}",
            stats.peak_buffered_bytes
        );
    }

    #[test]
    fn a_streaming_built_index_is_searchable_at_768d() {
        // 768d for the same reason the in-memory recall test uses it: a
        // 128d-only guard is how the DiskANN fix shipped broken.
        let dim = 768;
        let data = clustered(2_000, dim, 10);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());

        let (idx, stats) = SpannIndex::build_into_streaming(
            store,
            "idx",
            &SliceSource::new(&refs),
            40,
            DistanceMetric::Cosine,
            2_000 * entry_bytes(dim) / 6,
        )
        .unwrap();
        let idx = idx.with_nprobe(8);
        assert!(stats.groups > 1, "expected a split at this budget");

        let mut hit = 0usize;
        let mut total = 0usize;
        for q in data.iter().take(50) {
            let truth = brute_force(&data, q, 10, DistanceMetric::Cosine);
            for r in &idx.search(q, 10).unwrap() {
                if truth.contains(&r.id) {
                    hit += 1;
                }
            }
            total += truth.len();
        }
        let recall = hit as f64 / total as f64;
        // Lower than the in-memory 0.90 bar on purpose: centroids come from a
        // sixth of the corpus here, and pretending sampling is free would make
        // this a test of nothing.
        assert!(recall >= 0.80, "recall@10 = {recall:.3}, expected >= 0.80");
    }

    #[test]
    fn streaming_over_nothing_or_ragged_input_is_refused() {
        let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
        let empty: Vec<&[f32]> = Vec::new();
        assert!(
            SpannIndex::build_into_streaming(
                store.clone(),
                "idx",
                &SliceSource::new(&empty),
                4,
                DistanceMetric::Cosine,
                1 << 20,
            )
            .is_err()
        );

        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0];
        let ragged: Vec<&[f32]> = vec![&a, &b];
        let err = SpannIndex::build_into_streaming(
            store,
            "idx",
            &SliceSource::new(&ragged),
            2,
            DistanceMetric::Cosine,
            1 << 20,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("dimension"), "unhelpful error: {err}");
    }

    #[test]
    fn the_adapter_streaming_build_matches_its_in_memory_build() {
        use crate::traits::VectorIndex;

        let dim = 64;
        let data = clustered(500, dim, 8);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let store = Arc::new(MemStore::default());
        register_store(store.clone());

        let mut batch =
            SpannVectorIndex::new("adapter-mem", DistanceMetric::Cosine, dim as u32).with_nlist(20);
        batch.build(&refs).unwrap();

        let mut streamed =
            SpannVectorIndex::new("adapter-stream", DistanceMetric::Cosine, dim as u32)
                .with_nlist(20);
        let stats = streamed
            .build_from_source(&SliceSource::new(&refs), 500 * entry_bytes(dim) / 5)
            .unwrap();

        assert!(
            stats.groups > 1,
            "expected a split at a fifth of the corpus"
        );
        assert_eq!(batch.len(), streamed.len());

        // Same answers from both, which is what a caller switching entry
        // points actually cares about.
        for q in data.iter().take(20) {
            let a = batch.search(q, 10).unwrap();
            let b = streamed.search(q, 10).unwrap();
            assert_eq!(
                a.iter().map(|r| r.id).collect::<Vec<_>>(),
                b.iter().map(|r| r.id).collect::<Vec<_>>()
            );
        }
    }

    /// What the streaming build actually costs and saves, at 768d.
    ///
    /// Ignored because it is a measurement, not a guard: it prints the
    /// tradeoff rather than asserting a threshold that would drift.
    /// `cargo test -p veda-index --lib report_the_streaming_build -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn report_the_streaming_build_memory_tradeoff() {
        let dim = 768;
        let n = 20_000;
        let data = clustered(n, dim, 40);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let corpus = n * entry_bytes(dim);

        println!("\ncorpus: {n} x {dim}d = {:.1} MB\n", corpus as f64 / 1e6);
        println!(
            "{:>10}  {:>7}  {:>7}  {:>12}  {:>9}  {:>11}",
            "budget", "groups", "passes", "peak buffered", "vs corpus", "big list"
        );
        for divisor in [1usize, 4, 16, 64] {
            let budget = corpus / divisor;
            let store: Arc<dyn RangeStore> = Arc::new(MemStore::default());
            let (idx, s) = SpannIndex::build_into_streaming(
                store,
                "idx",
                &SliceSource::new(&refs),
                200,
                DistanceMetric::Cosine,
                budget,
            )
            .unwrap();
            let biggest = idx
                .manifest
                .spans
                .iter()
                .map(|sp| sp.len as usize)
                .max()
                .unwrap_or(0);
            println!(
                "{:>9.1}M  {:>7}  {:>7}  {:>11.1}M  {:>8.1}%  {:>10.1}M",
                budget as f64 / 1e6,
                s.groups,
                s.passes,
                s.peak_buffered_bytes as f64 / 1e6,
                100.0 * s.peak_buffered_bytes as f64 / corpus as f64,
                biggest as f64 / 1e6,
            );
        }
        println!(
            "\nin-memory build holds the whole {:.1} MB corpus before it starts.",
            corpus as f64 / 1e6
        );
        println!(
            "the floor is the largest posting list, not the budget: a list is never\n\
             split across parts, so it is always buffered whole. nlist governs that\n\
             floor -- a budget below it just costs extra passes for no saving."
        );
    }
}
