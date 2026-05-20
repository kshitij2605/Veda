use std::collections::HashSet;
use std::fs;
use std::path::Path;

use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::traits::{SearchResult, VectorIndex};

/// Squared L2 distance between two slices. Used for PQ sub-vector distances
/// because squared L2 is additive across sub-spaces (unlike L2 with sqrt).
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// IVF-PQ (Inverted File with Product Quantization) index for memory-efficient
/// approximate nearest neighbor search on large-scale datasets.
///
/// Uses residual PQ: vectors are encoded as PQ codes of the residual
/// (vector - IVF centroid), which significantly improves quantization quality.
///
/// Search uses Asymmetric Distance Computation (ADC) on residuals.
pub struct IvfPqIndex {
    // IVF
    centroids: Vec<Vec<f32>>,
    inverted_lists: Vec<Vec<u32>>,
    vector_centroid: Vec<u32>, // vector_id -> centroid_id

    // PQ
    pq_centroids: Vec<Vec<Vec<f32>>>, // [sub_quantizer][code][sub_vector_dim]
    pq_codes: Vec<Vec<u8>>,           // [vector_id][sub_quantizer]

    // Config
    nlist: usize,
    nprobe: usize,
    m_pq: usize,
    nbits: usize,

    // Metadata
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    deleted: HashSet<u32>,
}

#[derive(Serialize, Deserialize)]
struct IvfPqIndexData {
    centroids: Vec<Vec<f32>>,
    inverted_lists: Vec<Vec<u32>>,
    vector_centroid: Vec<u32>,
    pq_centroids: Vec<Vec<Vec<f32>>>,
    pq_codes: Vec<Vec<u8>>,
    nlist: usize,
    nprobe: usize,
    m_pq: usize,
    nbits: usize,
    vectors: Vec<Vec<f32>>,
    metric: DistanceMetric,
    dimension: u32,
    deleted: Vec<u32>,
}

impl IvfPqIndex {
    /// Create a new IVF-PQ index with default parameters.
    ///
    /// Defaults: nlist=256, nprobe=8, m_pq=8, nbits=8.
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vector_centroid: Vec::new(),
            pq_centroids: Vec::new(),
            pq_codes: Vec::new(),
            nlist: 256,
            nprobe: 8,
            m_pq: 8,
            nbits: 8,
            vectors: Vec::new(),
            metric,
            dimension,
            deleted: HashSet::new(),
        }
    }

    /// Create a new IVF-PQ index with custom parameters.
    pub fn with_params(
        metric: DistanceMetric,
        dimension: u32,
        nlist: usize,
        nprobe: usize,
        m_pq: usize,
        nbits: usize,
    ) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            vector_centroid: Vec::new(),
            pq_centroids: Vec::new(),
            pq_codes: Vec::new(),
            nlist,
            nprobe,
            m_pq,
            nbits,
            vectors: Vec::new(),
            metric,
            dimension,
            deleted: HashSet::new(),
        }
    }

    /// Set the nprobe parameter at runtime.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe;
    }

    /// Get the current nprobe value.
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Number of PQ centroids per sub-quantizer (2^nbits).
    fn num_pq_centroids(&self) -> usize {
        1 << self.nbits
    }

    /// Dimension of each PQ sub-vector.
    fn sub_dim(&self) -> usize {
        self.dimension as usize / self.m_pq
    }

    /// Compute the residual: vector - centroid.
    fn compute_residual(vector: &[f32], centroid: &[f32]) -> Vec<f32> {
        vector.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }

    /// Train IVF centroids using k-means on all vectors.
    fn train_ivf(&mut self, vectors: &[&[f32]]) {
        let k = self.nlist.min(vectors.len());
        self.centroids = kmeans(vectors, k, 25);
        self.nlist = k;
    }

    /// Assign each vector to its nearest IVF centroid.
    fn assign_to_centroids(&mut self, vectors: &[&[f32]]) {
        self.inverted_lists = vec![Vec::new(); self.nlist];
        self.vector_centroid = Vec::with_capacity(vectors.len());
        for (id, v) in vectors.iter().enumerate() {
            let centroid_id = nearest_centroid(v, &self.centroids, self.metric);
            self.inverted_lists[centroid_id].push(id as u32);
            self.vector_centroid.push(centroid_id as u32);
        }
    }

    /// Train PQ codebooks on residuals (vector - centroid).
    fn train_pq(&mut self, vectors: &[&[f32]]) {
        let sub_dim = self.sub_dim();
        let num_codes = self.num_pq_centroids();

        // Compute all residuals.
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let c = self.vector_centroid[i] as usize;
                Self::compute_residual(v, &self.centroids[c])
            })
            .collect();

        self.pq_centroids = Vec::with_capacity(self.m_pq);
        for sq in 0..self.m_pq {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_vecs: Vec<Vec<f32>> = residuals.iter().map(|r| r[start..end].to_vec()).collect();
            let sub_refs: Vec<&[f32]> = sub_vecs.iter().map(|v| v.as_slice()).collect();
            let centroids = kmeans(&sub_refs, num_codes.min(sub_refs.len()), 25);
            self.pq_centroids.push(centroids);
        }
    }

    /// Encode all vectors with PQ codes (on residuals).
    fn encode_vectors(&mut self, vectors: &[&[f32]]) {
        let sub_dim = self.sub_dim();
        self.pq_codes = Vec::with_capacity(vectors.len());

        for (i, v) in vectors.iter().enumerate() {
            let c = self.vector_centroid[i] as usize;
            let residual = Self::compute_residual(v, &self.centroids[c]);
            let codes = self.encode_residual(&residual, sub_dim);
            self.pq_codes.push(codes);
        }
    }

    /// Encode a residual vector into PQ codes.
    fn encode_residual(&self, residual: &[f32], sub_dim: usize) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m_pq);
        for sq in 0..self.m_pq {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_vec = &residual[start..end];
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;
            for (code, centroid) in self.pq_centroids[sq].iter().enumerate() {
                let d = l2_squared(sub_vec, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_code = code as u8;
                }
            }
            codes.push(best_code);
        }
        codes
    }

    /// Precompute PQ distance table for the query residual (query - centroid).
    /// Returns table[sub_quantizer][code] = squared L2 distance.
    fn precompute_distance_table(&self, query_residual: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.sub_dim();
        let mut table = Vec::with_capacity(self.m_pq);

        for sq in 0..self.m_pq {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let query_sub = &query_residual[start..end];
            let dists: Vec<f32> = self.pq_centroids[sq]
                .iter()
                .map(|c| l2_squared(query_sub, c))
                .collect();
            table.push(dists);
        }

        table
    }

    /// Compute approximate distance for a vector using the precomputed distance table.
    fn adc_distance(table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (sq, &code) in codes.iter().enumerate() {
            dist += table[sq][code as usize];
        }
        dist
    }
}

/// Find the index of the nearest centroid to a vector.
fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>], metric: DistanceMetric) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = compute_distance(vector, c, metric);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Simple k-means clustering.
///
/// Initializes with k random points from `data`, then iterates assignment and
/// centroid recomputation until convergence or `max_iter` iterations.
fn kmeans(data: &[&[f32]], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(data.len());
    let dim = data[0].len();
    let mut rng = thread_rng();

    // Random initialization: pick k distinct points.
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(&mut rng);
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| data[i].to_vec()).collect();

    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid.
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = l2_squared(point, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c][j] += point[j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = sums[c][j] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

impl VectorIndex for IvfPqIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        // Reset state.
        self.vectors.clear();
        self.deleted.clear();
        self.centroids.clear();
        self.inverted_lists.clear();
        self.vector_centroid.clear();
        self.pq_centroids.clear();
        self.pq_codes.clear();

        if vectors.is_empty() {
            return Ok(());
        }

        // Validate dimensions.
        for v in vectors {
            if v.len() as u32 != self.dimension {
                return Err(VedaError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len() as u32,
                });
            }
        }

        if !(self.dimension as usize).is_multiple_of(self.m_pq) {
            return Err(VedaError::Index(format!(
                "Dimension {} must be divisible by m_pq {}",
                self.dimension, self.m_pq
            )));
        }

        // Store vectors.
        self.vectors = vectors.iter().map(|v| v.to_vec()).collect();

        // 1. Train IVF centroids.
        self.train_ivf(vectors);

        // 2. Assign vectors to centroids.
        self.assign_to_centroids(vectors);

        // 3. Train PQ codebooks on residuals.
        self.train_pq(vectors);

        // 4. Encode all vectors (residuals) with PQ.
        self.encode_vectors(vectors);

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }

        if k == 0 || self.vectors.is_empty() || self.centroids.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Find nprobe nearest IVF centroids.
        let nprobe = self.nprobe.min(self.nlist);
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, self.metric)))
            .collect();
        centroid_dists
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. For each probed centroid, compute query residual and ADC.
        let mut candidates: Vec<SearchResult> = Vec::new();
        for &(centroid_id, _) in centroid_dists.iter().take(nprobe) {
            // Query residual for this centroid.
            let query_residual = Self::compute_residual(query, &self.centroids[centroid_id]);
            let table = self.precompute_distance_table(&query_residual);

            for &vec_id in &self.inverted_lists[centroid_id] {
                if self.deleted.contains(&vec_id) {
                    continue;
                }
                let dist = Self::adc_distance(&table, &self.pq_codes[vec_id as usize]);
                candidates.push(SearchResult {
                    id: vec_id,
                    distance: dist,
                });
            }
        }

        // 3. Sort by approximate distance and take top candidates for reranking.
        candidates
            .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Rerank top candidates with exact distances for better accuracy.
        let rerank_count = (k * 10).min(candidates.len());
        let mut reranked: Vec<SearchResult> = candidates[..rerank_count]
            .iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: compute_distance(query, &self.vectors[c.id as usize], self.metric),
            })
            .collect();
        reranked
            .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);

        Ok(reranked)
    }

    /// Phase F: native filtered search. Skips non-bitmap doc_ids
    /// INSIDE the inverted-list walk, then reranks only filtered
    /// candidates. Avoids the default's 10x oversample on the
    /// full result set, which is wasteful when the filter is
    /// small relative to the index size (e.g. MVCC visibility
    /// dropping superseded versions).
    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &roaring::RoaringBitmap,
    ) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }
        if k == 0 || self.vectors.is_empty() || self.centroids.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Find nprobe nearest IVF centroids (same as search).
        let nprobe = self.nprobe.min(self.nlist);
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, self.metric)))
            .collect();
        centroid_dists
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. ADC distances, filtering by bitmap inside the walk.
        let mut candidates: Vec<SearchResult> = Vec::new();
        for &(centroid_id, _) in centroid_dists.iter().take(nprobe) {
            let query_residual = Self::compute_residual(query, &self.centroids[centroid_id]);
            let table = self.precompute_distance_table(&query_residual);
            for &vec_id in &self.inverted_lists[centroid_id] {
                if self.deleted.contains(&vec_id) || !filter.contains(vec_id) {
                    continue;
                }
                let dist = Self::adc_distance(&table, &self.pq_codes[vec_id as usize]);
                candidates.push(SearchResult { id: vec_id, distance: dist });
            }
        }

        // 3. Sort + rerank top candidates with exact distances.
        candidates.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        let rerank_count = (k * 4).min(candidates.len()).max(k);
        let rerank_count = rerank_count.min(candidates.len());
        let mut reranked: Vec<SearchResult> = candidates[..rerank_count]
            .iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: compute_distance(query, &self.vectors[c.id as usize], self.metric),
            })
            .collect();
        reranked.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(k);
        Ok(reranked)
    }

    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()> {
        if vector.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len() as u32,
            });
        }

        // Grow arrays if needed.
        if id as usize >= self.vectors.len() {
            self.vectors
                .resize(id as usize + 1, vec![0.0; self.dimension as usize]);
            self.pq_codes
                .resize(id as usize + 1, vec![0u8; self.m_pq]);
            self.vector_centroid.resize(id as usize + 1, 0);
        }
        self.vectors[id as usize] = vector.to_vec();
        self.deleted.remove(&id);

        // Assign to nearest centroid if centroids exist.
        if !self.centroids.is_empty() {
            let centroid_id = nearest_centroid(vector, &self.centroids, self.metric);
            // Remove from any existing inverted list.
            for list in &mut self.inverted_lists {
                list.retain(|&v| v != id);
            }
            self.inverted_lists[centroid_id].push(id);
            self.vector_centroid[id as usize] = centroid_id as u32;

            // Encode residual with PQ.
            if !self.pq_centroids.is_empty() {
                let sub_dim = self.sub_dim();
                let residual = Self::compute_residual(vector, &self.centroids[centroid_id]);
                let codes = self.encode_residual(&residual, sub_dim);
                self.pq_codes[id as usize] = codes;
            }
        }

        Ok(())
    }

    fn delete(&mut self, id: u32) -> Result<()> {
        if (id as usize) >= self.vectors.len() {
            return Err(VedaError::Index(format!(
                "Cannot delete vector with id {id}: id out of range"
            )));
        }
        self.deleted.insert(id);
        Ok(())
    }

    fn len(&self) -> usize {
        if self.vectors.is_empty() {
            return 0;
        }
        let total: usize = self.inverted_lists.iter().map(|l| l.len()).sum();
        total.saturating_sub(self.deleted.len())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let data = IvfPqIndexData {
            centroids: self.centroids.clone(),
            inverted_lists: self.inverted_lists.clone(),
            vector_centroid: self.vector_centroid.clone(),
            pq_centroids: self.pq_centroids.clone(),
            pq_codes: self.pq_codes.clone(),
            nlist: self.nlist,
            nprobe: self.nprobe,
            m_pq: self.m_pq,
            nbits: self.nbits,
            vectors: self.vectors.clone(),
            metric: self.metric,
            dimension: self.dimension,
            deleted: self.deleted.iter().copied().collect(),
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>
    where
        Self: Sized,
    {
        let bytes = fs::read(path)?;
        let data: IvfPqIndexData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("bincode deserialize failed: {e}")))?;

        if data.metric != metric {
            return Err(VedaError::Index(format!(
                "Metric mismatch: file has {:?}, expected {:?}",
                data.metric, metric
            )));
        }
        if data.dimension != dimension {
            return Err(VedaError::DimensionMismatch {
                expected: dimension,
                actual: data.dimension,
            });
        }

        Ok(Self {
            centroids: data.centroids,
            inverted_lists: data.inverted_lists,
            vector_centroid: data.vector_centroid,
            pq_centroids: data.pq_centroids,
            pq_codes: data.pq_codes,
            nlist: data.nlist,
            nprobe: data.nprobe,
            m_pq: data.m_pq,
            nbits: data.nbits,
            vectors: data.vectors,
            metric: data.metric,
            dimension: data.dimension,
            deleted: data.deleted.into_iter().collect(),
        })
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}
