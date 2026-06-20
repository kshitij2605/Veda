use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

use crate::distance::compute_distance;
use crate::traits::{SearchResult, VectorIndex};

/// A candidate entry for the max-heap used during search.
/// We use a max-heap so we can efficiently evict the *worst* (largest distance)
/// candidate when the heap exceeds `k`.
struct HeapCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance: larger distance = higher priority (will be evicted first).
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Brute-force exact nearest neighbor search.
///
/// Computes the distance from the query to every non-deleted vector and returns
/// the top-k closest results. This is simple and always correct, making it
/// ideal for small datasets and as ground-truth for testing approximate indexes.
pub struct FlatIndex {
    vectors: Vec<Vec<f32>>,
    deleted: HashSet<u32>,
    metric: DistanceMetric,
    dimension: u32,
}

/// Serializable form of `FlatIndex` for save/load round-trips.
#[derive(Serialize, Deserialize)]
struct FlatIndexData {
    vectors: Vec<Vec<f32>>,
    deleted: Vec<u32>,
    metric: DistanceMetric,
    dimension: u32,
}

impl FlatIndex {
    /// Create a new empty flat index.
    pub fn new(metric: DistanceMetric, dimension: u32) -> Self {
        Self {
            vectors: Vec::new(),
            deleted: HashSet::new(),
            metric,
            dimension,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        self.vectors.clear();
        self.deleted.clear();

        for (i, v) in vectors.iter().enumerate() {
            if v.len() as u32 != self.dimension {
                return Err(VedaError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len() as u32,
                });
            }
            let _ = i; // id == position
            self.vectors.push(v.to_vec());
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len() as u32,
            });
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        let mut heap: BinaryHeap<HeapCandidate> = BinaryHeap::with_capacity(k + 1);

        for (id, vector) in self.vectors.iter().enumerate() {
            let id = id as u32;
            if self.deleted.contains(&id) {
                continue;
            }

            let distance = compute_distance(query, vector, self.metric);

            if heap.len() < k {
                heap.push(HeapCandidate { id, distance });
            } else if let Some(worst) = heap.peek()
                && distance < worst.distance
            {
                heap.pop();
                heap.push(HeapCandidate { id, distance });
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        Ok(results)
    }

    /// Phase F: native filtered search. Flat is already brute-
    /// force, so the only optimization is skipping non-bitmap
    /// doc_ids INSIDE the scan instead of scoring everything
    /// then post-filtering. Eliminates the 10x oversample the
    /// default would impose.
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
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut heap: BinaryHeap<HeapCandidate> = BinaryHeap::with_capacity(k + 1);
        for (id, vector) in self.vectors.iter().enumerate() {
            let id = id as u32;
            if self.deleted.contains(&id) {
                continue;
            }
            if !filter.contains(id) {
                continue;
            }
            let distance = compute_distance(query, vector, self.metric);
            if heap.len() < k {
                heap.push(HeapCandidate { id, distance });
            } else if let Some(worst) = heap.peek()
                && distance < worst.distance
            {
                heap.pop();
                heap.push(HeapCandidate { id, distance });
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        Ok(results)
    }

    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()> {
        if vector.len() as u32 != self.dimension {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len() as u32,
            });
        }

        // Grow the vectors array if the id is beyond current capacity.
        if id as usize >= self.vectors.len() {
            self.vectors
                .resize(id as usize + 1, vec![0.0; self.dimension as usize]);
        }
        self.vectors[id as usize] = vector.to_vec();
        // Ensure the id is not marked as deleted.
        self.deleted.remove(&id);
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
        self.vectors.len() - self.deleted.len()
    }

    fn save(&self, path: &Path) -> Result<()> {
        let data = FlatIndexData {
            vectors: self.vectors.clone(),
            deleted: self.deleted.iter().copied().collect(),
            metric: self.metric,
            dimension: self.dimension,
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("bincode serialize failed: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: FlatIndexData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("bincode deserialize failed: {e}")))?;

        // Validate that the loaded data matches the expected parameters.
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
            vectors: data.vectors,
            deleted: data.deleted.into_iter().collect(),
            metric: data.metric,
            dimension: data.dimension,
        })
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
    fn test_new_index_is_empty() {
        let idx = FlatIndex::new(DistanceMetric::Cosine, 4);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_insert_and_search() {
        let mut idx = FlatIndex::new(DistanceMetric::Euclidean, 3);
        idx.insert(0, &[1.0, 0.0, 0.0]).unwrap();
        idx.insert(1, &[0.0, 1.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!((results[0].distance - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delete_removes_from_results() {
        let mut idx = FlatIndex::new(DistanceMetric::Euclidean, 2);
        idx.insert(0, &[0.0, 0.0]).unwrap();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[2.0, 0.0]).unwrap();

        idx.delete(1).unwrap();

        let results = idx.search(&[0.5, 0.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != 1));
    }

    #[test]
    fn search_filtered_returns_only_bitmap_members() {
        use roaring::RoaringBitmap;
        let mut idx = FlatIndex::new(DistanceMetric::Euclidean, 2);
        idx.insert(0, &[0.0, 0.0]).unwrap();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[2.0, 0.0]).unwrap();
        idx.insert(3, &[3.0, 0.0]).unwrap();
        idx.insert(4, &[4.0, 0.0]).unwrap();

        let mut filter = RoaringBitmap::new();
        filter.insert(1);
        filter.insert(3);
        // Query at 0.5: closest visible is 1 (dist 0.5), then 3 (dist 2.5).
        let results = idx.search_filtered(&[0.5, 0.0], 5, &filter).unwrap();
        let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn search_filtered_excludes_deleted_even_if_in_filter() {
        use roaring::RoaringBitmap;
        let mut idx = FlatIndex::new(DistanceMetric::Euclidean, 2);
        idx.insert(0, &[0.0, 0.0]).unwrap();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.delete(1).unwrap();
        let mut filter = RoaringBitmap::new();
        filter.insert(0);
        filter.insert(1); // tombstoned — must be excluded
        let results = idx.search_filtered(&[0.5, 0.0], 5, &filter).unwrap();
        let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn search_filtered_with_empty_filter_returns_empty() {
        use roaring::RoaringBitmap;
        let mut idx = FlatIndex::new(DistanceMetric::Euclidean, 2);
        idx.insert(0, &[0.0, 0.0]).unwrap();
        idx.insert(1, &[1.0, 0.0]).unwrap();
        let filter = RoaringBitmap::new();
        let results = idx.search_filtered(&[0.5, 0.0], 5, &filter).unwrap();
        assert!(results.is_empty());
    }
}
