use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::distance::compute_distance;
use crate::hnsw::graph::HnswGraph;
use veda_core::types::DistanceMetric;

/// A candidate for the min-heap (nearest candidates to process).
#[derive(Debug, Clone)]
struct MinCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior: smaller distance = higher priority.
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// A candidate for the max-heap (farthest result to potentially evict).
#[derive(Debug, Clone)]
struct MaxCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW search utilities.
pub struct HnswSearcher;

impl HnswSearcher {
    /// Search a single layer of the HNSW graph starting from the given entry points.
    ///
    /// This implements Algorithm 2 from the paper: SEARCH-LAYER(q, ep, ef, lc).
    ///
    /// Returns a list of (id, distance) pairs representing the `ef` nearest neighbors
    /// found at this layer, sorted by distance ascending.
    pub fn search_layer(
        query: &[f32],
        entry_points: &[(u32, f32)],
        ef: usize,
        layer: usize,
        graph: &HnswGraph,
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
        deleted: &HashSet<u32>,
    ) -> Vec<(u32, f32)> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();

        for &(ep_id, ep_dist) in entry_points {
            visited.insert(ep_id);
            candidates.push(MinCandidate {
                id: ep_id,
                distance: ep_dist,
            });
            results.push(MaxCandidate {
                id: ep_id,
                distance: ep_dist,
            });
        }

        while let Some(MinCandidate { id: c_id, distance: c_dist }) = candidates.pop() {
            // If the nearest candidate is farther than the farthest in results, stop.
            let farthest_dist = results.peek().map(|r| r.distance).unwrap_or(f32::MAX);
            if c_dist > farthest_dist {
                break;
            }

            // Explore neighbors of c at this layer.
            let neighbors = graph.get_neighbors(c_id, layer);
            for &neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let n_dist = compute_distance(query, &vectors[neighbor_id as usize], metric);

                let farthest_dist = results.peek().map(|r| r.distance).unwrap_or(f32::MAX);

                if results.len() < ef || n_dist < farthest_dist {
                    candidates.push(MinCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                    results.push(MaxCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Collect results, filter deleted, sort by distance ascending.
        let mut result_vec: Vec<(u32, f32)> = results
            .into_iter()
            .filter(|c| !deleted.contains(&c.id))
            .map(|c| (c.id, c.distance))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Greedily traverse a layer to find the single nearest neighbor to the query.
    ///
    /// Used for layers above the insertion level during insert and search
    /// (equivalent to search_layer with ef=1, but simpler).
    pub fn greedy_search(
        query: &[f32],
        entry_point: u32,
        layer: usize,
        graph: &HnswGraph,
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> (u32, f32) {
        let mut current_id = entry_point;
        let mut current_dist = compute_distance(query, &vectors[current_id as usize], metric);

        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(current_id, layer);
            for &neighbor_id in neighbors {
                let n_dist = compute_distance(query, &vectors[neighbor_id as usize], metric);
                if n_dist < current_dist {
                    current_id = neighbor_id;
                    current_dist = n_dist;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        (current_id, current_dist)
    }
}
