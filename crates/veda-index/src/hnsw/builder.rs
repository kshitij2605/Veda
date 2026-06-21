use std::collections::HashSet;

use rand::Rng;

use crate::hnsw::distance_provider::{DistanceProvider, RawVectorsProvider};
use crate::hnsw::graph::HnswGraph;
use crate::hnsw::searcher::HnswSearcher;
use veda_core::types::DistanceMetric;

/// HNSW index builder — handles insertion of vectors into the graph.
pub struct HnswBuilder;

impl HnswBuilder {
    /// Select a random level for a new node using the exponential distribution
    /// from the HNSW paper: floor(-ln(uniform(0,1)) * mL), clamped to max_level_cap.
    pub fn random_level(ml: f64, max_level_cap: usize) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        // Avoid log(0) which is -infinity.
        let r = r.max(1e-15);
        let level = (-r.ln() * ml).floor() as usize;
        level.min(max_level_cap)
    }

    /// Build the distance provider over the corpus, selecting the
    /// normalized-dot fast path for Cosine indexes (matches the search path).
    fn provider(
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
        normalized: bool,
    ) -> RawVectorsProvider<'_> {
        if normalized {
            RawVectorsProvider::new_normalized_cosine(vectors)
        } else {
            RawVectorsProvider::new(vectors, metric)
        }
    }

    /// Distance between two stored vectors by id, via the same metric /
    /// normalized fast path the search uses.
    fn dist_ids(dp: &dyn DistanceProvider, vectors: &[Vec<f32>], a: u32, b: u32) -> f32 {
        dp.distance(&vectors[a as usize], b)
    }

    /// SELECT-NEIGHBORS-HEURISTIC (Malkov & Yashunin, Algorithm 4).
    ///
    /// Given a `base` node and `candidates` (id, distance-to-base) — which
    /// need NOT be pre-sorted — greedily pick up to `m` diversified
    /// neighbours: a candidate `c` is accepted only if it is closer to
    /// `base` than to every already-selected neighbour. This prunes
    /// redundant edges that all point into the same direction, yielding a
    /// graph with better navigability (higher recall) and shorter search
    /// paths (fewer distance comps / hops) than plain nearest-`m`.
    fn select_neighbors_heuristic(
        base: u32,
        candidates: &[(u32, f32)],
        m: usize,
        dp: &dyn DistanceProvider,
        vectors: &[Vec<f32>],
    ) -> Vec<u32> {
        // Work over candidates sorted nearest-first to the base.
        let mut sorted: Vec<(u32, f32)> = candidates
            .iter()
            .copied()
            .filter(|(c, _)| *c != base)
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<u32> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for (cand, dist_to_base) in &sorted {
            if selected.len() >= m {
                break;
            }
            // Accept `cand` only if it is closer to `base` than to any
            // already-selected neighbour (the diversification test).
            let mut good = true;
            for &s in &selected {
                let dist_to_sel = Self::dist_ids(dp, vectors, *cand, s);
                if dist_to_sel < *dist_to_base {
                    good = false;
                    break;
                }
            }
            if good {
                selected.push(*cand);
            } else {
                pruned.push(*cand);
            }
        }

        // keepPrunedConnections: if the diversity filter left us short of `m`,
        // backfill with the closest pruned candidates (they were already
        // distance-sorted). Keeps node degree near `m` so navigability /
        // recall don't collapse on dense clusters where many candidates fail
        // the diversity test.
        if selected.len() < m {
            for cand in pruned {
                if selected.len() >= m {
                    break;
                }
                selected.push(cand);
            }
        }
        selected
    }

    /// Insert a single vector into the HNSW graph.
    ///
    /// Implements Algorithm 1 (INSERT) from the Malkov & Yashunin paper,
    /// using Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) for both the initial
    /// neighbour selection and the back-link pruning.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        id: u32,
        vector: &[f32],
        graph: &mut HnswGraph,
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
        normalized: bool,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        ml: f64,
    ) {
        let new_level = Self::random_level(ml, 16);

        // Add the node to the graph.
        graph.add_node(id, new_level);

        // If this is the first node, just set it as entry point.
        let ep = match graph.entry_point() {
            Some(ep) => ep,
            None => {
                graph.set_entry_point(id);
                graph.set_max_level(new_level);
                return;
            }
        };

        let current_max_level = graph.max_level();
        let mut ep_id = ep;
        let dp = Self::provider(vectors, metric, normalized);
        let mut ep_dist = dp.distance(vector, ep_id);

        // Phase 1: Greedily traverse from the top level down to new_level + 1.
        // At each of these upper layers, we just find the single nearest neighbor.
        let top = current_max_level;
        if top > new_level {
            for layer in (new_level + 1..=top).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(vector, ep_id, layer, graph, &dp);
                ep_id = nearest_id;
                ep_dist = nearest_dist;
            }
        }

        // Phase 2: For layers min(new_level, top) down to 0, do a full search
        // and connect the new node to the best neighbors.
        let insert_top = new_level.min(top);
        let deleted = HashSet::new(); // No filtering during construction.

        for layer in (0..=insert_top).rev() {
            let entry_points = vec![(ep_id, ep_dist)];
            let candidates = HnswSearcher::search_layer(
                vector,
                &entry_points,
                ef_construction,
                layer,
                graph,
                &dp,
                &deleted,
            );

            let max_connections = if layer == 0 { m_max0 } else { m };

            // Algorithm 4: diversified neighbour selection over the candidate set.
            let neighbors =
                Self::select_neighbors_heuristic(id, &candidates, max_connections, &dp, vectors);

            // Set bidirectional connections.
            graph.set_neighbors(id, layer, neighbors.clone());

            for &neighbor_id in &neighbors {
                // Add the new node as a neighbor of each selected neighbor.
                let mut n_neighbors: Vec<u32> = graph.get_neighbors(neighbor_id, layer).to_vec();
                if !n_neighbors.contains(&id) {
                    n_neighbors.push(id);
                }

                // If the neighbor now has too many connections, prune with the
                // same diversifying heuristic (Algorithm 4) rather than a plain
                // nearest-`max_conn` truncation.
                let max_conn = if layer == 0 { m_max0 } else { m };
                if n_neighbors.len() > max_conn {
                    let scored: Vec<(u32, f32)> = n_neighbors
                        .iter()
                        .map(|&nid| (nid, Self::dist_ids(&dp, vectors, neighbor_id, nid)))
                        .collect();
                    n_neighbors = Self::select_neighbors_heuristic(
                        neighbor_id,
                        &scored,
                        max_conn,
                        &dp,
                        vectors,
                    );
                }

                graph.set_neighbors(neighbor_id, layer, n_neighbors);
            }

            // Update entry point for the next layer down.
            if !candidates.is_empty() {
                ep_id = candidates[0].0;
                ep_dist = candidates[0].1;
            }
        }

        // If the new node's level is higher than the current max, update entry point.
        if new_level > current_max_level {
            graph.set_entry_point(id);
            graph.set_max_level(new_level);
        }
    }
}
