use std::collections::HashSet;

use rand::Rng;

use crate::distance::compute_distance;
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

    /// Insert a single vector into the HNSW graph.
    ///
    /// Implements Algorithm 1 (INSERT) from the Malkov & Yashunin paper.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        id: u32,
        vector: &[f32],
        graph: &mut HnswGraph,
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
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
        let mut ep_dist = compute_distance(vector, &vectors[ep_id as usize], metric);

        // Phase 1: Greedily traverse from the top level down to new_level + 1.
        // At each of these upper layers, we just find the single nearest neighbor.
        let top = current_max_level;
        if top > new_level {
            for layer in (new_level + 1..=top).rev() {
                let (nearest_id, nearest_dist) =
                    HnswSearcher::greedy_search(vector, ep_id, layer, graph, vectors, metric);
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
                vectors,
                metric,
                &deleted,
            );

            // Select the M nearest neighbors from candidates.
            let max_connections = if layer == 0 { m_max0 } else { m };
            let neighbors: Vec<u32> = candidates
                .iter()
                .filter(|(cid, _)| *cid != id)
                .take(max_connections)
                .map(|(cid, _)| *cid)
                .collect();

            // Set bidirectional connections.
            graph.set_neighbors(id, layer, neighbors.clone());

            for &neighbor_id in &neighbors {
                // Add the new node as a neighbor of each selected neighbor.
                let mut n_neighbors: Vec<u32> =
                    graph.get_neighbors(neighbor_id, layer).to_vec();
                n_neighbors.push(id);

                // If the neighbor now has too many connections, prune.
                let max_conn = if layer == 0 { m_max0 } else { m };
                if n_neighbors.len() > max_conn {
                    // Prune: keep only the max_conn nearest to the neighbor.
                    let neighbor_vec = &vectors[neighbor_id as usize];
                    let mut scored: Vec<(u32, f32)> = n_neighbors
                        .iter()
                        .map(|&nid| {
                            let dist =
                                compute_distance(neighbor_vec, &vectors[nid as usize], metric);
                            (nid, dist)
                        })
                        .collect();
                    scored.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    scored.truncate(max_conn);
                    n_neighbors = scored.into_iter().map(|(nid, _)| nid).collect();
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
