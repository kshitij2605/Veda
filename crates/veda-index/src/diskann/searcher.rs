use std::cmp::Ordering;
use std::collections::HashSet;

use crate::diskann::graph::VamanaGraph;
use crate::distance::compute_distance;
use veda_core::types::DistanceMetric;

/// Greedy/beam search on the Vamana graph.
///
/// Starts at the given entry point, maintains a sorted candidate list of size
/// `l_search`, and iteratively explores unvisited candidates' neighbors.
///
/// Returns up to `l_search` candidates as (id, distance) sorted by distance ascending.
pub fn greedy_search(
    query: &[f32],
    entry: u32,
    l_search: usize,
    graph: &VamanaGraph,
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<(u32, f32)> {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut expanded: HashSet<u32> = HashSet::new();

    // candidates: sorted list of (id, distance), ascending by distance.
    let entry_dist = compute_distance(query, &vectors[entry as usize], metric);
    let mut candidates: Vec<(u32, f32)> = vec![(entry, entry_dist)];
    visited.insert(entry);

    loop {
        // Find the best (closest) candidate not yet expanded.
        let next = candidates.iter().find(|&&(id, _)| !expanded.contains(&id));
        let (c_id, _) = match next {
            Some(&c) => c,
            None => break,
        };
        expanded.insert(c_id);

        // Expand neighbors.
        for &neighbor in graph.get_neighbors(c_id) {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            let n_dist = compute_distance(query, &vectors[neighbor as usize], metric);

            // Insert if candidates not full, or if better than the worst.
            if candidates.len() < l_search {
                let pos = candidates
                    .binary_search_by(|&(_, d)| d.partial_cmp(&n_dist).unwrap_or(Ordering::Equal))
                    .unwrap_or_else(|pos| pos);
                candidates.insert(pos, (neighbor, n_dist));
            } else if n_dist < candidates.last().unwrap().1 {
                candidates.pop();
                let pos = candidates
                    .binary_search_by(|&(_, d)| d.partial_cmp(&n_dist).unwrap_or(Ordering::Equal))
                    .unwrap_or_else(|pos| pos);
                candidates.insert(pos, (neighbor, n_dist));
            }
        }
    }

    candidates
}
