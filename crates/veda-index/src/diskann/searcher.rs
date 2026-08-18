use std::cmp::Ordering;
use std::collections::HashSet;

use crate::diskann::graph::VamanaGraph;
use crate::distance::compute_distance;
use veda_core::types::DistanceMetric;

/// A node and its distance to the query.
pub type Scored = (u32, f32);

/// The `l_search` closest candidates, and every node the walk visited.
/// Both are sorted by distance ascending.
pub type SearchAndVisited = (Vec<Scored>, Vec<Scored>);

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
    greedy_search_with_visited(query, entry, l_search, graph, vectors, metric).0
}

/// Greedy search that also returns every node whose distance it computed.
///
/// Index construction prunes over this visited set rather than over the
/// `l_search` closest candidates (Algorithm 3 of the DiskANN paper). The
/// distinction decides whether the graph stays connected: the closest
/// candidates for a node are all its immediate neighbours, so pruning over
/// them yields only short edges. The visited set additionally holds the
/// nodes the walk passed through on its way in from the medoid, which is
/// where every long-range edge comes from. Without them a corpus of well
/// separated clusters builds one disconnected component per cluster, and
/// search can never leave the medoid's own cluster.
///
/// Both returned lists are sorted by distance ascending, as `robust_prune`
/// requires.
pub fn greedy_search_with_visited(
    query: &[f32],
    entry: u32,
    l_search: usize,
    graph: &VamanaGraph,
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> SearchAndVisited {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut visited_scored: Vec<(u32, f32)> = Vec::new();
    let mut expanded: HashSet<u32> = HashSet::new();

    // candidates: sorted list of (id, distance), ascending by distance.
    let entry_dist = compute_distance(query, &vectors[entry as usize], metric);
    let mut candidates: Vec<(u32, f32)> = vec![(entry, entry_dist)];
    visited.insert(entry);
    visited_scored.push((entry, entry_dist));

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
            visited_scored.push((neighbor, n_dist));

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

    visited_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    (candidates, visited_scored)
}
