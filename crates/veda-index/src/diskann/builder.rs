use rand::Rng;
use rand::seq::SliceRandom;

use crate::diskann::graph::VamanaGraph;
use crate::diskann::searcher::greedy_search_with_visited;
use crate::distance::compute_distance;
use veda_core::types::DistanceMetric;

/// Slots of each node's degree budget reserved for long-range edges.
///
/// `robust_prune` discards a candidate `v` only when `alpha * d(v, p*) <= d(v, p)`
/// — that is, only when the already-selected `p*` is substantially closer to `v`
/// than `p` is. In high dimension distances concentrate, so that condition stops
/// firing. Measured on a 10k clustered corpus, every intra-cluster distance falls
/// in [0.0084, 0.0120] at 768d — a spread of 1.4x, against 2.3x at 128d — which
/// leaves `d(v, p*) ~= d(v, p)` and keeps essentially every candidate. The degree
/// budget then fills with the `r` nearest neighbours, all inside the node's own
/// cluster, and no long edge survives: 257 cross-cluster edges out of 628,763,
/// with 999 of 10,000 nodes reachable from the medoid, and recall@10 of 10% where
/// the same code scores 100% at 128d.
///
/// Reserving a few slots for uniformly random targets restores navigability
/// without relying on the pruning rule to free them. A handful of random
/// out-edges per node makes the graph connected with high probability, and greedy
/// search needs only one edge into the query's neighbourhood to descend into it.
const LONG_EDGE_BUDGET: usize = 4;

/// Cap on the candidate pool `robust_prune` scans, matching the reference DiskANN
/// implementation's `maxc`.
///
/// Pruning normally shrinks the pool as selection proceeds. Under the distance
/// concentration described above it shrinks nothing, so each of the `r` selection
/// rounds rescans the entire visited set. Capping the pool bounds that work. The
/// pool arrives sorted by distance, so truncating keeps the closest candidates,
/// the ones that make good local edges; long-range connectivity comes from the
/// reserved slots instead.
///
/// This is a bound, not a cure: the 10k x 768d build still takes ~435s against
/// 14s for HNSW, because the reverse-edge pass — not this prune — dominates.
/// Widening the degree budget so that pass fires less often was tried and
/// measured *worse* (630s), since longer lists inflate every visited set in
/// turn. The build cost at high dimension remains an open problem.
const MAX_CANDIDATES: usize = 750;

/// Write `local` as `id`'s neighbours with the reserved long-range edges appended.
///
/// Used for every write during the build so that neither the main pass nor the
/// reverse-edge prune can drop them: both would, since the reserved targets are
/// by construction the farthest entries in any list they appear in.
fn set_with_long_edges(
    graph: &mut VamanaGraph,
    id: u32,
    mut local: Vec<u32>,
    long_edges: &[Vec<u32>],
) {
    for &target in &long_edges[id as usize] {
        if target != id && !local.contains(&target) {
            local.push(target);
        }
    }
    graph.set_neighbors(id, local);
}

/// Compute the medoid: the vector closest to the centroid of all vectors.
pub fn compute_medoid(vectors: &[Vec<f32>], metric: DistanceMetric) -> u32 {
    let n = vectors.len();
    let dim = vectors[0].len();

    // Compute centroid.
    let mut centroid = vec![0.0f32; dim];
    for v in vectors {
        for (c, &val) in centroid.iter_mut().zip(v.iter()) {
            *c += val;
        }
    }
    let inv_n = 1.0 / n as f32;
    for c in &mut centroid {
        *c *= inv_n;
    }

    // Find vector closest to centroid.
    let mut best_id = 0u32;
    let mut best_dist = f32::MAX;
    for (i, v) in vectors.iter().enumerate() {
        let d = compute_distance(&centroid, v, metric);
        if d < best_dist {
            best_dist = d;
            best_id = i as u32;
        }
    }
    best_id
}

/// RobustPrune: select up to R neighbors for node p from candidate set V.
///
/// The alpha parameter controls diversity: higher alpha prunes more aggressively,
/// favoring a diverse set of neighbors over the strictly closest ones.
pub fn robust_prune(
    p: u32,
    candidates: &[(u32, f32)], // (id, distance_to_p), already sorted by distance
    alpha: f32,
    r: usize,
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<u32> {
    // Work with a mutable list of (id, dist_to_p), already sorted.
    let mut remaining: Vec<(u32, f32)> = candidates
        .iter()
        .filter(|&&(id, _)| id != p)
        .copied()
        .collect();

    let mut result: Vec<u32> = Vec::with_capacity(r);

    while !remaining.is_empty() && result.len() < r {
        // Pick the closest to p.
        let (p_star, _dist_p_star) = remaining[0];
        result.push(p_star);
        remaining.remove(0);

        // Remove from remaining all v where dist(v, p*) * alpha <= dist(v, p).
        remaining.retain(|&(v_id, dist_v_p)| {
            let dist_v_pstar =
                compute_distance(&vectors[v_id as usize], &vectors[p_star as usize], metric);
            dist_v_pstar * alpha > dist_v_p
        });
    }

    result
}

/// Build the Vamana graph from a set of vectors.
pub fn build_vamana(
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
    r: usize,
    l_build: usize,
    alpha: f32,
) -> VamanaGraph {
    let n = vectors.len();
    let mut graph = VamanaGraph::new(n);

    // Step 1: Compute medoid.
    let medoid = compute_medoid(vectors, metric);
    graph.set_medoid(medoid);

    // Step 2: Initialize with random neighbors.
    let mut rng = rand::thread_rng();
    for i in 0..n {
        let mut neighbors: Vec<u32> = Vec::with_capacity(r);
        while neighbors.len() < r.min(n - 1) {
            let j = rng.gen_range(0..n) as u32;
            if j != i as u32 && !neighbors.contains(&j) {
                neighbors.push(j);
            }
        }
        graph.set_neighbors(i as u32, neighbors);
    }

    // Reserved long-range targets, drawn once and re-attached on every write.
    let long_budget = LONG_EDGE_BUDGET.min(n - 1);
    let long_edges: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut picks: Vec<u32> = Vec::with_capacity(long_budget);
            while picks.len() < long_budget {
                let j = rng.gen_range(0..n) as u32;
                if j != i as u32 && !picks.contains(&j) {
                    picks.push(j);
                }
            }
            picks
        })
        .collect();

    // The pruned local neighbours share the budget with the reserved slots.
    let local_r = r.saturating_sub(long_budget).max(1);

    // Step 3: Iterate over all vectors in random order.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(&mut rng);

    for &p in &order {
        // Greedy search from the medoid, keeping the whole visited set:
        // pruning over just the L closest would give p only short edges, and
        // on a clustered corpus that builds one disconnected component per
        // cluster. See `greedy_search_with_visited`.
        let (_, visited) = greedy_search_with_visited(
            &vectors[p as usize],
            medoid,
            l_build,
            &graph,
            vectors,
            metric,
        );

        // RobustPrune to select neighbors for p.
        let pool = &visited[..visited.len().min(MAX_CANDIDATES)];
        let new_neighbors = robust_prune(p, pool, alpha, local_r, vectors, metric);
        set_with_long_edges(&mut graph, p, new_neighbors.clone(), &long_edges);

        // Add reverse edges and prune over-full neighbors.
        for &neighbor in &new_neighbors {
            let mut n_neighbors: Vec<u32> = graph.get_neighbors(neighbor).to_vec();
            if !n_neighbors.contains(&p) {
                n_neighbors.push(p);
            }

            if n_neighbors.len() > r {
                // Build candidate set with distances to neighbor.
                let mut scored: Vec<(u32, f32)> = n_neighbors
                    .iter()
                    .map(|&nid| {
                        let d = compute_distance(
                            &vectors[neighbor as usize],
                            &vectors[nid as usize],
                            metric,
                        );
                        (nid, d)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let pruned = robust_prune(neighbor, &scored, alpha, local_r, vectors, metric);
                set_with_long_edges(&mut graph, neighbor, pruned, &long_edges);
            } else {
                set_with_long_edges(&mut graph, neighbor, n_neighbors, &long_edges);
            }
        }
    }

    graph
}
