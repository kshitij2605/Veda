use rand::seq::SliceRandom;
use rand::Rng;

use crate::distance::compute_distance;
use crate::diskann::graph::VamanaGraph;
use crate::diskann::searcher::greedy_search;
use veda_core::types::DistanceMetric;

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
            let dist_v_pstar = compute_distance(
                &vectors[v_id as usize],
                &vectors[p_star as usize],
                metric,
            );
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

    // Step 3: Iterate over all vectors in random order.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(&mut rng);

    for &p in &order {
        // Greedy search from medoid to find L nearest neighbors of p.
        let candidates = greedy_search(
            &vectors[p as usize],
            medoid,
            l_build,
            &graph,
            vectors,
            metric,
        );

        // RobustPrune to select neighbors for p.
        let new_neighbors = robust_prune(p, &candidates, alpha, r, vectors, metric);
        graph.set_neighbors(p, new_neighbors.clone());

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

                let pruned = robust_prune(neighbor, &scored, alpha, r, vectors, metric);
                graph.set_neighbors(neighbor, pruned);
            } else {
                graph.set_neighbors(neighbor, n_neighbors);
            }
        }
    }

    graph
}
