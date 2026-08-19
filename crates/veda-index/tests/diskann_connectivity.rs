//! The Vamana graph must stay connected across clusters.
//!
//! Greedy search only ever reaches nodes on a directed path from the medoid,
//! so a graph whose clusters are not linked to each other caps recall at the
//! medoid cluster's share of the corpus no matter how large `l_search` is.
//! That is a build-time property, so it is asserted on the graph directly
//! rather than inferred from a recall number.

use std::collections::VecDeque;

use veda_core::types::DistanceMetric;
use veda_index::diskann::builder::build_vamana;

const N: usize = 2_000;
const CLUSTERS: usize = 5;
const SIGMA: f32 = 0.10;
const CENTROID_SEED: u64 = 0xC0FF_EE01;
const CORPUS_SEED: u64 = 42;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f32_signed(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((self.state >> 32) as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
}

fn random_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32_signed()).collect())
        .collect()
}

/// Same generator as `Corpus::Clustered` in `tests/ann_benchmark.rs`, so a
/// failure here explains a recall number measured there. Node `i` belongs to
/// cluster `i % CLUSTERS`.
fn clustered_corpus(dim: usize) -> Vec<Vec<f32>> {
    let centroids = random_dataset(CLUSTERS, dim, CENTROID_SEED);
    let noise = random_dataset(N, dim, CORPUS_SEED);
    (0..N)
        .map(|i| {
            let c = &centroids[i % CLUSTERS];
            (0..dim).map(|d| c[d] + noise[i][d] * SIGMA).collect()
        })
        .collect()
}

/// Nodes reachable from the medoid by following directed edges.
fn reachable_from_medoid(graph: &veda_index::diskann::graph::VamanaGraph) -> Vec<bool> {
    let medoid = graph.medoid().expect("a built graph has a medoid");
    let mut seen = vec![false; graph.len()];
    let mut queue = VecDeque::new();
    seen[medoid as usize] = true;
    queue.push_back(medoid);
    while let Some(node) = queue.pop_front() {
        for &next in graph.get_neighbors(node) {
            if !seen[next as usize] {
                seen[next as usize] = true;
                queue.push_back(next);
            }
        }
    }
    seen
}

fn assert_every_cluster_reachable(dim: usize) {
    let vectors = clustered_corpus(dim);
    let graph = build_vamana(&vectors, DistanceMetric::Cosine, 64, 100, 1.2);

    let seen = reachable_from_medoid(&graph);

    let mut per_cluster = vec![0usize; CLUSTERS];
    for (i, &s) in seen.iter().enumerate() {
        if s {
            per_cluster[i % CLUSTERS] += 1;
        }
    }
    let reached = seen.iter().filter(|&&s| s).count();

    // Pruning legitimately leaves a few nodes with no inbound edge, so this is
    // not "every node". What it rules out is whole clusters being cut off.
    //
    // The corpus is kept small enough to stay cheap in a debug CI build.
    // Cluster size (400) only has to exceed r (64) for either defect to
    // appear: that is what makes a node's 64 nearest candidates all
    // intra-cluster.
    assert!(
        reached * 100 >= N * 95,
        "dim={dim}: only {reached}/{N} nodes reachable from the medoid;          per cluster: {per_cluster:?}"
    );
    let starved: Vec<usize> = (0..CLUSTERS)
        .filter(|&c| per_cluster[c] * 2 < N / CLUSTERS)
        .collect();
    assert!(
        starved.is_empty(),
        "dim={dim}: clusters {starved:?} are mostly unreachable from the          medoid; per cluster: {per_cluster:?}"
    );
}

#[test]
fn every_cluster_is_reachable_from_the_medoid_at_128d() {
    assert_every_cluster_reachable(128);
}

/// The same assertion at a dimension real embeddings actually use.
///
/// Pruning over the visited set fixed 128d and left this case broken, because
/// the two failures have different causes: the first was a candidate set with
/// no long-range members, the second is distance concentration stopping the
/// pruning rule from ever firing, so the degree budget fills with nearest
/// neighbours before any long-range candidate is reached. A 128d-only guard
/// passed throughout. Embeddings are 768-1024d, so that is the case that has
/// to be covered.
#[test]
fn every_cluster_is_reachable_from_the_medoid_at_768d() {
    assert_every_cluster_reachable(768);
}
