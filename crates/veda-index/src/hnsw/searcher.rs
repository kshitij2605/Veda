use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use roaring::RoaringBitmap;

use crate::hnsw::distance_provider::DistanceProvider;
use crate::hnsw::graph::HnswGraph;
use crate::traits::SearchResult;

/// Generation-stamped dense visited set + reusable candidate/result
/// heaps, kept per-thread so `search_layer` allocates nothing on the hot
/// path. `stamp[id] == generation` means "visited this call"; bumping
/// `generation` resets the whole set in O(1) (no per-call zeroing).
struct SearchScratch {
    stamp: Vec<u32>,
    generation: u32,
    candidates: BinaryHeap<MinCandidate>,
    results: BinaryHeap<MaxCandidate>,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            stamp: Vec::new(),
            generation: 0,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    /// Begin a new traversal over `slots` nodes: grow + reset the visited
    /// stamp and clear the reusable heaps. Returns the active generation.
    fn begin(&mut self, slots: usize) -> u32 {
        if self.stamp.len() < slots {
            self.stamp.resize(slots, 0);
        }
        // Bump the generation; on wrap, zero the buffer once so a stale
        // stamp from ~4 billion calls ago can't read as visited.
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
        self.candidates.clear();
        self.results.clear();
        self.generation
    }

    #[inline]
    fn is_visited(&self, id: u32, cur_gen: u32) -> bool {
        self.stamp.get(id as usize).is_some_and(|&s| s == cur_gen)
    }

    #[inline]
    fn mark_visited(&mut self, id: u32, cur_gen: u32) {
        if let Some(slot) = self.stamp.get_mut(id as usize) {
            *slot = cur_gen;
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

/// Strategy for filtered search based on filter selectivity.
enum FilterStrategy {
    /// >50% selectivity: run normal search with 2x ef, then filter results.
    PreFilter,
    /// 1-50% selectivity: visit all neighbors during traversal but only add
    /// to results if the candidate passes the filter.
    InAlgorithm,
    /// <1% selectivity: heavy oversampling with ef * 20, then filter. Falls
    /// back to brute-force scan of filtered IDs if not enough results.
    PostFilter,
}

fn select_strategy(filter_count: u64, total_count: u64) -> FilterStrategy {
    if total_count == 0 {
        return FilterStrategy::PreFilter;
    }
    let selectivity = filter_count as f64 / total_count as f64;
    if selectivity > 0.5 {
        FilterStrategy::PreFilter
    } else if selectivity > 0.01 {
        FilterStrategy::InAlgorithm
    } else {
        FilterStrategy::PostFilter
    }
}

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
        dp: &dyn DistanceProvider,
        deleted: &HashSet<u32>,
    ) -> Vec<(u32, f32)> {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let cur_gen = s.begin(graph.slot_count());

            for &(ep_id, ep_dist) in entry_points {
                s.mark_visited(ep_id, cur_gen);
                s.candidates.push(MinCandidate {
                    id: ep_id,
                    distance: ep_dist,
                });
                s.results.push(MaxCandidate {
                    id: ep_id,
                    distance: ep_dist,
                });
            }

            while let Some(MinCandidate {
                id: c_id,
                distance: c_dist,
            }) = s.candidates.pop()
            {
                // If the nearest candidate is farther than the farthest in results, stop.
                let farthest_dist = s.results.peek().map(|r| r.distance).unwrap_or(f32::MAX);
                if c_dist > farthest_dist {
                    break;
                }

                // Explore neighbors of c at this layer.
                let neighbors = graph.get_neighbors(c_id, layer);
                for &neighbor_id in neighbors {
                    if s.is_visited(neighbor_id, cur_gen) {
                        continue;
                    }
                    s.mark_visited(neighbor_id, cur_gen);

                    let n_dist = dp.distance(query, neighbor_id);

                    let farthest_dist = s.results.peek().map(|r| r.distance).unwrap_or(f32::MAX);

                    if s.results.len() < ef || n_dist < farthest_dist {
                        s.candidates.push(MinCandidate {
                            id: neighbor_id,
                            distance: n_dist,
                        });
                        s.results.push(MaxCandidate {
                            id: neighbor_id,
                            distance: n_dist,
                        });
                        if s.results.len() > ef {
                            s.results.pop();
                        }
                    }
                }
            }

            // Drain results, filter deleted, sort by distance ascending.
            let mut result_vec: Vec<(u32, f32)> = s
                .results
                .drain()
                .filter(|c| !deleted.contains(&c.id))
                .map(|c| (c.id, c.distance))
                .collect();
            result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            result_vec
        })
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
        dp: &dyn DistanceProvider,
    ) -> (u32, f32) {
        let mut current_id = entry_point;
        let mut current_dist = dp.distance(query, current_id);

        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(current_id, layer);
            for &neighbor_id in neighbors {
                let n_dist = dp.distance(query, neighbor_id);
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

    /// Search a layer with a filter bitmap.
    ///
    /// Only returns results whose IDs are present in `filter`. The search
    /// strategy adapts based on the filter selectivity (fraction of vectors
    /// matching the filter):
    ///
    /// * **High selectivity (>50%)** — pre-filter: run normal HNSW search
    ///   with `ef * 2`, then discard non-matching results.
    /// * **Medium selectivity (1-50%)** — in-algorithm filter: traverse the
    ///   graph normally (visiting all neighbors for connectivity) but only
    ///   admit candidates into the result set when they pass the filter.
    /// * **Low selectivity (<1%)** — post-filter with heavy oversampling
    ///   (`ef * 20`). If too few results survive, fall back to brute-force
    ///   scan of the filtered IDs.
    #[allow(clippy::too_many_arguments)]
    pub fn search_layer_filtered(
        graph: &HnswGraph,
        dp: &dyn DistanceProvider,
        query: &[f32],
        entry_points: &[(u32, f32)],
        ef: usize,
        layer: usize,
        deleted: &HashSet<u32>,
        filter: &RoaringBitmap,
        k: usize,
    ) -> Vec<SearchResult> {
        let total_count = dp.len() as u64;
        let filter_count = filter.len();

        match select_strategy(filter_count, total_count) {
            FilterStrategy::PreFilter => {
                // High selectivity: most candidates pass the filter.
                let adjusted_ef = ef * 2;
                let results =
                    Self::search_layer(query, entry_points, adjusted_ef, layer, graph, dp, deleted);
                results
                    .into_iter()
                    .filter(|&(id, _)| filter.contains(id))
                    .take(k)
                    .map(|(id, distance)| SearchResult { id, distance })
                    .collect()
            }
            FilterStrategy::InAlgorithm => Self::search_layer_in_algorithm_filter(
                graph,
                dp,
                query,
                entry_points,
                ef,
                layer,
                deleted,
                filter,
                k,
            ),
            FilterStrategy::PostFilter => {
                // Low selectivity: oversample heavily, then filter.
                let adjusted_ef = ef * 20;
                let results =
                    Self::search_layer(query, entry_points, adjusted_ef, layer, graph, dp, deleted);
                let filtered: Vec<SearchResult> = results
                    .into_iter()
                    .filter(|&(id, _)| filter.contains(id))
                    .take(k)
                    .map(|(id, distance)| SearchResult { id, distance })
                    .collect();

                if filtered.len() >= k {
                    return filtered;
                }

                // Fallback: brute-force scan of filtered IDs only.
                Self::brute_force_filtered(dp, query, deleted, filter, k)
            }
        }
    }

    /// In-algorithm filtered search: traverse the HNSW graph visiting all
    /// neighbors (preserving connectivity) but only add to the result set
    /// if the candidate passes both the deleted and filter checks.
    #[allow(clippy::too_many_arguments)]
    fn search_layer_in_algorithm_filter(
        graph: &HnswGraph,
        dp: &dyn DistanceProvider,
        query: &[f32],
        entry_points: &[(u32, f32)],
        ef: usize,
        layer: usize,
        deleted: &HashSet<u32>,
        filter: &RoaringBitmap,
        k: usize,
    ) -> Vec<SearchResult> {
        let total_count = dp.len() as u64;
        let filter_count = filter.len();
        let selectivity = if total_count > 0 {
            filter_count as f64 / total_count as f64
        } else {
            1.0
        };

        // Increase ef inversely proportional to selectivity, capped at ef * 10.
        let adjusted_ef = if selectivity > 0.0 {
            ((ef as f64 / selectivity) as usize).min(ef * 10)
        } else {
            ef * 10
        };
        // ACORN 2-hop expansion through filter-failing nodes can
        // blow up the visit count when selectivity is very low and
        // the working heap stays small (so `farthest_working` is
        // f32::MAX and the cheap termination check never fires).
        // Hard cap on visits as a safety net — generous enough that
        // we don't trip it under normal operation but tight enough
        // that an adversarial selectivity can't degenerate to a
        // brute-force scan.
        let max_visits = adjusted_ef.saturating_mul(4).max(1_024);

        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::new();
        // The `working` heap tracks the best candidates seen so far (filtered
        // or not) so the traversal termination condition works correctly.
        let mut working: BinaryHeap<MaxCandidate> = BinaryHeap::new();
        // The `results` heap holds only filtered, non-deleted candidates.
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();

        for &(ep_id, ep_dist) in entry_points {
            visited.insert(ep_id);
            candidates.push(MinCandidate {
                id: ep_id,
                distance: ep_dist,
            });
            working.push(MaxCandidate {
                id: ep_id,
                distance: ep_dist,
            });
            if !deleted.contains(&ep_id) && filter.contains(ep_id) {
                results.push(MaxCandidate {
                    id: ep_id,
                    distance: ep_dist,
                });
            }
        }

        while let Some(MinCandidate {
            id: c_id,
            distance: c_dist,
        }) = candidates.pop()
        {
            if visited.len() >= max_visits {
                break;
            }
            let farthest_working = working.peek().map(|r| r.distance).unwrap_or(f32::MAX);
            if c_dist > farthest_working {
                break;
            }

            let neighbors = graph.get_neighbors(c_id, layer);
            for &neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let n_dist = dp.distance(query, neighbor_id);

                let farthest_working = working.peek().map(|r| r.distance).unwrap_or(f32::MAX);

                let passes_filter = !deleted.contains(&neighbor_id) && filter.contains(neighbor_id);

                // ACORN-1-style expansion (arXiv:2403.04871, shipped by
                // Qdrant 1.16 Nov 2025 / Weaviate 1.27 / Vespa Sep 2025).
                // Two cases:
                //  - Filter-passing neighbours go through the usual
                //    "admit if better than the worst working candidate"
                //    test that bounds the working heap.
                //  - Filter-failing neighbours are enqueued for 2-hop
                //    expansion *unconditionally* (modulo dedup via
                //    `visited`). They do NOT enter the working heap —
                //    so they can't extend the traversal indefinitely
                //    via the `c_dist > farthest_working` termination
                //    check — but they DO get popped from `candidates`
                //    so the search can find filter-passing rows in
                //    their neighbourhood. This is what stops selective
                //    filters from dead-ending in unfiltered regions of
                //    the graph.
                if passes_filter && (working.len() < adjusted_ef || n_dist < farthest_working) {
                    candidates.push(MinCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                    working.push(MaxCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                    if working.len() > adjusted_ef {
                        working.pop();
                    }
                    results.push(MaxCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                } else if !passes_filter {
                    // ACORN 2-hop bypass: enqueue for traversal but
                    // don't admit into the working heap or results.
                    candidates.push(MinCandidate {
                        id: neighbor_id,
                        distance: n_dist,
                    });
                }
            }
        }

        let mut result_vec: Vec<SearchResult> = results
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec.truncate(k);
        result_vec
    }

    /// Brute-force scan over only the IDs in the filter bitmap.
    fn brute_force_filtered(
        dp: &dyn DistanceProvider,
        query: &[f32],
        deleted: &HashSet<u32>,
        filter: &RoaringBitmap,
        k: usize,
    ) -> Vec<SearchResult> {
        let mut heap: BinaryHeap<MaxCandidate> = BinaryHeap::with_capacity(k + 1);

        for id in filter.iter() {
            if deleted.contains(&id) {
                continue;
            }
            if (id as usize) >= dp.len() {
                continue;
            }
            let distance = dp.distance(query, id);
            if heap.len() < k {
                heap.push(MaxCandidate { id, distance });
            } else if let Some(worst) = heap.peek()
                && distance < worst.distance
            {
                heap.pop();
                heap.push(MaxCandidate { id, distance });
            }
        }

        let mut result_vec: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }
}
