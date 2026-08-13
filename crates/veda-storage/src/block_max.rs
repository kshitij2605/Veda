//! Block-Max WAND: disjunctive top-k retrieval that skips whole blocks of
//! postings whose best possible contribution cannot reach the current
//! top-k threshold.
//!
//! Deliberately domain-agnostic. Everything specific to a scoring model
//! lives behind `PostingCursor`, so the same core serves BM25 postings and
//! sparse-vector postings without change.
//!
//! Reference: Ding & Suel, "Faster top-k document retrieval using
//! block-max indexes", SIGIR 2011.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Postings per block. 128 is the standard choice: large enough that the
/// per-block metadata is negligible against the postings it covers, small
/// enough that a block's maximum stays a tight bound on its members.
pub const BLOCK_SIZE: usize = 128;

/// Ceiling on the heap's preallocated capacity. Purely a preallocation
/// hint — the heap never holds more than `k` entries regardless of this
/// value — so capping it just avoids an oversized (or, at `k ==
/// usize::MAX`, outright panicking) allocation request for a `k` no real
/// top-k result set would ever reach.
const MAX_HEAP_PREALLOC: usize = 4096;

/// A cursor over one term's postings, positioned at a document.
///
/// Implementors must visit postings in ascending `doc_id` order — the
/// algorithm's skipping is only sound if the lists are doc-sorted.
pub trait PostingCursor {
    /// Current doc_id, or `None` once the list is exhausted.
    fn doc(&self) -> Option<u32>;

    /// Advance to the first doc_id `>= target`, exhausting the cursor if
    /// there is none. Never moves backwards.
    fn advance_to(&mut self, target: u32);

    /// Advance past the current doc.
    fn next(&mut self);

    /// Upper bound on the contribution of any posting in the current
    /// block. Meaningless once exhausted.
    fn block_max(&self) -> f32;

    /// Last doc_id in the current block, used to skip the block wholesale.
    ///
    /// Must be `>= doc()` whenever `doc()` is `Some`. The traversal skips
    /// a block by jumping straight to `last + 1`, and relies on `last`
    /// never trailing the cursor's own position for that jump to make
    /// forward progress; a cursor that violates this can stall the
    /// traversal.
    fn block_last_doc(&self) -> Option<u32>;

    /// Upper bound on the contribution of any posting in the whole list.
    /// Constant for the cursor's lifetime.
    fn list_max(&self) -> f32;

    /// Exact contribution of the current doc.
    fn score(&self) -> f32;
}

/// How much work the traversal avoided. Used by tests to prove the pruning
/// path is not inert, and by the benchmark to report effectiveness.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    /// Documents fully scored across all cursors.
    pub docs_evaluated: u64,
    /// Block alignments rejected without scoring any of their postings.
    pub blocks_skipped: u64,
}

/// Heap entry ordered so that `BinaryHeap::peek` yields the *worst* result
/// currently held — lowest score, and among equal scores the highest
/// doc_id, since the final order breaks ties by ascending doc_id.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    score: f32,
    doc: u32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then(self.doc.cmp(&other.doc))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Disjunctive top-k over `cursors`.
///
/// Returns `(doc_id, score)` sorted by score descending with ties broken by
/// doc_id ascending, truncated to `k`, plus traversal statistics.
///
/// `theta` scales the pruning threshold. `1.0` is rank-safe: the result is
/// identical to scoring every posting exhaustively. Values above `1.0`
/// prune more aggressively and may omit true top-k results. Values below
/// `1.0` and non-finite values are clamped to `1.0` — they could only make
/// the traversal slower without changing the result.
///
/// A document's score is the sum of every cursor positioned on it at
/// scoring time, added in `cursors`' *original* order — the order the
/// `Vec` was passed in, not the doc-sorted order the traversal sorts them
/// into internally (ties in that internal sort are broken by original
/// position specifically so this holds). `f32` addition is not
/// associative, so this order is significant, and it feeds heap admission,
/// pivot selection and block-skip thresholds, not just the reported score
/// — a caller that wants `theta = 1.0` results identical to some reference
/// scorer, bit for bit, must build `cursors` in the same order that
/// reference scorer sums a document's terms in (e.g. query term order),
/// not e.g. an order derived from iterating a `HashMap`, which would make
/// scores — and, at tie boundaries, which documents are admitted at all —
/// nondeterministic from run to run.
pub fn block_max_wand<C: PostingCursor>(
    cursors: Vec<C>,
    k: usize,
    theta: f32,
) -> (Vec<(u32, f32)>, PruneStats) {
    let mut stats = PruneStats::default();
    if k == 0 {
        return (Vec::new(), stats);
    }
    let theta = if theta.is_finite() && theta > 1.0 {
        theta
    } else {
        1.0
    };

    // Pair each cursor with its original construction index so that ties —
    // multiple cursors positioned at the same doc — always resolve in the
    // caller's original order, no matter how the traversal's history of
    // skips and advances happened to interleave them beforehand. This is
    // what lets a doc-sorted, term-ordered pivot sum come out bit-identical
    // to a term-ordered exhaustive sum: same values, same order, same
    // starting point, every time.
    let mut cursors: Vec<(usize, C)> = cursors.into_iter().enumerate().collect();

    let mut heap: BinaryHeap<Entry> =
        BinaryHeap::with_capacity(k.saturating_add(1).min(MAX_HEAP_PREALLOC));

    'outer: loop {
        cursors.retain(|(_, c)| c.doc().is_some());
        if cursors.is_empty() {
            break;
        }
        cursors.sort_by_key(|(i, c)| (c.doc().unwrap_or(u32::MAX), *i));

        // Until the heap is full nothing may be pruned, so the bound is
        // -inf rather than 0.0 — a corpus whose scores are all zero must
        // still fill k results.
        let bound = if heap.len() >= k {
            theta * heap.peek().map(|e| e.score).unwrap_or(0.0)
        } else {
            f32::NEG_INFINITY
        };

        // Pivot: the first cursor at which the cumulative list maxima could
        // still reach the threshold. `>=` (not `>`) because a candidate
        // that exactly ties the threshold can still win on doc_id.
        let mut acc = 0.0f32;
        let mut pivot = None;
        for (i, (_, c)) in cursors.iter().enumerate() {
            acc += c.list_max();
            if acc >= bound {
                pivot = Some(i);
                break;
            }
        }
        let Some(p) = pivot else {
            // No remaining document can reach the threshold: drop
            // everything still in flight and stop immediately. This is
            // BMW's single biggest win, and it must stay cheap — walking
            // the abandoned tail block-by-block just to count it would
            // turn "we already know the answer" into real, unbounded
            // seek traffic against a long posting list, exactly when the
            // caller asked for more pruning (higher theta), not less.
            break;
        };
        let pivot_doc = match cursors[p].1.doc() {
            Some(d) => d,
            None => break,
        };
        let first_doc = match cursors[0].1.doc() {
            Some(d) => d,
            None => break,
        };

        if first_doc < pivot_doc {
            // Cursors trail the pivot: catch them up. The pivot's own
            // selection already proves no doc before pivot_doc can reach
            // the bound (the cursors before it, summed, fall short of the
            // threshold by definition) — so any block of a lagging cursor
            // that ends before pivot_doc is provably worthless and can be
            // skipped wholesale rather than seeked through blindly. Only
            // the boundary block (the one containing or straddling
            // pivot_doc) gets a plain seek, since skipping it outright
            // could jump past pivot_doc itself. Guaranteed progress: each
            // branch strictly advances the cursor's doc.
            for (_, c) in cursors[..p].iter_mut() {
                while let Some(d) = c.doc().filter(|&d| d < pivot_doc) {
                    match c.block_last_doc() {
                        Some(last) if last < pivot_doc => {
                            debug_assert!(
                                last >= d,
                                "PostingCursor::block_last_doc() must be >= doc()"
                            );
                            let target = last.saturating_add(1);
                            if target <= d {
                                // Only reachable via a cursor that violates
                                // the block_last_doc >= doc() contract (the
                                // assertion above already caught this in
                                // debug builds) or sits at doc = u32::MAX;
                                // either way nothing can move it forward,
                                // so stop rather than spin.
                                break 'outer;
                            }
                            stats.blocks_skipped += 1;
                            c.advance_to(target);
                        }
                        _ => c.advance_to(pivot_doc),
                    }
                }
            }
            continue;
        }

        // Cursors past `p` that coincidentally sit at the same doc are
        // guaranteed contiguous with `p` by the doc-sorted order. Their
        // contribution still counts toward pivot_doc's achievable score,
        // so the "necessary" range for the block-level bound must extend
        // to cover them — stopping at `p` would understate the bound and
        // could skip a doc that actually reaches the threshold.
        let end = p + cursors[p..].partition_point(|(_, c)| c.doc() == Some(pivot_doc));

        // Block-max check. `<` (not `<=`) preserves exact ties, as above.
        let block_sum: f32 = cursors[..end].iter().map(|(_, c)| c.block_max()).sum();
        if block_sum < bound {
            stats.blocks_skipped += 1;
            let block_end = cursors[..end]
                .iter()
                .filter_map(|(_, c)| {
                    let last = c.block_last_doc();
                    if let Some(l) = last {
                        debug_assert!(
                            l >= pivot_doc,
                            "PostingCursor::block_last_doc() must be >= doc()"
                        );
                    }
                    last
                })
                .min()
                .map_or(u32::MAX, |d| d.saturating_add(1));
            let next_after_pivot = cursors
                .get(end)
                .and_then(|(_, c)| c.doc())
                .unwrap_or(u32::MAX);
            // Never overshoot the next candidate, and always make progress.
            let target = block_end
                .min(next_after_pivot)
                .max(pivot_doc.saturating_add(1));
            if target <= pivot_doc {
                // Only reachable at doc = u32::MAX, where nothing (not even
                // pivot_doc + 1) can exceed the current position; stop
                // rather than spin.
                break;
            }
            for (_, c) in cursors[..end].iter_mut() {
                if c.doc().is_some_and(|d| d < target) {
                    c.advance_to(target);
                }
            }
            continue;
        }

        // Score the pivot. Cursors are doc-sorted and cursors[0] is at the
        // pivot, so every cursor positioned there forms a prefix. The
        // stable-by-original-index sort above means this prefix is always
        // visited in the same original construction order, so the sum
        // below matches a term-ordered exhaustive sum exactly.
        let mut score = 0.0f32;
        for (_, c) in cursors.iter_mut() {
            if c.doc() == Some(pivot_doc) {
                score += c.score();
                c.next();
            } else {
                break;
            }
        }
        stats.docs_evaluated += 1;

        if heap.len() < k {
            heap.push(Entry {
                score,
                doc: pivot_doc,
            });
        } else if let Some(worst) = heap.peek().copied()
            && (score > worst.score || (score == worst.score && pivot_doc < worst.doc))
        {
            heap.pop();
            heap.push(Entry {
                score,
                doc: pivot_doc,
            });
        }
    }

    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|e| (e.doc, e.score)).collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test double: an explicit posting list of `(doc_id, score)` with
    /// block metadata derived the same way a real cursor derives it.
    struct VecCursor {
        postings: Vec<(u32, f32)>,
        blocks: Vec<(u32, f32)>,
        list_max: f32,
        pos: usize,
    }

    impl VecCursor {
        fn new(mut postings: Vec<(u32, f32)>) -> Self {
            postings.sort_by_key(|p| p.0);
            let mut blocks = Vec::new();
            for chunk in postings.chunks(BLOCK_SIZE) {
                let last_doc = chunk.last().map(|p| p.0).unwrap_or(0);
                let max = chunk.iter().map(|p| p.1).fold(f32::MIN, f32::max);
                blocks.push((last_doc, max));
            }
            let list_max = blocks.iter().map(|b| b.1).fold(f32::MIN, f32::max);
            Self {
                postings,
                blocks,
                list_max,
                pos: 0,
            }
        }
    }

    impl PostingCursor for VecCursor {
        fn doc(&self) -> Option<u32> {
            self.postings.get(self.pos).map(|p| p.0)
        }
        fn advance_to(&mut self, target: u32) {
            self.pos += self.postings[self.pos.min(self.postings.len())..]
                .partition_point(|p| p.0 < target);
        }
        fn next(&mut self) {
            self.pos += 1;
        }
        fn block_max(&self) -> f32 {
            self.blocks
                .get(self.pos / BLOCK_SIZE)
                .map_or(f32::MIN, |b| b.1)
        }
        fn block_last_doc(&self) -> Option<u32> {
            self.blocks.get(self.pos / BLOCK_SIZE).map(|b| b.0)
        }
        fn list_max(&self) -> f32 {
            self.list_max
        }
        fn score(&self) -> f32 {
            self.postings[self.pos].1
        }
    }

    /// Reference implementation: score everything, sort, truncate. This is
    /// the oracle every correctness test compares against.
    fn exhaustive(lists: &[Vec<(u32, f32)>], k: usize) -> Vec<(u32, f32)> {
        let mut acc: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for list in lists {
            for &(doc, s) in list {
                *acc.entry(doc).or_insert(0.0) += s;
            }
        }
        let mut out: Vec<(u32, f32)> = acc.into_iter().collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k);
        out
    }

    fn cursors_for(lists: &[Vec<(u32, f32)>]) -> Vec<VecCursor> {
        lists.iter().cloned().map(VecCursor::new).collect()
    }

    /// xorshift, so failures reproduce exactly.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let (hits, _) = block_max_wand(Vec::<VecCursor>::new(), 10, 1.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn zero_k_returns_empty() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 0, 1.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn single_list_matches_exhaustive() {
        let lists = vec![vec![(5u32, 1.0f32), (1, 3.0), (9, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, 1.0);
        assert_eq!(hits, exhaustive(&lists, 2));
    }

    #[test]
    fn disjoint_lists_sum_correctly() {
        let lists = vec![
            vec![(1u32, 1.0f32), (2, 1.0)],
            vec![(2u32, 5.0f32), (3, 1.0)],
        ];
        let (hits, _) = block_max_wand(cursors_for(&lists), 3, 1.0);
        assert_eq!(hits, exhaustive(&lists, 3));
        assert_eq!(hits[0], (2, 6.0), "doc 2 appears in both lists");
    }

    #[test]
    fn k_larger_than_corpus() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 100, 1.0);
        assert_eq!(hits, exhaustive(&lists, 100));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn exact_score_ties_are_not_pruned() {
        // Every doc scores exactly 1.0. A `<=` pruning comparison would
        // stop after the first k and miss lower doc_ids that must win the
        // tie-break; `<` keeps them in play.
        let lists = vec![(0u32..500).map(|d| (d, 1.0f32)).collect::<Vec<_>>()];
        let (hits, _) = block_max_wand(cursors_for(&lists), 5, 1.0);
        assert_eq!(hits, exhaustive(&lists, 5));
        assert_eq!(
            hits.iter().map(|h| h.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn randomized_differential_against_exhaustive() {
        let mut rng = Rng(0xDEAD_BEEF_0000_0001);
        for trial in 0..300 {
            let n_lists = 1 + (rng.next_u32() % 6) as usize;
            let n_docs = 1 + (rng.next_u32() % 2000);
            let lists: Vec<Vec<(u32, f32)>> = (0..n_lists)
                .map(|_| {
                    let len = 1 + (rng.next_u32() % n_docs) as usize;
                    let mut seen = std::collections::HashSet::new();
                    let mut list: Vec<(u32, f32)> = Vec::new();
                    for _ in 0..len {
                        let doc = rng.next_u32() % n_docs;
                        if seen.insert(doc) {
                            // Quantized scores so exact ties actually occur.
                            let s = (rng.next_u32() % 20) as f32 / 4.0;
                            list.push((doc, s));
                        }
                    }
                    list.sort_by_key(|p| p.0);
                    list
                })
                .collect();
            for k in [1usize, 3, 10, 50] {
                let (hits, _) = block_max_wand(cursors_for(&lists), k, 1.0);
                assert_eq!(hits, exhaustive(&lists, k), "trial {trial} k {k}");
            }
        }
    }

    #[test]
    fn blocks_are_actually_skipped() {
        // One very common low-scoring term and one rare high-scoring term.
        // Once the heap fills from the rare term, most of the common term's
        // blocks cannot reach the threshold and must be skipped wholesale.
        let common: Vec<(u32, f32)> = (0u32..20_000).map(|d| (d, 0.01)).collect();
        let rare: Vec<(u32, f32)> = (0u32..40).map(|d| (d * 500, 50.0)).collect();
        let lists = vec![common, rare];
        let (hits, stats) = block_max_wand(cursors_for(&lists), 10, 1.0);
        assert_eq!(hits, exhaustive(&lists, 10));
        assert!(
            stats.blocks_skipped > 0,
            "no blocks skipped — the pruning path is inert"
        );
    }

    #[test]
    fn block_max_check_skips_low_scoring_blocks() {
        // A single list: with only one cursor, `p` is always 0 (there is
        // no second cursor to trail behind, so the catch-up branch can
        // never fire) and `list_max` is the list's true maximum, so the
        // pivot search can never fail either. Any nonzero
        // `blocks_skipped` observed here can therefore only have come
        // from the block-max rejection (`block_sum < bound`) itself,
        // unlike `blocks_are_actually_skipped`, whose two-list, uniform
        // per-list scores make `block_max == list_max` for every block
        // and so give that check no coverage at all.
        let mut postings: Vec<(u32, f32)> =
            (0u32..128).map(|d| (d, (d % 17) as f32 + 1.0)).collect();
        postings.extend((128u32..20_000).map(|d| (d, 0.01)));
        let lists = vec![postings];
        let (hits, stats) = block_max_wand(cursors_for(&lists), 10, 1.0);
        assert_eq!(hits, exhaustive(&lists, 10));
        assert!(
            stats.blocks_skipped > 0,
            "no blocks skipped — the block-max check specifically is inert"
        );
    }

    #[test]
    fn theta_above_one_prunes_at_least_as_much() {
        let common: Vec<(u32, f32)> = (0u32..20_000).map(|d| (d, 0.01)).collect();
        let rare: Vec<(u32, f32)> = (0u32..40).map(|d| (d * 500, 50.0)).collect();
        let lists = vec![common, rare];
        let (_, exact) = block_max_wand(cursors_for(&lists), 10, 1.0);
        let (approx_hits, approx) = block_max_wand(cursors_for(&lists), 10, 4.0);
        assert!(approx.docs_evaluated <= exact.docs_evaluated);
        assert!(approx_hits.len() <= 10);
    }

    #[test]
    fn theta_below_one_is_clamped_to_exact() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0), (3, 3.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, 0.1);
        assert_eq!(hits, exhaustive(&lists, 2));
    }

    #[test]
    fn non_finite_theta_is_clamped_to_exact() {
        let lists = vec![vec![(1u32, 1.0f32), (2, 2.0), (3, 3.0)]];
        let (hits, _) = block_max_wand(cursors_for(&lists), 2, f32::NAN);
        assert_eq!(hits, exhaustive(&lists, 2));
    }

    /// Regression test for the `(doc, original_index)` tie-break in the
    /// per-iteration `sort_by_key`: this is the one property that makes
    /// `block_max_wand`'s pivot-doc score sum bit-identical to a
    /// term-ordered exhaustive sum. Every other test in this module uses
    /// scores that are exact multiples of `0.25` (or small integers), which
    /// are exactly representable in binary and so sum to the same `f32`
    /// result regardless of addition order — none of them would notice if
    /// the `*i` were deleted from that sort key.
    ///
    /// Four lists, three of which are present or absent on a different
    /// period (5, 11, 3) so their relative doc-tie order keeps reshuffling
    /// across iterations instead of staying list-order the whole time, plus
    /// a fourth present on every doc — this is the structure under which an
    /// earlier round of this same feature was caught producing 1-ULP
    /// mismatches against an oracle. `1.0 / 3.0` is not exactly
    /// representable in binary, so the different summation order the
    /// traversal takes without the tie-break genuinely lands on a different
    /// last mantissa bit here (confirmed by hand while writing this test:
    /// temporarily deleting the `*i` tie-break made this test fail with a
    /// concrete `left != right` diff; restoring it made it pass again).
    #[test]
    fn tie_break_keeps_non_dyadic_sums_bit_identical_to_exhaustive() {
        const UNIT: f32 = 1.0 / 3.0;
        let always: Vec<(u32, f32)> = (0u32..3000)
            .map(|d| (d, UNIT * ((d % 5) + 1) as f32))
            .collect();
        let period_11: Vec<(u32, f32)> = (0u32..3000)
            .filter(|d| d % 11 != 0)
            .map(|d| (d, UNIT * ((d % 11) + 1) as f32))
            .collect();
        let period_3: Vec<(u32, f32)> = (0u32..3000)
            .filter(|d| d % 3 != 0)
            .map(|d| (d, UNIT * ((d % 3) + 1) as f32))
            .collect();
        let constant: Vec<(u32, f32)> = (0u32..3000).map(|d| (d, UNIT)).collect();
        let lists = vec![always, period_11, period_3, constant];
        for k in [1usize, 5, 10, 100, 1000, 5000] {
            let (hits, _) = block_max_wand(cursors_for(&lists), k, 1.0);
            assert_eq!(hits, exhaustive(&lists, k), "k {k}");
        }
    }
}
