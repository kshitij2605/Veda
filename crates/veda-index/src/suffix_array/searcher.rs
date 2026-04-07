use super::{PatternMatch, PatternSearchConfig, SuffixArrayIndex};

impl SuffixArrayIndex {
    /// Find all exact occurrences of a pattern using binary search on the suffix array.
    pub fn search_exact(&self, pattern: &str) -> Vec<PatternMatch> {
        if pattern.is_empty() || self.sa.is_empty() {
            return Vec::new();
        }

        let pat = pattern.as_bytes();
        let (lo, hi) = self.sa_range(pat);
        if lo > hi {
            return Vec::new();
        }

        let mut matches = Vec::new();
        for &sa_idx in &self.sa[lo..=hi] {
            let pos = sa_idx as usize;
            if let Some((doc_id, doc_start)) = self.doc_for_offset(pos) {
                matches.push(PatternMatch {
                    doc_id,
                    offset: pos - doc_start,
                    length: pat.len(),
                });
            }
        }
        matches
    }

    /// Fuzzy pattern search with edit distance tolerance.
    ///
    /// Strategy:
    /// 1. Split pattern into words
    /// 2. For each word, find exact + fuzzy (edit distance <= max_edits) matches
    /// 3. If preserve_order, check that matched words appear in ascending order
    ///    within each document
    /// 4. Return matches for qualifying documents
    pub fn search_fuzzy(
        &self,
        pattern: &str,
        config: &PatternSearchConfig,
    ) -> Vec<PatternMatch> {
        if pattern.is_empty() || self.sa.is_empty() {
            return Vec::new();
        }

        let words: Vec<&str> = pattern.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        // For each word, collect matches grouped by doc_id.
        // word_doc_positions[word_idx] = Vec<(doc_id, offset_in_doc)>
        let mut word_doc_positions: Vec<Vec<(u32, usize)>> = Vec::with_capacity(words.len());

        for word in &words {
            let mut positions: Vec<(u32, usize)> = Vec::new();

            // Exact matches for the word.
            for m in self.search_exact(word) {
                positions.push((m.doc_id, m.offset));
            }

            // Fuzzy: generate edit-distance variants and search each.
            if config.fuzzy && config.max_edits > 0 {
                let variants = generate_variants(word, config.max_edits);
                for variant in &variants {
                    for m in self.search_exact(variant) {
                        positions.push((m.doc_id, m.offset));
                    }
                }
            }

            // Deduplicate by (doc_id, offset).
            positions.sort();
            positions.dedup();
            word_doc_positions.push(positions);
        }

        // Find documents that contain ALL words.
        let mut result_matches = Vec::new();

        if words.len() == 1 {
            // Single word: every match qualifies.
            for &(doc_id, offset) in &word_doc_positions[0] {
                result_matches.push(PatternMatch {
                    doc_id,
                    offset,
                    length: words[0].len(),
                });
                if result_matches.len() >= config.max_results {
                    return result_matches;
                }
            }
            return result_matches;
        }

        // Collect doc_ids that appear in all word position lists.
        let mut common_docs: std::collections::BTreeSet<u32> = word_doc_positions[0]
            .iter()
            .map(|&(doc_id, _)| doc_id)
            .collect();
        for word_pos in &word_doc_positions[1..] {
            let doc_set: std::collections::BTreeSet<u32> =
                word_pos.iter().map(|&(doc_id, _)| doc_id).collect();
            common_docs = common_docs.intersection(&doc_set).copied().collect();
        }

        for doc_id in &common_docs {
            // Gather per-word offsets for this document.
            let per_word: Vec<Vec<usize>> = word_doc_positions
                .iter()
                .map(|positions| {
                    positions
                        .iter()
                        .filter(|&&(did, _)| did == *doc_id)
                        .map(|&(_, off)| off)
                        .collect()
                })
                .collect();

            if config.preserve_order {
                // Check that we can pick one offset per word in ascending order.
                if has_ascending_sequence(&per_word) {
                    // Report the first word's first matching offset for simplicity.
                    if let Some(&first_off) = per_word[0].iter().min() {
                        result_matches.push(PatternMatch {
                            doc_id: *doc_id,
                            offset: first_off,
                            length: pattern.len(),
                        });
                    }
                }
            } else {
                // All words present, no order constraint.
                if let Some(&first_off) = per_word[0].iter().min() {
                    result_matches.push(PatternMatch {
                        doc_id: *doc_id,
                        offset: first_off,
                        length: pattern.len(),
                    });
                }
            }

            if result_matches.len() >= config.max_results {
                break;
            }
        }

        result_matches
    }

    /// Binary search to find the range [lo, hi] of suffix array entries
    /// whose suffixes start with `pattern`.
    fn sa_range(&self, pattern: &[u8]) -> (usize, usize) {
        let n = self.sa.len();

        // Find lower bound: first suffix >= pattern.
        let lo = {
            let mut left = 0usize;
            let mut right = n;
            while left < right {
                let mid = left + (right - left) / 2;
                let suffix_start = self.sa[mid] as usize;
                let suffix_end = (suffix_start + pattern.len()).min(self.text.len());
                let suffix = &self.text[suffix_start..suffix_end];
                if suffix < pattern {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            left
        };

        // Find upper bound: last suffix that starts with pattern.
        let hi = {
            let mut left = lo;
            let mut right = n;
            while left < right {
                let mid = left + (right - left) / 2;
                let suffix_start = self.sa[mid] as usize;
                let suffix_end = (suffix_start + pattern.len()).min(self.text.len());
                let suffix = &self.text[suffix_start..suffix_end];
                if suffix <= pattern {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            // hi is left - 1 (the last position that matches), but we need to
            // handle the case where left == lo (no matches).
            if left == 0 {
                return (1, 0); // empty range sentinel
            }
            left - 1
        };

        if lo > hi {
            return (1, 0); // empty range sentinel
        }

        // Verify the range boundaries actually match.
        let check_start = self.sa[lo] as usize;
        let check_end = (check_start + pattern.len()).min(self.text.len());
        if &self.text[check_start..check_end] != pattern {
            return (1, 0); // empty range sentinel
        }

        (lo, hi)
    }
}

/// Check whether we can pick one offset from each word's offset list
/// such that they form a strictly ascending sequence.
fn has_ascending_sequence(per_word: &[Vec<usize>]) -> bool {
    if per_word.is_empty() {
        return true;
    }
    // Greedy: for each word, pick the smallest offset > previous.
    let mut prev = 0usize;
    let mut first = true;
    for offsets in per_word {
        let mut sorted = offsets.clone();
        sorted.sort();
        let found = if first {
            sorted.first().copied()
        } else {
            sorted.iter().find(|&&o| o > prev).copied()
        };
        match found {
            Some(off) => {
                prev = off;
                first = false;
            }
            None => return false,
        }
    }
    true
}

/// Generate all strings within edit distance `max_edits` of `word`.
/// Supports substitution, insertion, and deletion operations.
fn generate_variants(word: &str, max_edits: usize) -> Vec<String> {
    let mut results = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let alphabet: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    // For tractability, only generate single-edit variants (recursive for max_edits > 1
    // would be exponential). Apply edits up to max_edits times iteratively.
    let mut current_level = vec![chars.clone()];

    for _ in 0..max_edits {
        let mut next_level = Vec::new();
        for base in &current_level {
            // Substitutions.
            for i in 0..base.len() {
                for &c in &alphabet {
                    if c != base[i] {
                        let mut variant = base.clone();
                        variant[i] = c;
                        next_level.push(variant);
                    }
                }
            }
            // Deletions.
            for i in 0..base.len() {
                let mut variant = base.clone();
                variant.remove(i);
                next_level.push(variant);
            }
            // Insertions.
            for i in 0..=base.len() {
                for &c in &alphabet {
                    let mut variant = base.clone();
                    variant.insert(i, c);
                    next_level.push(variant);
                }
            }
        }
        // Deduplicate and exclude the original word.
        next_level.sort();
        next_level.dedup();
        next_level.retain(|v| v != &chars);
        for v in &next_level {
            let s: String = v.iter().collect();
            results.push(s);
        }
        current_level = next_level;
    }

    results.sort();
    results.dedup();
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_variants_substitution() {
        let variants = generate_variants("cat", 1);
        // "bat" is a substitution of 'c' -> 'b'
        assert!(variants.contains(&"bat".to_string()));
        // "cat" itself should not be in the variants
        assert!(!variants.contains(&"cat".to_string()));
    }

    #[test]
    fn test_generate_variants_deletion() {
        let variants = generate_variants("cat", 1);
        assert!(variants.contains(&"at".to_string()));
        assert!(variants.contains(&"ct".to_string()));
        assert!(variants.contains(&"ca".to_string()));
    }

    #[test]
    fn test_generate_variants_insertion() {
        let variants = generate_variants("at", 1);
        assert!(variants.contains(&"cat".to_string()));
        assert!(variants.contains(&"bat".to_string()));
    }

    #[test]
    fn test_has_ascending_sequence() {
        assert!(has_ascending_sequence(&[vec![1, 5], vec![3, 7]]));
        assert!(!has_ascending_sequence(&[vec![5], vec![3]]));
        assert!(has_ascending_sequence(&[vec![1], vec![2], vec![3]]));
    }
}
