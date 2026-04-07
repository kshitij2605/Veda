use super::SuffixArrayIndex;

impl SuffixArrayIndex {
    /// Build a suffix array from a collection of (doc_id, text) pairs.
    ///
    /// Uses an O(n log^2 n) prefix-doubling algorithm:
    /// 1. Concatenate all texts with sentinel (\0) separators
    /// 2. Sort suffixes by single character, compute ranks
    /// 3. Repeatedly double comparison length, re-sort, recompute ranks
    /// 4. Stop when all ranks are unique
    pub fn build(documents: &[(u32, &str)]) -> Self {
        let mut text = Vec::new();
        let mut doc_boundaries = Vec::new();

        for &(doc_id, doc_text) in documents {
            let start = text.len();
            text.extend_from_slice(doc_text.as_bytes());
            let end = text.len();
            doc_boundaries.push((start, end, doc_id));
            text.push(0); // sentinel separator
        }

        let n = text.len();
        if n == 0 {
            return Self {
                text,
                sa: Vec::new(),
                doc_boundaries,
                doc_count: documents.len() as u32,
            };
        }

        let sa = build_suffix_array(&text);

        Self {
            text,
            sa,
            doc_boundaries,
            doc_count: documents.len() as u32,
        }
    }
}

/// Build a suffix array using the O(n log^2 n) prefix-doubling algorithm.
fn build_suffix_array(text: &[u8]) -> Vec<u32> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }

    // Initialize suffix array and rank by single characters.
    let mut sa: Vec<u32> = (0..n as u32).collect();
    let mut rank: Vec<i64> = text.iter().map(|&b| b as i64).collect();
    let mut new_rank = vec![0i64; n];

    let mut k: usize = 1;
    while k < n {
        // Sort by (rank[i], rank[i+k]).
        let rank_ref = &rank;
        sa.sort_by(|&a, &b| {
            let a = a as usize;
            let b = b as usize;
            let ra = rank_ref[a];
            let rb = rank_ref[b];
            if ra != rb {
                return ra.cmp(&rb);
            }
            let ra2 = if a + k < n { rank_ref[a + k] } else { -1 };
            let rb2 = if b + k < n { rank_ref[b + k] } else { -1 };
            ra2.cmp(&rb2)
        });

        // Recompute ranks based on the new sorted order.
        new_rank[sa[0] as usize] = 0;
        for i in 1..n {
            let prev = sa[i - 1] as usize;
            let curr = sa[i] as usize;
            let same = rank[curr] == rank[prev] && {
                let r1 = if curr + k < n { rank[curr + k] } else { -1 };
                let r2 = if prev + k < n { rank[prev + k] } else { -1 };
                r1 == r2
            };
            new_rank[curr] = new_rank[prev] + if same { 0 } else { 1 };
        }

        std::mem::swap(&mut rank, &mut new_rank);

        // If all ranks are unique, we're done.
        if rank[sa[n - 1] as usize] == (n as i64 - 1) {
            break;
        }

        k *= 2;
    }

    sa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_array_sorted() {
        let text = b"banana\0";
        let sa = build_suffix_array(text);
        // Verify that suffixes are in lexicographic order.
        for i in 1..sa.len() {
            let a = &text[sa[i - 1] as usize..];
            let b = &text[sa[i] as usize..];
            assert!(a < b, "suffix at {} not < suffix at {}", sa[i - 1], sa[i]);
        }
    }

    #[test]
    fn test_build_empty() {
        let index = SuffixArrayIndex::build(&[]);
        assert!(index.sa.is_empty());
        assert!(index.text.is_empty());
        assert_eq!(index.doc_count, 0);
    }
}
