//! Adaptive query routing.
//!
//! Inspects a natural-language query and recommends a retrieval strategy:
//! sparse / dense / hybrid, with or without a rerank stage. Implements the
//! cheap heuristics from "Adaptive-RAG" (2024) — token count, question-mark
//! detection, presence of named entities (approximated by capitalised
//! words). For production quality this would be replaced by a learnt
//! classifier; this version is purely a fast first-pass router.

use serde::Serialize;

/// Recommended retrieval strategy for a query.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum RouteStrategy {
    /// Short keyword-like query: prefer sparse (BM25 / SPLADE) retrieval.
    Sparse,
    /// Typical semantic query: dense vector search.
    Dense,
    /// Long or multi-faceted query: dense + sparse fusion, then rerank.
    Hybrid,
}

/// A routing recommendation including a suggested `k` and whether to apply
/// a cross-encoder rerank stage.
#[derive(Debug, Clone, Serialize)]
pub struct RouteRecommendation {
    pub strategy: RouteStrategy,
    pub rerank: bool,
    pub suggested_k: usize,
    pub reasoning: String,
}

/// Analyse a query and return a routing recommendation.
pub fn route_query(query: &str) -> RouteRecommendation {
    let trimmed = query.trim();
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let n_tokens = tokens.len();
    let has_question_mark = trimmed.contains('?');
    let capitalised_words = tokens
        .iter()
        .filter(|t| t.chars().next().is_some_and(|c| c.is_uppercase()))
        .count();
    let avg_token_len = if n_tokens == 0 {
        0.0
    } else {
        tokens.iter().map(|t| t.len()).sum::<usize>() as f32 / n_tokens as f32
    };

    // Very short → sparse (keyword-style). A long word like "photosynthesis"
    // suggests a semantic concept rather than a keyword search, so we only
    // route to sparse when tokens are short on average.
    if n_tokens <= 3 && avg_token_len <= 6.0 {
        return RouteRecommendation {
            strategy: RouteStrategy::Sparse,
            rerank: false,
            suggested_k: 20,
            reasoning: format!(
                "{} short tokens, no clear semantic intent → keyword/sparse path",
                n_tokens
            ),
        };
    }

    // Long, multi-clause or question → hybrid + rerank.
    if n_tokens >= 12 || has_question_mark || capitalised_words >= 2 {
        return RouteRecommendation {
            strategy: RouteStrategy::Hybrid,
            rerank: true,
            suggested_k: 50,
            reasoning: if has_question_mark {
                "question form detected → hybrid + cross-encoder rerank".into()
            } else if n_tokens >= 12 {
                format!("{n_tokens} tokens, long query → hybrid + rerank for precision")
            } else {
                format!("{capitalised_words} entities → hybrid + rerank for disambiguation")
            },
        };
    }

    // Default mid-range: dense vector search.
    RouteRecommendation {
        strategy: RouteStrategy::Dense,
        rerank: false,
        suggested_k: 10,
        reasoning: format!("{n_tokens}-token semantic query → dense vector path"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_keyword_routes_sparse() {
        let r = route_query("red shoes");
        assert_eq!(r.strategy, RouteStrategy::Sparse);
        assert!(!r.rerank);
    }

    #[test]
    fn question_form_routes_hybrid() {
        let r = route_query("how does HNSW pruning work?");
        assert_eq!(r.strategy, RouteStrategy::Hybrid);
        assert!(r.rerank);
    }

    #[test]
    fn long_query_routes_hybrid() {
        let r = route_query(
            "find me documents about distributed databases and how they handle network partitions during write operations",
        );
        assert_eq!(r.strategy, RouteStrategy::Hybrid);
    }

    #[test]
    fn typical_query_routes_dense() {
        let r = route_query("documents about photosynthesis");
        assert_eq!(r.strategy, RouteStrategy::Dense);
    }
}
