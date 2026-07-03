//! The 3-tier tunable parameter space (design doc §Campaign engine
//! mechanics): tier 1 is query-time-only (cheap per trial), tier 2 needs
//! a reindex without re-embedding (medium), tier 3 needs a full
//! re-chunk + re-embed (expensive).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FusionMode {
    Rrf,
    Weighted,
}
const FUSION_MODES: [FusionMode; 2] = [FusionMode::Rrf, FusionMode::Weighted];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexTypeChoice {
    Hnsw,
    DiskAnn,
    IvfPq,
    Flat,
    RaBitQ,
    Sq8,
}
const INDEX_TYPES: [IndexTypeChoice; 6] = [
    IndexTypeChoice::Hnsw,
    IndexTypeChoice::DiskAnn,
    IndexTypeChoice::IvfPq,
    IndexTypeChoice::Flat,
    IndexTypeChoice::RaBitQ,
    IndexTypeChoice::Sq8,
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChunkerStrategyChoice {
    MarkdownRecursive,
    Semantic,
    Page,
    FixedWindow,
    SlidingWindow,
    Section,
    RecursiveCharacter,
}
const CHUNKER_STRATEGIES: [ChunkerStrategyChoice; 7] = [
    ChunkerStrategyChoice::MarkdownRecursive,
    ChunkerStrategyChoice::Semantic,
    ChunkerStrategyChoice::Page,
    ChunkerStrategyChoice::FixedWindow,
    ChunkerStrategyChoice::SlidingWindow,
    ChunkerStrategyChoice::Section,
    ChunkerStrategyChoice::RecursiveCharacter,
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier1Config {
    pub ef_search: usize,
    pub fusion_mode: FusionMode,
    pub dense_weight: f64,
    pub rerank: bool,
    pub top_k: usize,
    pub truncate_to: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier2Config {
    pub index_type: IndexTypeChoice,
    pub ef_construction: usize,
    pub m: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier3Config {
    pub chunker_strategy: ChunkerStrategyChoice,
    pub chunk_size: usize,
    pub overlap: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub tier1: Tier1Config,
    pub tier2: Tier2Config,
    pub tier3: Tier3Config,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tier1: Tier1Config {
                ef_search: 96,
                fusion_mode: FusionMode::Rrf,
                dense_weight: 0.5,
                rerank: false,
                top_k: 10,
                truncate_to: None,
            },
            tier2: Tier2Config {
                index_type: IndexTypeChoice::Hnsw,
                ef_construction: 200,
                m: 16,
            },
            tier3: Tier3Config {
                chunker_strategy: ChunkerStrategyChoice::MarkdownRecursive,
                chunk_size: 1500,
                overlap: 0,
            },
        }
    }
}

fn index_of<T: PartialEq + Copy>(options: &[T], value: T) -> f64 {
    options.iter().position(|o| *o == value).unwrap_or(0) as f64
}
fn from_index<T: Copy>(options: &[T], idx: f64) -> T {
    let i = (idx.round() as usize).min(options.len() - 1);
    options[i]
}

impl PipelineConfig {
    /// Flatten to the `Vec<f64>` shape `MultiObjectiveSearch` operates
    /// over. Order is fixed: [ef_search, fusion_mode_idx, dense_weight,
    /// rerank(0/1), top_k, index_type_idx, ef_construction, m,
    /// chunker_idx, chunk_size, overlap]. `truncate_to` is intentionally
    /// excluded from the search vector for v1 (kept at its default) —
    /// it's a niche MRL knob, not core to the search space.
    pub fn to_param_vec(&self) -> Vec<f64> {
        vec![
            self.tier1.ef_search as f64,
            index_of(&FUSION_MODES, self.tier1.fusion_mode),
            self.tier1.dense_weight,
            if self.tier1.rerank { 1.0 } else { 0.0 },
            self.tier1.top_k as f64,
            index_of(&INDEX_TYPES, self.tier2.index_type),
            self.tier2.ef_construction as f64,
            self.tier2.m as f64,
            index_of(&CHUNKER_STRATEGIES, self.tier3.chunker_strategy),
            self.tier3.chunk_size as f64,
            self.tier3.overlap as f64,
        ]
    }

    pub fn from_param_vec(v: &[f64]) -> Self {
        let mut cfg = Self::default();
        cfg.tier1.ef_search = v[0].round().max(1.0) as usize;
        cfg.tier1.fusion_mode = from_index(&FUSION_MODES, v[1]);
        cfg.tier1.dense_weight = v[2];
        cfg.tier1.rerank = v[3] >= 0.5;
        cfg.tier1.top_k = v[4].round().max(1.0) as usize;
        cfg.tier2.index_type = from_index(&INDEX_TYPES, v[5]);
        cfg.tier2.ef_construction = v[6].round().max(1.0) as usize;
        cfg.tier2.m = v[7].round().max(1.0) as usize;
        cfg.tier3.chunker_strategy = from_index(&CHUNKER_STRATEGIES, v[8]);
        cfg.tier3.chunk_size = v[9].round().max(1.0) as usize;
        cfg.tier3.overlap = v[10].round().max(0.0) as usize;
        cfg
    }

    /// True when `self` and `other` share the same tier-2 AND tier-3
    /// config — i.e. evaluating `other` after `self` needs no new
    /// `/reindex` call, only a different tier-1 query. Used by the
    /// campaign engine (Phase 4) to group trials under one materialized
    /// base.
    pub fn same_tier2_and_3_base(&self, other: &Self) -> bool {
        self.tier2 == other.tier2 && self.tier3 == other.tier3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_round_trips_through_param_vec() {
        let cfg = PipelineConfig::default();
        let vec = cfg.to_param_vec();
        let back = PipelineConfig::from_param_vec(&vec);
        assert_eq!(back, cfg);
    }

    #[test]
    fn tier_boundary_helper_reports_whether_two_configs_share_a_base() {
        let a = PipelineConfig::default();
        let mut b = a.clone();
        b.tier1.ef_search = 64;
        assert!(
            a.same_tier2_and_3_base(&b),
            "only tier1 differs -> same base"
        );

        let mut c = a.clone();
        c.tier2.index_type = IndexTypeChoice::Flat;
        assert!(
            !a.same_tier2_and_3_base(&c),
            "tier2 differs -> different base"
        );
    }
}
