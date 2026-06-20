//! ColPali-shape multi-vector field: end-to-end through SegmentWriter +
//! SegmentReader.
//!
//! ColPali (and the cousins ColBERT / ColQwen) represent a document as
//! a *list* of token vectors and a query as another list, scored with
//! MaxSim — for each query token, take the max dot product with any
//! document token; sum across query tokens. This validates that
//! Veda's `multi_vector` field type round-trips correctly through the
//! existing storage layer, since MUVERA + PLAID + Light-ColPali all
//! sit on top of that storage.

use tempfile::TempDir;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;
use veda_storage::segment::{SegmentReader, SegmentWriter};

const COLPALI_DIM: u32 = 16;
const TOKENS_PER_DOC: usize = 8;

fn schema() -> CollectionSchema {
    CollectionSchema::builder("colpali_smoke")
        .add_field(FieldDefinition::multi_vector(
            "colpali_tokens",
            COLPALI_DIM,
            64, // max_vectors per record
            DistanceMetric::DotProduct,
        ))
        .build()
        .unwrap()
}

/// Build a synthetic per-document multi-vector. Each "token" is a
/// deterministic pseudo-random vector seeded by `(doc_seed, t)`.
fn tokens_for(doc_seed: u32) -> Vec<Vec<f32>> {
    (0..TOKENS_PER_DOC as u32)
        .map(|t| {
            let mut s = doc_seed.wrapping_mul(31).wrapping_add(t);
            (0..COLPALI_DIM)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((s >> 16) as f32 / 32768.0) - 1.0
                })
                .collect()
        })
        .collect()
}

fn mk_record(doc_seed: u32) -> Record {
    let mut r = Record::new();
    r.set(
        "colpali_tokens",
        FieldValue::MultiVector(tokens_for(doc_seed)),
    );
    r
}

#[test]
fn multi_vector_field_round_trips_through_segment() {
    let dir = TempDir::new().unwrap();
    let mut w = SegmentWriter::create(dir.path(), &schema()).unwrap();
    let rec_a = mk_record(7);
    let rec_b = mk_record(42);
    let id_a = rec_a.id();
    let id_b = rec_b.id();
    let tokens_a = tokens_for(7);
    w.insert(&rec_a).unwrap();
    w.insert(&rec_b).unwrap();
    let _meta = w.seal().unwrap();

    let r = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(r.doc_id_for_uuid(&id_a), Some(0));
    assert_eq!(r.doc_id_for_uuid(&id_b), Some(1));

    // Multi-vector cells come back identical to what was written.
    let rows_a = r.get_multi_vector("colpali_tokens", 0).unwrap();
    assert_eq!(rows_a.len(), tokens_a.len());
    for (got, want) in rows_a.iter().zip(&tokens_a) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }
}

#[test]
fn maxsim_query_picks_the_matching_document() {
    // Build a 4-doc corpus + a query that's the average of one
    // doc's tokens — MaxSim should rank that doc first.
    let dir = TempDir::new().unwrap();
    let mut w = SegmentWriter::create(dir.path(), &schema()).unwrap();
    let mut docs = Vec::new();
    for seed in 0..4u32 {
        let rec = mk_record(seed);
        docs.push((rec.id(), tokens_for(seed)));
        w.insert(&rec).unwrap();
    }
    let _meta = w.seal().unwrap();
    let r = SegmentReader::open(dir.path()).unwrap();

    // Query: the second-token of doc 2. With high probability MaxSim
    // (dot product) ranks doc 2 highest because its own token will
    // exactly match itself.
    let target_doc = 2usize;
    let query_tokens = vec![docs[target_doc].1[1].clone()];

    // Compute MaxSim manually for each doc — that's the contract the
    // multi_vector layer is expected to deliver via downstream code.
    let mut scores = Vec::new();
    for doc_id in 0..docs.len() as u32 {
        let doc_rows = r.get_multi_vector("colpali_tokens", doc_id).unwrap();
        let mut sum_max = 0.0f32;
        for q_tok in &query_tokens {
            let mut best = f32::NEG_INFINITY;
            for d_tok in doc_rows {
                let dot: f32 = q_tok.iter().zip(d_tok).map(|(a, b)| a * b).sum();
                if dot > best {
                    best = dot;
                }
            }
            sum_max += best;
        }
        scores.push((doc_id, sum_max));
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(
        scores[0].0 as usize, target_doc,
        "MaxSim top-1 should be doc {target_doc}; got ranking {scores:?}"
    );
}
