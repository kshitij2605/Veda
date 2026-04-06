use veda_core::simd;
use veda_core::types::DistanceMetric;

/// Compute the distance between two vectors using the given metric.
///
/// For dot product the raw score is negated so that, regardless of metric,
/// **lower values indicate greater similarity**.
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => simd::cosine_distance(a, b),
        DistanceMetric::Euclidean => simd::euclidean_distance(a, b),
        DistanceMetric::DotProduct => -simd::dot_product(a, b),
        DistanceMetric::Manhattan => simd::manhattan_distance(a, b),
        _ => panic!("Unsupported metric for dense vectors: {:?}", metric),
    }
}
