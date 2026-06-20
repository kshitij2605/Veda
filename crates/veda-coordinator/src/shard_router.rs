use std::sync::atomic::{AtomicU64, Ordering};

use crate::raft_types::{ClusterState, NodeStatus};
use veda_core::error::VedaError;

/// Information about where a shard lives.
#[derive(Debug, Clone)]
pub struct ShardLocation {
    pub collection: String,
    pub shard_id: u32,
    pub node_id: u64,
    pub node_address: String,
}

/// Strategy for selecting among replicas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplicaSelection {
    RoundRobin,
    Random,
    /// Always pick first active replica (for deterministic testing).
    First,
}

/// Routes queries to the appropriate shards and nodes.
pub struct ShardRouter {
    replica_selection: ReplicaSelection,
    round_robin_counter: AtomicU64,
}

impl ShardRouter {
    pub fn new(selection: ReplicaSelection) -> Self {
        Self {
            replica_selection: selection,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Get all shard locations for a collection.
    /// Returns one node per shard (selected from replicas based on strategy).
    /// Only returns active nodes.
    pub fn route_query(
        &self,
        collection: &str,
        state: &ClusterState,
    ) -> Result<Vec<ShardLocation>, VedaError> {
        let shards = state
            .shard_assignments
            .get(collection)
            .ok_or_else(|| VedaError::CollectionNotFound(collection.to_string()))?;

        let mut locations = Vec::with_capacity(shards.len());

        for (&shard_id, node_ids) in shards {
            let active: Vec<_> = node_ids
                .iter()
                .filter_map(|&nid| {
                    let info = state.nodes.get(&nid)?;
                    if info.status == NodeStatus::Active {
                        Some((nid, &info.address))
                    } else {
                        None
                    }
                })
                .collect();

            if active.is_empty() {
                return Err(VedaError::Query(format!(
                    "no active replicas for shard {shard_id} of collection '{collection}'"
                )));
            }

            let (node_id, node_address) = self.select_replica(&active);
            locations.push(ShardLocation {
                collection: collection.to_string(),
                shard_id,
                node_id,
                node_address: node_address.to_string(),
            });
        }

        Ok(locations)
    }

    /// Get all replica locations for a specific shard (for replication-aware queries).
    pub fn get_shard_replicas(
        &self,
        collection: &str,
        shard_id: u32,
        state: &ClusterState,
    ) -> Result<Vec<ShardLocation>, VedaError> {
        let shards = state
            .shard_assignments
            .get(collection)
            .ok_or_else(|| VedaError::CollectionNotFound(collection.to_string()))?;

        let node_ids = shards.get(&shard_id).ok_or_else(|| {
            VedaError::Query(format!(
                "shard {shard_id} not found in collection '{collection}'"
            ))
        })?;

        let locations = node_ids
            .iter()
            .filter_map(|&nid| {
                let info = state.nodes.get(&nid)?;
                if info.status == NodeStatus::Active {
                    Some(ShardLocation {
                        collection: collection.to_string(),
                        shard_id,
                        node_id: nid,
                        node_address: info.address.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(locations)
    }

    /// Get all shard locations for a collection (all replicas, not just selected one).
    /// Used for data operations that need to write to all replicas.
    pub fn route_write(
        &self,
        collection: &str,
        state: &ClusterState,
    ) -> Result<Vec<ShardLocation>, VedaError> {
        let shards = state
            .shard_assignments
            .get(collection)
            .ok_or_else(|| VedaError::CollectionNotFound(collection.to_string()))?;

        let mut locations = Vec::new();

        for (&shard_id, node_ids) in shards {
            for &nid in node_ids {
                if let Some(info) = state.nodes.get(&nid)
                    && info.status == NodeStatus::Active
                {
                    locations.push(ShardLocation {
                        collection: collection.to_string(),
                        shard_id,
                        node_id: nid,
                        node_address: info.address.clone(),
                    });
                }
            }
        }

        Ok(locations)
    }

    /// Determine which shard a record belongs to.
    ///
    /// Uses jump consistent hash (Lamping & Veach, 2014): only ~1/N keys
    /// move when shard_count changes from N to N+1, vs ~(N-1)/N for plain
    /// modular hashing.
    ///
    /// Returns 0 if `shard_count == 0` to avoid divide-by-zero.
    pub fn assign_shard(record_id: &uuid::Uuid, shard_count: u32) -> u32 {
        if shard_count == 0 {
            return 0;
        }
        let bytes = record_id.as_bytes();
        // First 8 bytes of the UUID as a u64 seed.
        let key = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        jump_consistent_hash(key, shard_count)
    }

    /// Select one replica from the active list based on the configured strategy.
    fn select_replica<'a>(&self, active: &[(u64, &'a String)]) -> (u64, &'a String) {
        match self.replica_selection {
            ReplicaSelection::First => active[0],
            ReplicaSelection::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                active[(idx as usize) % active.len()]
            }
            ReplicaSelection::Random => {
                // Simple pseudo-random selection using atomic counter + bit mixing.
                let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                let mut h = counter;
                h ^= h >> 33;
                h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 33;
                h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
                h ^= h >> 33;
                active[(h as usize) % active.len()]
            }
        }
    }
}

/// Jump consistent hash (Lamping & Veach 2014).
///
/// Maps a 64-bit key onto `[0, num_buckets)` such that only ~1/num_buckets of
/// keys move when num_buckets changes by 1.
fn jump_consistent_hash(mut key: u64, num_buckets: u32) -> u32 {
    let mut b: i64 = -1;
    let mut j: i64 = 0;
    let n = num_buckets as i64;
    while j < n {
        b = j;
        // LCG step (multiplier from the paper).
        key = key.wrapping_mul(2862933555777941757).wrapping_add(1);
        // Floating-point step to next bucket.
        let r = (b + 1) as f64 * ((1u64 << 31) as f64 / ((key >> 33) + 1) as f64);
        j = r as i64;
    }
    b as u32
}

#[cfg(test)]
mod jump_tests {
    use super::*;

    #[test]
    fn test_assigns_in_range() {
        for i in 0..1000u64 {
            let h = jump_consistent_hash(i, 16);
            assert!(h < 16);
        }
    }

    #[test]
    fn test_zero_buckets_returns_default() {
        // Caller guards on shard_count == 0; jump_consistent_hash itself
        // returns the last computed bucket which is -1 cast to u32. The
        // public API never invokes it with 0, but make sure the wrapper does.
        assert_eq!(ShardRouter::assign_shard(&uuid::Uuid::new_v4(), 0), 0);
    }

    #[test]
    fn test_stability_when_growing() {
        // When growing N -> N+1, most keys should stay put.
        let mut moved = 0;
        let total = 10_000;
        for i in 0..total {
            let key = i as u64;
            let before = jump_consistent_hash(key, 8);
            let after = jump_consistent_hash(key, 9);
            if before != after {
                moved += 1;
            }
        }
        // Expected fraction moved ~= 1/9 ≈ 11.1%; allow generous slack.
        let frac = moved as f64 / total as f64;
        assert!(
            (0.05..=0.20).contains(&frac),
            "moved fraction {frac} out of expected ~1/9 band"
        );
    }
}
