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

    /// Determine which shard a record belongs to based on a hash of its ID.
    /// Uses consistent hashing: shard_id = hash(record_id) % shard_count
    pub fn assign_shard(record_id: &uuid::Uuid, shard_count: u32) -> u32 {
        let bytes = record_id.as_bytes();
        // Use first 8 bytes as a u64 for hashing.
        let hash = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        (hash % shard_count as u64) as u32
    }

    /// Select one replica from the active list based on the configured strategy.
    fn select_replica<'a>(&self, active: &[( u64, &'a String)]) -> (u64, &'a String) {
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
