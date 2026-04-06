use std::collections::HashMap;

use uuid::Uuid;

use veda_coordinator::raft_types::{ClusterState, NodeInfo, NodeStatus};
use veda_coordinator::shard_router::{ReplicaSelection, ShardRouter};

/// Build a ClusterState with the given collections, nodes, and shard assignments.
fn make_state(
    collections: &[(&str, &str)],
    nodes: &[(u64, &str, NodeStatus)],
    assignments: &[(&str, u32, &[u64])],
) -> ClusterState {
    let mut state = ClusterState::default();

    for &(name, schema) in collections {
        state
            .collections
            .insert(name.to_string(), schema.to_string());
    }

    for &(id, addr, ref status) in nodes {
        state.nodes.insert(
            id,
            NodeInfo {
                address: addr.to_string(),
                roles: vec!["data".into()],
                status: status.clone(),
            },
        );
    }

    for &(coll, shard_id, node_ids) in assignments {
        state
            .shard_assignments
            .entry(coll.to_string())
            .or_default()
            .insert(shard_id, node_ids.to_vec());
    }

    state
}

// ---------------------------------------------------------------------------
// 1. Basic routing
// ---------------------------------------------------------------------------

#[test]
fn basic_routing_returns_one_location_per_shard() {
    let state = make_state(
        &[("vectors", r#"{"dim": 128}"#)],
        &[
            (1, "10.0.0.1:9000", NodeStatus::Active),
            (2, "10.0.0.2:9000", NodeStatus::Active),
        ],
        &[
            ("vectors", 0, &[1]),
            ("vectors", 1, &[2]),
            ("vectors", 2, &[1]),
            ("vectors", 3, &[2]),
        ],
    );

    let router = ShardRouter::new(ReplicaSelection::First);
    let locations = router.route_query("vectors", &state).unwrap();

    assert_eq!(locations.len(), 4);

    // Each shard should appear exactly once.
    let mut shard_ids: Vec<u32> = locations.iter().map(|l| l.shard_id).collect();
    shard_ids.sort();
    assert_eq!(shard_ids, vec![0, 1, 2, 3]);

    // All locations should reference the correct collection.
    for loc in &locations {
        assert_eq!(loc.collection, "vectors");
    }
}

// ---------------------------------------------------------------------------
// 2. Replica selection — RoundRobin
// ---------------------------------------------------------------------------

#[test]
fn round_robin_cycles_through_replicas() {
    let state = make_state(
        &[("col", "{}")],
        &[
            (1, "n1:9000", NodeStatus::Active),
            (2, "n2:9000", NodeStatus::Active),
            (3, "n3:9000", NodeStatus::Active),
        ],
        &[("col", 0, &[1, 2, 3])],
    );

    let router = ShardRouter::new(ReplicaSelection::RoundRobin);

    let mut selected = Vec::new();
    for _ in 0..6 {
        let locs = router.route_query("col", &state).unwrap();
        assert_eq!(locs.len(), 1);
        selected.push(locs[0].node_id);
    }

    // Should cycle through 1, 2, 3, 1, 2, 3.
    assert_eq!(selected, vec![1, 2, 3, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// 3. Replica selection — Random
// ---------------------------------------------------------------------------

#[test]
fn random_selects_all_replicas_over_many_iterations() {
    let state = make_state(
        &[("col", "{}")],
        &[
            (1, "n1:9000", NodeStatus::Active),
            (2, "n2:9000", NodeStatus::Active),
            (3, "n3:9000", NodeStatus::Active),
        ],
        &[("col", 0, &[1, 2, 3])],
    );

    let router = ShardRouter::new(ReplicaSelection::Random);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..100 {
        let locs = router.route_query("col", &state).unwrap();
        seen.insert(locs[0].node_id);
    }

    // All 3 nodes should be selected at least once.
    assert!(
        seen.contains(&1) && seen.contains(&2) && seen.contains(&3),
        "Expected all 3 nodes to be selected, got: {seen:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. Active node filtering
// ---------------------------------------------------------------------------

#[test]
fn filters_out_draining_and_inactive_nodes() {
    let state = make_state(
        &[("col", "{}")],
        &[
            (1, "n1:9000", NodeStatus::Active),
            (2, "n2:9000", NodeStatus::Draining),
            (3, "n3:9000", NodeStatus::Inactive),
        ],
        &[("col", 0, &[1, 2, 3])],
    );

    let router = ShardRouter::new(ReplicaSelection::First);
    let locs = router.route_query("col", &state).unwrap();

    assert_eq!(locs.len(), 1);
    assert_eq!(locs[0].node_id, 1);
    assert_eq!(locs[0].node_address, "n1:9000");
}

// ---------------------------------------------------------------------------
// 5. Write routing returns all active replicas
// ---------------------------------------------------------------------------

#[test]
fn write_routing_returns_all_active_replicas() {
    let state = make_state(
        &[("col", "{}")],
        &[
            (1, "n1:9000", NodeStatus::Active),
            (2, "n2:9000", NodeStatus::Active),
            (3, "n3:9000", NodeStatus::Inactive),
        ],
        &[
            ("col", 0, &[1, 2, 3]),
            ("col", 1, &[1, 2]),
        ],
    );

    let router = ShardRouter::new(ReplicaSelection::First);
    let locs = router.route_write("col", &state).unwrap();

    // Shard 0: nodes 1, 2 active (3 is inactive). Shard 1: nodes 1, 2 active.
    // Total: 4 locations.
    assert_eq!(locs.len(), 4);

    let shard0_nodes: Vec<u64> = locs
        .iter()
        .filter(|l| l.shard_id == 0)
        .map(|l| l.node_id)
        .collect();
    assert_eq!(shard0_nodes.len(), 2);
    assert!(shard0_nodes.contains(&1));
    assert!(shard0_nodes.contains(&2));
}

// ---------------------------------------------------------------------------
// 6. Shard assignment — roughly uniform distribution
// ---------------------------------------------------------------------------

#[test]
fn shard_assignment_distributes_roughly_uniformly() {
    let shard_count = 4;
    let mut counts = HashMap::new();

    for _ in 0..1000 {
        let id = Uuid::new_v4();
        let shard = ShardRouter::assign_shard(&id, shard_count);
        assert!(shard < shard_count);
        *counts.entry(shard).or_insert(0u32) += 1;
    }

    // Each shard should get at least some records (expect ~250 each).
    for shard in 0..shard_count {
        let count = counts.get(&shard).copied().unwrap_or(0);
        assert!(
            count > 100,
            "shard {shard} got only {count} records out of 1000 — distribution is too skewed"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Collection not found
// ---------------------------------------------------------------------------

#[test]
fn route_query_errors_on_missing_collection() {
    let state = ClusterState::default();
    let router = ShardRouter::new(ReplicaSelection::First);

    let err = router.route_query("nonexistent", &state).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent"),
        "error should mention the collection name: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 8. No active replicas
// ---------------------------------------------------------------------------

#[test]
fn errors_when_no_active_replicas_for_shard() {
    let state = make_state(
        &[("col", "{}")],
        &[
            (1, "n1:9000", NodeStatus::Inactive),
            (2, "n2:9000", NodeStatus::Inactive),
        ],
        &[("col", 0, &[1, 2])],
    );

    let router = ShardRouter::new(ReplicaSelection::First);
    let err = router.route_query("col", &state).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no active replicas"),
        "error should mention no active replicas: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 9. Multiple collections routed independently
// ---------------------------------------------------------------------------

#[test]
fn multiple_collections_routed_independently() {
    let state = make_state(
        &[("images", r#"{"dim": 512}"#), ("text", r#"{"dim": 768}"#)],
        &[
            (1, "n1:9000", NodeStatus::Active),
            (2, "n2:9000", NodeStatus::Active),
        ],
        &[
            ("images", 0, &[1]),
            ("images", 1, &[2]),
            ("text", 0, &[2]),
            ("text", 1, &[1]),
        ],
    );

    let router = ShardRouter::new(ReplicaSelection::First);

    let image_locs = router.route_query("images", &state).unwrap();
    let text_locs = router.route_query("text", &state).unwrap();

    assert_eq!(image_locs.len(), 2);
    assert_eq!(text_locs.len(), 2);

    // images shard 0 -> node 1, text shard 0 -> node 2
    let img_s0 = image_locs.iter().find(|l| l.shard_id == 0).unwrap();
    assert_eq!(img_s0.node_id, 1);

    let txt_s0 = text_locs.iter().find(|l| l.shard_id == 0).unwrap();
    assert_eq!(txt_s0.node_id, 2);
}

// ---------------------------------------------------------------------------
// 10. Shard assignment determinism
// ---------------------------------------------------------------------------

#[test]
fn shard_assignment_is_deterministic() {
    let id = Uuid::new_v4();
    let shard_count = 8;

    let first = ShardRouter::assign_shard(&id, shard_count);

    for _ in 0..100 {
        assert_eq!(ShardRouter::assign_shard(&id, shard_count), first);
    }
}
