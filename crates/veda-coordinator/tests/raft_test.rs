//! Integration tests for the Veda coordinator.
//!
//! These tests exercise the coordinator through its async channel-based
//! interface — the same interface that callers will use in production.
//! The underlying state machine is deterministic, so results are reproducible.

use veda_coordinator::network::{Coordinator, CoordinatorHandle};
use veda_coordinator::raft_types::{ClusterCommand, ClusterResponse};

/// Helper: start a coordinator and return its handle.
fn start() -> CoordinatorHandle {
    Coordinator::start()
}

// -----------------------------------------------------------------------
// 1. Single-node leader election (coordinator starts and accepts commands)
// -----------------------------------------------------------------------

#[tokio::test]
async fn single_node_becomes_leader_and_accepts_commands() {
    let handle = start();

    // The coordinator should accept a command immediately — it is the leader.
    let resp = handle
        .execute(ClusterCommand::CreateCollection {
            name: "test_collection".into(),
            schema_json: r#"{"dim": 128}"#.into(),
        })
        .await
        .expect("coordinator should be reachable");

    assert_eq!(resp, ClusterResponse::Ok);

    // Verify state was applied.
    let state = handle.query_state().await.unwrap();
    assert!(state.collections.contains_key("test_collection"));
    assert_eq!(state.last_applied_index, 1);

    handle.shutdown().await.ok();
}

// -----------------------------------------------------------------------
// 2. Log replication — command is committed and applied
// -----------------------------------------------------------------------

#[tokio::test]
async fn log_replication_create_collection() {
    let handle = start();

    let resp = handle
        .execute(ClusterCommand::CreateCollection {
            name: "vectors".into(),
            schema_json: r#"{"dim": 256, "metric": "cosine"}"#.into(),
        })
        .await
        .unwrap();
    assert_eq!(resp, ClusterResponse::Ok);

    // Read back the state to verify the command was applied.
    let state = handle.query_state().await.unwrap();
    assert_eq!(
        state.collections.get("vectors").unwrap(),
        r#"{"dim": 256, "metric": "cosine"}"#,
    );
    assert_eq!(state.last_applied_index, 1);

    handle.shutdown().await.ok();
}

// -----------------------------------------------------------------------
// 3. State machine application — multiple command types
// -----------------------------------------------------------------------

#[tokio::test]
async fn state_machine_applies_mixed_commands() {
    let handle = start();

    // Create a collection.
    let r = handle
        .execute(ClusterCommand::CreateCollection {
            name: "images".into(),
            schema_json: r#"{"dim": 512}"#.into(),
        })
        .await
        .unwrap();
    assert_eq!(r, ClusterResponse::Ok);

    // Register two nodes.
    for (id, addr) in [(1, "10.0.0.1:9000"), (2, "10.0.0.2:9000")] {
        let r = handle
            .execute(ClusterCommand::RegisterNode {
                node_id: id,
                address: addr.into(),
                roles: vec!["data".into()],
            })
            .await
            .unwrap();
        assert_eq!(r, ClusterResponse::Ok);
    }

    // Assign shards.
    let r = handle
        .execute(ClusterCommand::AssignShard {
            collection: "images".into(),
            shard_id: 0,
            node_id: 1,
        })
        .await
        .unwrap();
    assert_eq!(r, ClusterResponse::Ok);

    let r = handle
        .execute(ClusterCommand::AssignShard {
            collection: "images".into(),
            shard_id: 1,
            node_id: 2,
        })
        .await
        .unwrap();
    assert_eq!(r, ClusterResponse::Ok);

    // Verify full state.
    let state = handle.query_state().await.unwrap();
    assert_eq!(state.collections.len(), 1);
    assert_eq!(state.nodes.len(), 2);
    assert_eq!(state.last_applied_index, 5);

    let shard0_nodes = state
        .shard_assignments
        .get("images")
        .and_then(|s| s.get(&0))
        .unwrap();
    assert_eq!(shard0_nodes, &vec![1u64]);

    let shard1_nodes = state
        .shard_assignments
        .get("images")
        .and_then(|s| s.get(&1))
        .unwrap();
    assert_eq!(shard1_nodes, &vec![2u64]);

    handle.shutdown().await.ok();
}

// -----------------------------------------------------------------------
// 4. Multiple sequential commands — ordering guarantee
// -----------------------------------------------------------------------

#[tokio::test]
async fn multiple_commands_applied_in_order() {
    let handle = start();

    // Submit 10 CreateCollection commands sequentially.
    for i in 0..10 {
        let name = format!("col_{i}");
        let resp = handle
            .execute(ClusterCommand::CreateCollection {
                name: name.clone(),
                schema_json: format!(r#"{{"id": {i}}}"#),
            })
            .await
            .unwrap();
        assert_eq!(resp, ClusterResponse::Ok, "command {i} should succeed");
    }

    let state = handle.query_state().await.unwrap();
    assert_eq!(state.collections.len(), 10);
    assert_eq!(state.last_applied_index, 10);

    // Verify each collection exists with the correct schema.
    for i in 0..10 {
        let name = format!("col_{i}");
        let schema = state.collections.get(&name).unwrap();
        assert_eq!(schema, &format!(r#"{{"id": {i}}}"#));
    }

    handle.shutdown().await.ok();
}

// -----------------------------------------------------------------------
// 5. Error handling — duplicate creation, missing entities
// -----------------------------------------------------------------------

#[tokio::test]
async fn error_handling_duplicate_and_missing() {
    let handle = start();

    // Create a collection.
    let _ = handle
        .execute(ClusterCommand::CreateCollection {
            name: "dup".into(),
            schema_json: "{}".into(),
        })
        .await
        .unwrap();

    // Duplicate create should return an error response (not a transport error).
    let resp = handle
        .execute(ClusterCommand::CreateCollection {
            name: "dup".into(),
            schema_json: "{}".into(),
        })
        .await
        .unwrap();
    assert!(matches!(resp, ClusterResponse::Error(_)));

    // Drop a non-existent collection.
    let resp = handle
        .execute(ClusterCommand::DropCollection {
            name: "no_such".into(),
        })
        .await
        .unwrap();
    assert!(matches!(resp, ClusterResponse::Error(_)));

    // Deregister a non-existent node.
    let resp = handle
        .execute(ClusterCommand::DeregisterNode { node_id: 999 })
        .await
        .unwrap();
    assert!(matches!(resp, ClusterResponse::Error(_)));

    handle.shutdown().await.ok();
}

// -----------------------------------------------------------------------
// 6. Full lifecycle — create, assign, unassign, drop
// -----------------------------------------------------------------------

#[tokio::test]
async fn full_lifecycle() {
    let handle = start();

    // Create collection + node.
    handle
        .execute(ClusterCommand::CreateCollection {
            name: "docs".into(),
            schema_json: r#"{"dim": 64}"#.into(),
        })
        .await
        .unwrap();
    handle
        .execute(ClusterCommand::RegisterNode {
            node_id: 1,
            address: "host1:8000".into(),
            roles: vec!["data".into(), "index".into()],
        })
        .await
        .unwrap();

    // Assign shard.
    handle
        .execute(ClusterCommand::AssignShard {
            collection: "docs".into(),
            shard_id: 0,
            node_id: 1,
        })
        .await
        .unwrap();

    let state = handle.query_state().await.unwrap();
    assert_eq!(
        state
            .shard_assignments
            .get("docs")
            .and_then(|s| s.get(&0))
            .unwrap(),
        &vec![1u64],
    );

    // Unassign shard.
    handle
        .execute(ClusterCommand::UnassignShard {
            collection: "docs".into(),
            shard_id: 0,
            node_id: 1,
        })
        .await
        .unwrap();

    let state = handle.query_state().await.unwrap();
    assert!(
        state
            .shard_assignments
            .get("docs")
            .and_then(|s| s.get(&0))
            .unwrap()
            .is_empty()
    );

    // Drop collection.
    handle
        .execute(ClusterCommand::DropCollection {
            name: "docs".into(),
        })
        .await
        .unwrap();

    let state = handle.query_state().await.unwrap();
    assert!(!state.collections.contains_key("docs"));
    assert!(!state.shard_assignments.contains_key("docs"));

    // Deregister node.
    handle
        .execute(ClusterCommand::DeregisterNode { node_id: 1 })
        .await
        .unwrap();

    let state = handle.query_state().await.unwrap();
    assert!(state.nodes.is_empty());

    handle.shutdown().await.ok();
}
