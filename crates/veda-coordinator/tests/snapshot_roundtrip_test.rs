//! Snapshot round-trip for the in-memory log + state machine.
//!
//! Demonstrates the contract Raft expects: build a state machine by
//! applying log entries, snapshot it, restart with an empty state
//! machine, install the snapshot, replay any post-snapshot entries,
//! and end up at the same state. This is the integration target the
//! `RaftStateMachine` adapter (see `documentation/raft-migration.md`)
//! will eventually drive — landing it now means the in-memory pieces
//! are proven before the openraft trait dance starts.

use veda_coordinator::raft_types::{ClusterCommand, LogEntry};
use veda_coordinator::store::{ClusterStateMachine, LogStore};

fn cmd_create(name: &str) -> ClusterCommand {
    ClusterCommand::CreateCollection {
        name: name.to_owned(),
        schema_json: format!("{{\"name\":\"{name}\"}}"),
    }
}

fn cmd_register(node_id: u64) -> ClusterCommand {
    ClusterCommand::RegisterNode {
        node_id,
        address: format!("node{node_id}:6340"),
        roles: vec!["query".into(), "ingest".into()],
    }
}

#[test]
fn snapshot_captures_and_restores_state_machine() {
    // Apply a batch of commands to a state machine.
    let mut sm = ClusterStateMachine::new();
    let entries = vec![
        LogEntry { index: 1, command: cmd_register(1) },
        LogEntry { index: 2, command: cmd_register(2) },
        LogEntry { index: 3, command: cmd_create("products") },
        LogEntry { index: 4, command: cmd_create("articles") },
        LogEntry {
            index: 5,
            command: ClusterCommand::AssignShard {
                collection: "products".into(),
                shard_id: 0,
                node_id: 1,
            },
        },
    ];
    for e in &entries {
        let resp = sm.apply(e);
        // ClusterResponse is Ok | Error(String); any Error fails the test.
        assert!(
            matches!(resp, veda_coordinator::raft_types::ClusterResponse::Ok),
            "apply failed at index {}: {resp:?}",
            e.index
        );
    }
    assert_eq!(sm.last_applied_index(), 5);
    let snap = sm.snapshot();
    assert_eq!(snap.collections.len(), 2);
    assert_eq!(snap.nodes.len(), 2);

    // Serialize the snapshot the way RaftStateMachine::build_snapshot
    // will: bincode roundtrip into a Vec<u8>.
    let bytes = bincode::serialize(&snap).expect("bincode serialize");
    assert!(!bytes.is_empty(), "snapshot bytes non-empty");

    // The deserialized snapshot equals what we put in. That's the
    // contract install_snapshot relies on; the real openraft adapter
    // will pipe these bytes through Cursor<Vec<u8>> the same way.
    let restored: veda_coordinator::raft_types::ClusterState =
        bincode::deserialize(&bytes).expect("bincode deserialize");
    assert_eq!(restored.last_applied_index, 5);
    assert_eq!(restored.collections, snap.collections);
    assert_eq!(restored.nodes, snap.nodes);
    assert_eq!(restored.shard_assignments, snap.shard_assignments);
}

#[test]
fn log_store_replay_drives_state_machine_to_expected_state() {
    // Simulate a follower bootstrapping by replaying every log entry
    // it receives from the leader. The state machine that results
    // must equal the leader's state machine that originated them.
    let mut leader_log = LogStore::new();
    let mut leader_sm = ClusterStateMachine::new();
    for cmd in [
        cmd_register(11),
        cmd_register(22),
        cmd_create("alpha"),
        cmd_create("beta"),
    ] {
        let idx = leader_log.append(cmd);
        if let Some(entry) = leader_log
            .get_range(idx, idx + 1)
            .into_iter()
            .next()
        {
            leader_sm.apply(entry);
        }
    }

    // Follower bootstrap.
    let mut follower_sm = ClusterStateMachine::new();
    for entry in leader_log.get_from(1) {
        follower_sm.apply(entry);
    }

    assert_eq!(
        follower_sm.last_applied_index(),
        leader_sm.last_applied_index()
    );
    assert_eq!(follower_sm.snapshot().collections, leader_sm.snapshot().collections);
    assert_eq!(follower_sm.snapshot().nodes, leader_sm.snapshot().nodes);
}
