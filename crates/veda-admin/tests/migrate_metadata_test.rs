//! End-to-end: build a v0-shaped raft_state.bin blob by driving
//! a `ClusterStateMachine` through CreateCollection +
//! InsertRecord + BeginTxn, then run the migration helper +
//! the actual CLI binary against it.

use std::process::Command;

use veda_coordinator::raft_storage::{build_persisted_blob_for_state, migrate_persisted_blob};
use veda_coordinator::raft_types::{ClusterCommand, ClusterResponse, LogEntry};
use veda_coordinator::store::ClusterStateMachine;

fn apply(sm: &mut ClusterStateMachine, index: u64, command: ClusterCommand) {
    let entry = LogEntry { index, command };
    let resp = sm.apply(&entry);
    assert!(matches!(resp, ClusterResponse::Ok), "apply: {resp:?}");
}

fn v0_blob() -> Vec<u8> {
    let mut sm = ClusterStateMachine::new();
    apply(
        &mut sm,
        1,
        ClusterCommand::CreateCollection {
            name: "docs".into(),
            schema_json: "{}".into(),
        },
    );
    apply(
        &mut sm,
        2,
        ClusterCommand::InsertRecord {
            collection: "docs".into(),
            record_id: "r1".into(),
            record_json: "{\"k\":\"v\"}".into(),
        },
    );
    apply(
        &mut sm,
        3,
        ClusterCommand::InsertRecord {
            collection: "docs".into(),
            record_id: "r2".into(),
            record_json: "{\"k\":\"v2\"}".into(),
        },
    );
    apply(
        &mut sm,
        4,
        ClusterCommand::BeginTxn {
            txn_id: "T1".into(),
        },
    );
    build_persisted_blob_for_state(sm).unwrap()
}

#[test]
fn migrate_drops_records_and_open_txns_from_blob() {
    let v0 = v0_blob();
    let (v1, recs, txns) = migrate_persisted_blob(&v0).unwrap();
    assert_eq!(recs, 2, "expected 2 records dropped, got {recs}");
    assert_eq!(txns, 1, "expected 1 open_txn dropped, got {txns}");
    // Migration is idempotent.
    let (_, recs2, txns2) = migrate_persisted_blob(&v1).unwrap();
    assert_eq!(recs2, 0);
    assert_eq!(txns2, 0);
}

#[test]
fn migrate_rejects_garbage_blob() {
    let err = migrate_persisted_blob(b"not a valid raft_state.bin").unwrap_err();
    assert!(err.contains("decode"), "expected decode err, got: {err}");
}

/// Drive the actual CLI binary against a tempdir with a real v0
/// blob on disk.
#[test]
fn cli_migrates_real_blob_on_disk() {
    let bin = env!("CARGO_BIN_EXE_veda-admin");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raft_state.bin");
    std::fs::write(&path, v0_blob()).unwrap();

    // Dry run reports counts without rewriting.
    let v0_size = std::fs::metadata(&path).unwrap().len();
    let out = Command::new(bin)
        .args(["migrate-metadata", "--dir"])
        .arg(dir.path())
        .arg("--dry-run")
        .output()
        .expect("spawn veda-admin");
    assert!(out.status.success(), "dry-run failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("would drop 2 record"));
    assert!(stdout.contains("1 open txn"));
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        v0_size,
        "dry-run must not modify the blob"
    );

    // Real run rewrites + leaves a backup.
    let out = Command::new(bin)
        .args(["migrate-metadata", "--dir"])
        .arg(dir.path())
        .output()
        .expect("spawn veda-admin");
    assert!(out.status.success(), "migrate failed: {}", String::from_utf8_lossy(&out.stderr));
    // A timestamped backup must exist.
    let mut found_backup = false;
    for e in std::fs::read_dir(dir.path()).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        if name.starts_with("raft_state.bin.bak.") {
            found_backup = true;
        }
    }
    assert!(found_backup, "no .bak.<ts> backup left behind");
    // Re-run is a no-op (counts come back as 0).
    let out = Command::new(bin)
        .args(["migrate-metadata", "--dir"])
        .arg(dir.path())
        .arg("--no-backup")
        .output()
        .expect("spawn veda-admin");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("would drop 0 record"), "stdout: {stdout}");
}

/// Missing file should produce a clear error.
#[test]
fn cli_errors_on_missing_state_bin() {
    let bin = env!("CARGO_BIN_EXE_veda-admin");
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(bin)
        .args(["migrate-metadata", "--dir"])
        .arg(dir.path())
        .arg("--dry-run")
        .output()
        .expect("spawn veda-admin");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("raft_state.bin"),
        "expected error mentioning raft_state.bin, got: {stderr}"
    );
}
