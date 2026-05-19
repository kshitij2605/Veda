//! Append entries, close, reopen — confirm all entries return in
//! order with monotonic seq_id.

use bytes::Bytes;
use veda_log::{LogOp, ShardLog};
use veda_time::Hlc;

fn entry_payload(i: u64) -> Bytes {
    Bytes::from(format!("{{\"i\":{i}}}").into_bytes())
}

#[test]
fn append_then_reopen_returns_all_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let n: u64 = 1_000;

    // Phase 1 — write + drop.
    {
        let log = ShardLog::open(tmp.path()).unwrap();
        for i in 1..=n {
            let seq = log
                .append_one(
                    Hlc::new(1_700_000_000_000_000_000 + i, 0),
                    0,
                    Bytes::copy_from_slice(&i.to_le_bytes()),
                    LogOp::Put,
                    entry_payload(i),
                )
                .unwrap();
            assert_eq!(seq, i);
        }
        assert_eq!(log.durable_seq(), n);
    }

    // Phase 2 — reopen and read.
    {
        let log = ShardLog::open(tmp.path()).unwrap();
        let entries = log.read_from(1).unwrap();
        assert_eq!(entries.len(), n as usize);
        for (idx, e) in entries.iter().enumerate() {
            let expected_seq = (idx as u64) + 1;
            assert_eq!(e.seq_id, expected_seq);
            assert_eq!(e.op, LogOp::Put);
            assert_eq!(e.payload, entry_payload(expected_seq));
        }
        // Next append continues monotonically.
        let next = log
            .append_one(
                Hlc::new(1, 0),
                0,
                Bytes::from_static(b"post-reopen"),
                LogOp::Put,
                Bytes::from_static(b"x"),
            )
            .unwrap();
        assert_eq!(next, n + 1);
    }
}

#[test]
fn read_from_skips_to_requested_seq() {
    let tmp = tempfile::tempdir().unwrap();
    let log = ShardLog::open(tmp.path()).unwrap();
    for i in 1u64..=10 {
        log.append_one(
            Hlc::new(1, 0),
            0,
            Bytes::copy_from_slice(&i.to_le_bytes()),
            LogOp::Put,
            entry_payload(i),
        )
        .unwrap();
    }
    let entries = log.read_from(5).unwrap();
    assert_eq!(entries.len(), 6);
    assert_eq!(entries[0].seq_id, 5);
    assert_eq!(entries.last().unwrap().seq_id, 10);
}

#[test]
fn empty_log_reopen_is_clean() {
    let tmp = tempfile::tempdir().unwrap();
    let log = ShardLog::open(tmp.path()).unwrap();
    assert_eq!(log.durable_seq(), 0);
    assert_eq!(log.next_seq(), 1);
    let entries = log.read_from(1).unwrap();
    assert!(entries.is_empty());
}
