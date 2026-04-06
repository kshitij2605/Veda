use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use uuid::Uuid;
use veda_core::record::{FieldValue, Record};
use veda_storage::wal::{InMemoryWal, WalEntry, WalReader, WalWriter};

fn make_insert_entry(collection: &str, shard_id: u32) -> WalEntry {
    let mut record = Record::new();
    record.set("name", FieldValue::String("test".to_owned()));
    WalEntry::Insert {
        collection: collection.to_owned(),
        shard_id,
        record,
    }
}

fn make_delete_entry(collection: &str, shard_id: u32) -> WalEntry {
    WalEntry::Delete {
        collection: collection.to_owned(),
        shard_id,
        record_id: Uuid::new_v4(),
    }
}

#[test]
fn append_and_read() {
    let wal = InMemoryWal::new();
    for i in 0..10 {
        wal.append(make_insert_entry("col", i)).unwrap();
    }
    let entries = wal.read_from(1, 10).unwrap();
    assert_eq!(entries.len(), 10);
}

#[test]
fn sequence_numbering() {
    let wal = InMemoryWal::new();
    for _ in 0..5 {
        wal.append(make_insert_entry("col", 0)).unwrap();
    }
    let entries = wal.read_from(1, 10).unwrap();
    let seqs: Vec<u64> = entries.iter().map(|(s, _)| *s).collect();
    assert_eq!(seqs, vec![1, 2, 3, 4, 5]);
}

#[test]
fn read_from_offset() {
    let wal = InMemoryWal::new();
    for i in 0..20 {
        wal.append(make_insert_entry("col", i)).unwrap();
    }
    let entries = wal.read_from(11, 10).unwrap();
    assert_eq!(entries.len(), 10);
    assert_eq!(entries[0].0, 11);
    assert_eq!(entries[9].0, 20);
}

#[test]
fn max_entries_limit() {
    let wal = InMemoryWal::new();
    for i in 0..100 {
        wal.append(make_insert_entry("col", i)).unwrap();
    }
    let entries = wal.read_from(1, 5).unwrap();
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[4].0, 5);
}

#[test]
fn acknowledge_and_purge() {
    let wal = InMemoryWal::new();
    for i in 0..10 {
        wal.append(make_insert_entry("col", i)).unwrap();
    }
    wal.acknowledge(5).unwrap();
    assert_eq!(wal.entry_count(), 5);
    // Only entries 6-10 remain
    let entries = wal.read_from(1, 10).unwrap();
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0].0, 6);
}

#[test]
fn latest_sequence() {
    let wal = InMemoryWal::new();
    assert_eq!(wal.latest_seq().unwrap(), 0);
    wal.append(make_insert_entry("col", 0)).unwrap();
    assert_eq!(wal.latest_seq().unwrap(), 1);
    wal.append(make_insert_entry("col", 0)).unwrap();
    assert_eq!(wal.latest_seq().unwrap(), 2);
    for _ in 0..8 {
        wal.append(make_insert_entry("col", 0)).unwrap();
    }
    assert_eq!(wal.latest_seq().unwrap(), 10);
}

#[test]
fn empty_wal() {
    let wal = InMemoryWal::new();
    let entries = wal.read_from(1, 10).unwrap();
    assert!(entries.is_empty());
    assert_eq!(wal.latest_seq().unwrap(), 0);
}

#[test]
fn insert_entry_serde_round_trip() {
    let mut record = Record::new();
    record.set("title", FieldValue::String("hello".to_owned()));
    record.set("score", FieldValue::Float64(0.95));
    let entry = WalEntry::Insert {
        collection: "docs".to_owned(),
        shard_id: 3,
        record,
    };
    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: WalEntry = serde_json::from_str(&json).unwrap();
    match deserialized {
        WalEntry::Insert {
            collection,
            shard_id,
            record,
        } => {
            assert_eq!(collection, "docs");
            assert_eq!(shard_id, 3);
            assert_eq!(
                record.get("title"),
                Some(&FieldValue::String("hello".to_owned()))
            );
            assert_eq!(record.get("score"), Some(&FieldValue::Float64(0.95)));
        }
        _ => panic!("expected Insert variant"),
    }
}

#[test]
fn delete_entry_serde_round_trip() {
    let entry = make_delete_entry("images", 7);
    let id = match &entry {
        WalEntry::Delete { record_id, .. } => *record_id,
        _ => panic!("expected Delete variant"),
    };
    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: WalEntry = serde_json::from_str(&json).unwrap();
    match deserialized {
        WalEntry::Delete {
            collection,
            shard_id,
            record_id,
        } => {
            assert_eq!(collection, "images");
            assert_eq!(shard_id, 7);
            assert_eq!(record_id, id);
        }
        _ => panic!("expected Delete variant"),
    }
}

#[test]
fn concurrent_access() {
    let wal = Arc::new(InMemoryWal::new());
    let num_threads = 8;
    let entries_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let wal = Arc::clone(&wal);
            thread::spawn(move || {
                let mut seqs = Vec::with_capacity(entries_per_thread);
                for i in 0..entries_per_thread {
                    let seq = wal
                        .append(make_insert_entry("col", (t * 100 + i) as u32))
                        .unwrap();
                    seqs.push(seq);
                }
                seqs
            })
        })
        .collect();

    let mut all_seqs = Vec::new();
    for h in handles {
        all_seqs.extend(h.join().unwrap());
    }

    assert_eq!(wal.entry_count(), num_threads * entries_per_thread);
    // All sequence numbers are unique
    let unique: HashSet<u64> = all_seqs.iter().copied().collect();
    assert_eq!(unique.len(), num_threads * entries_per_thread);
}
