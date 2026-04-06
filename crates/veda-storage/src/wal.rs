use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use veda_core::error::Result;
use veda_core::record::Record;

/// A WAL entry representing a write operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    Insert {
        collection: String,
        shard_id: u32,
        record: Record,
    },
    Delete {
        collection: String,
        shard_id: u32,
        record_id: Uuid,
    },
}

/// Trait for WAL writers (producers).
pub trait WalWriter: Send + Sync {
    /// Append an entry to the WAL. Returns the sequence number.
    fn append(&self, entry: WalEntry) -> Result<u64>;
}

/// Trait for WAL readers (consumers).
pub trait WalReader: Send + Sync {
    /// Read entries from the WAL starting at the given sequence number.
    /// Returns entries and their sequence numbers.
    fn read_from(&self, seq: u64, max_entries: usize) -> Result<Vec<(u64, WalEntry)>>;

    /// Get the latest sequence number in the WAL.
    fn latest_seq(&self) -> Result<u64>;

    /// Acknowledge that entries up to the given sequence have been processed.
    /// Allows the WAL to purge old entries.
    fn acknowledge(&self, seq: u64) -> Result<()>;
}

/// In-memory WAL for testing and standalone mode.
pub struct InMemoryWal {
    entries: RwLock<Vec<(u64, WalEntry)>>,
    next_seq: AtomicU64,
    ack_seq: AtomicU64,
}

impl InMemoryWal {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            next_seq: AtomicU64::new(1),
            ack_seq: AtomicU64::new(0),
        }
    }

    pub fn entry_count(&self) -> usize {
        self.entries.read().unwrap().len()
    }
}

impl Default for InMemoryWal {
    fn default() -> Self {
        Self::new()
    }
}

impl WalWriter for InMemoryWal {
    fn append(&self, entry: WalEntry) -> Result<u64> {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let mut entries = self.entries.write().unwrap();
        entries.push((seq, entry));
        Ok(seq)
    }
}

impl WalReader for InMemoryWal {
    fn read_from(&self, seq: u64, max_entries: usize) -> Result<Vec<(u64, WalEntry)>> {
        let entries = self.entries.read().unwrap();
        let result: Vec<(u64, WalEntry)> = entries
            .iter()
            .filter(|(s, _)| *s >= seq)
            .take(max_entries)
            .cloned()
            .collect();
        Ok(result)
    }

    fn latest_seq(&self) -> Result<u64> {
        let current = self.next_seq.load(Ordering::SeqCst);
        if current <= 1 { Ok(0) } else { Ok(current - 1) }
    }

    fn acknowledge(&self, seq: u64) -> Result<()> {
        self.ack_seq.store(seq, Ordering::SeqCst);
        let mut entries = self.entries.write().unwrap();
        entries.retain(|(s, _)| *s > seq);
        Ok(())
    }
}

/// Placeholder for NATS JetStream WAL (to be implemented when NATS is available).
/// Implements the same WalWriter/WalReader traits.
pub struct NatsWal {
    // Will hold async-nats client, stream name, etc.
    _placeholder: (),
}
