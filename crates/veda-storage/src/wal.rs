use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// NATS JetStream-backed WAL. Async-only — uses `async-nats` directly rather
/// than implementing the sync `WalWriter`/`WalReader` traits, which would
/// require blocking inside async runtime.
///
/// Stream layout:
///   - One JetStream stream per Veda instance (default name `"veda-wal"`).
///   - Subjects: `veda.wal.{collection}.{shard}` — collection/shard-scoped.
///   - WalEntry bodies serialised via bincode for compactness.
///
/// NATS' own message `sequence` doubles as our `seq` — no separate counter.
pub struct NatsWal {
    client: async_nats::jetstream::Context,
    stream_name: String,
}

impl NatsWal {
    /// Connect to NATS and ensure the WAL stream exists.
    ///
    /// `url` example: `"nats://nats:4222"`. `stream_name` example: `"veda-wal"`.
    pub async fn connect(url: &str, stream_name: impl Into<String>) -> Result<Self> {
        let client = async_nats::connect(url).await.map_err(|e| {
            veda_core::error::VedaError::Storage(format!("nats connect failed: {e}"))
        })?;
        let js = async_nats::jetstream::new(client);
        let stream_name = stream_name.into();
        js.get_or_create_stream(async_nats::jetstream::stream::Config {
            name: stream_name.clone(),
            subjects: vec![format!("veda.wal.>")],
            retention: async_nats::jetstream::stream::RetentionPolicy::Limits,
            ..Default::default()
        })
        .await
        .map_err(|e| {
            veda_core::error::VedaError::Storage(format!("nats stream create failed: {e}"))
        })?;
        Ok(Self {
            client: js,
            stream_name,
        })
    }

    fn subject_for(entry: &WalEntry) -> String {
        match entry {
            WalEntry::Insert {
                collection,
                shard_id,
                ..
            }
            | WalEntry::Delete {
                collection,
                shard_id,
                ..
            } => format!("veda.wal.{collection}.{shard_id}"),
        }
    }

    /// Publish a WAL entry. Returns the NATS-assigned stream sequence.
    pub async fn append(&self, entry: WalEntry) -> Result<u64> {
        let subject = Self::subject_for(&entry);
        let payload = bincode::serialize(&entry)
            .map_err(|e| veda_core::error::VedaError::Storage(format!("wal serialise: {e}")))?;
        let ack = self
            .client
            .publish(subject, payload.into())
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats publish failed: {e}")))?
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats ack failed: {e}")))?;
        Ok(ack.sequence)
    }

    /// Read up to `max_entries` entries with sequence >= `seq`. Drains a
    /// short-lived ephemeral consumer so callers can use this for recovery
    /// or replay.
    pub async fn read_from(&self, seq: u64, max_entries: usize) -> Result<Vec<(u64, WalEntry)>> {
        use futures::StreamExt;

        let stream = self
            .client
            .get_stream(&self.stream_name)
            .await
            .map_err(|e| {
                veda_core::error::VedaError::Storage(format!("nats stream lookup: {e}"))
            })?;

        let consumer = stream
            .create_consumer(async_nats::jetstream::consumer::pull::Config {
                deliver_policy: async_nats::jetstream::consumer::DeliverPolicy::ByStartSequence {
                    start_sequence: seq.max(1),
                },
                ..Default::default()
            })
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats consumer: {e}")))?;

        let mut messages = consumer
            .fetch()
            .max_messages(max_entries)
            .messages()
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats fetch: {e}")))?;

        let mut out = Vec::with_capacity(max_entries);
        while let Some(m) = messages.next().await {
            let m =
                m.map_err(|e| veda_core::error::VedaError::Storage(format!("nats msg: {e}")))?;
            let info = m
                .info()
                .map_err(|e| veda_core::error::VedaError::Storage(format!("nats info: {e}")))?;
            let entry: WalEntry = bincode::deserialize(&m.payload).map_err(|e| {
                veda_core::error::VedaError::Storage(format!("wal deserialise: {e}"))
            })?;
            out.push((info.stream_sequence, entry));
            m.ack().await.ok();
        }
        Ok(out)
    }

    /// Stream's current last sequence.
    pub async fn latest_seq(&self) -> Result<u64> {
        let mut stream = self
            .client
            .get_stream(&self.stream_name)
            .await
            .map_err(|e| {
                veda_core::error::VedaError::Storage(format!("nats stream lookup: {e}"))
            })?;
        let info = stream
            .info()
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats stream info: {e}")))?;
        Ok(info.state.last_sequence)
    }

    /// Purge messages with sequence <= `seq`. Bounds memory for long-running streams.
    pub async fn acknowledge(&self, seq: u64) -> Result<()> {
        let stream = self
            .client
            .get_stream(&self.stream_name)
            .await
            .map_err(|e| {
                veda_core::error::VedaError::Storage(format!("nats stream lookup: {e}"))
            })?;
        stream
            .purge()
            .sequence(seq + 1)
            .await
            .map_err(|e| veda_core::error::VedaError::Storage(format!("nats purge: {e}")))?;
        Ok(())
    }
}
