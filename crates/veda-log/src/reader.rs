//! Log tailer — yields ordered, gapless entries from a `ShardLog`.
//!
//! Phase B ships a polling tailer: caller drives `next_batch`
//! periodically. The async-stream variant (which woke on durable_seq
//! advances) lands when the LogApplier (Phase B follow-up / Phase C)
//! actually consumes from a tailer.

use std::sync::Arc;
use std::time::Duration;

use crate::format::LogEntryV2;
use crate::shard_log::{ShardLog, ShardLogError};

#[derive(Debug)]
pub struct LogTailer {
    log: Arc<ShardLog>,
    last_seen: u64,
}

impl LogTailer {
    /// Construct a tailer starting at `from` (the first seq_id the
    /// next read will include).
    pub fn new(log: Arc<ShardLog>, from: u64) -> Self {
        Self {
            log,
            last_seen: from.saturating_sub(1),
        }
    }

    /// Return all entries newer than the last batch returned.
    /// Returns an empty Vec if no new entries are durable yet.
    pub fn next_batch(&mut self) -> Result<Vec<LogEntryV2>, ShardLogError> {
        let batch = self.log.read_from(self.last_seen + 1)?;
        if let Some(last) = batch.last() {
            self.last_seen = last.seq_id;
        }
        Ok(batch)
    }

    /// Highest seq_id the tailer has acknowledged consuming.
    pub fn last_seen(&self) -> u64 {
        self.last_seen
    }

    /// Lag in seq_ids vs the log's durable head. Useful for
    /// metrics + back-pressure decisions in Phase C.
    pub fn lag(&self) -> u64 {
        self.log.durable_seq().saturating_sub(self.last_seen)
    }

    /// Block (via tokio sleep) until the tailer has caught up to
    /// `target` or the deadline elapses. Polling implementation;
    /// the stream-based variant arrives when LogApplier lands.
    pub async fn wait_for(
        &mut self,
        target: u64,
        timeout: Duration,
    ) -> Result<bool, ShardLogError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let _ = self.next_batch()?;
            if self.last_seen >= target {
                return Ok(true);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(false);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}
