//! Per-entry append-only WAL for the Raft log.
//!
//! Replaces the "rewrite the entire log on every mutation"
//! approach in [`crate::raft_storage::MemLogState`]. The vote and
//! the state machine snapshot stay as atomic-rewrite files (they're
//! small and infrequently changed); only the log proper switches
//! to a true append log.
//!
//! ## File format
//!
//! `{dir}/raft_log.wal` is a sequence of records. Each record:
//!
//! ```text
//! [u32 LE crc32 of payload][u32 LE payload length][payload]
//! ```
//!
//! where the payload is a bincode-serialised [`WalRecord`]. Read
//! recovery scans the file in order; if the last record is torn
//! (less than `length` bytes followed) or CRC-bad (mid-write
//! corruption), the recovery truncates the file at the last
//! valid record boundary and continues. This handles
//! kill-during-fsync without manual intervention.
//!
//! ## Records
//!
//! - `Entry(Entry<VedaTypeConfig>)` — a new log entry, appended
//!   at the leader and replicated. The vast majority of records
//!   are these.
//! - `Truncate(LogId)` — drop all entries with `index >= LogId.index`.
//!   Used by Raft conflict resolution.
//! - `Purge(LogId)` — drop all entries with `index <= LogId.index`.
//!   Used after a snapshot install.
//! - `Committed(LogId)` — record the latest committed log id. Not
//!   strictly necessary for correctness (committed can be
//!   re-derived) but lets recovery skip computing it.
//!
//! ## Compaction
//!
//! The WAL is never rewritten in-place. Over time it grows even
//! when entries get truncated/purged, because the markers stay
//! on disk. A compaction pass that rewrites the file with only
//! the surviving entries is a follow-up; for the per-entry write
//! amplification fix, that work is not required yet.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openraft::{Entry, LogId, Vote};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::raft::{NodeId, VedaTypeConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    Entry(Entry<VedaTypeConfig>),
    Truncate(LogId<NodeId>),
    Purge(LogId<NodeId>),
    Committed(LogId<NodeId>),
    Vote(Vote<NodeId>),
}

/// Append-only WAL. Cheap to clone; the underlying file handle is
/// shared via Arc<Mutex<_>>.
#[derive(Debug, Clone)]
pub struct WalLog {
    inner: Arc<Mutex<WalInner>>,
}

#[derive(Debug)]
struct WalInner {
    file: File,
    path: PathBuf,
    /// Track how many bytes have been written for diagnostics
    /// (handy in tests). Not used for correctness.
    bytes_written: u64,
}

impl WalLog {
    /// Open or create the WAL at `{dir}/raft_log.wal`.
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("raft_log.wal");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let bytes_written = file.metadata()?.len();
        Ok(Self {
            inner: Arc::new(Mutex::new(WalInner {
                file,
                path,
                bytes_written,
            })),
        })
    }

    /// Append a record, fsyncing before returning. Errors fail the
    /// caller — losing a vote/log append is not survivable.
    pub async fn append(&self, record: &WalRecord) -> std::io::Result<()> {
        let payload = bincode::serialize(record).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "payload >4GB"))?;
        let crc = crc32fast::hash(&payload);
        let mut frame = Vec::with_capacity(8 + payload.len());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&payload);

        let mut guard = self.inner.lock().await;
        // The OpenOptions::append above makes every write
        // O_APPEND on POSIX, so we don't need to seek.
        guard.file.write_all(&frame)?;
        guard.file.sync_data()?;
        guard.bytes_written += frame.len() as u64;
        Ok(())
    }

    /// Read every valid record from disk. On a torn last record
    /// (kill-during-fsync), truncates the file at the last clean
    /// boundary and returns the records that landed cleanly.
    /// Errors only fire for genuine I/O problems, not corruption.
    pub async fn replay(&self) -> std::io::Result<Vec<WalRecord>> {
        let mut guard = self.inner.lock().await;
        // Re-read from the start by opening a fresh handle — the
        // append handle is positioned at EOF and seek-on-append
        // is platform-dependent.
        let path = guard.path.clone();
        drop(guard);

        let mut f = File::open(&path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let (records, clean_len) = parse_records(&buf);
        if clean_len != buf.len() {
            // Last record was torn — truncate the file at the
            // last clean boundary and log it.
            tracing::warn!(
                clean_len,
                total = buf.len(),
                path = %path.display(),
                "WAL: torn final record detected; truncating",
            );
            let f = OpenOptions::new().write(true).open(&path)?;
            f.set_len(clean_len as u64)?;
            f.sync_data()?;
            let mut guard = self.inner.lock().await;
            guard.bytes_written = clean_len as u64;
        }
        Ok(records)
    }

    pub async fn bytes_written(&self) -> u64 {
        self.inner.lock().await.bytes_written
    }
}

/// Parse as many valid records as possible from `bytes`. Returns
/// (records, clean_len) where `clean_len` is the byte offset of
/// the first record that failed to parse (CRC mismatch or torn).
fn parse_records(bytes: &[u8]) -> (Vec<WalRecord>, usize) {
    let mut records = Vec::new();
    let mut pos = 0;
    while pos + 8 <= bytes.len() {
        let crc = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start + len;
        if payload_end > bytes.len() {
            // Torn record — header says `len` bytes follow but
            // we have fewer. Stop here.
            break;
        }
        let payload = &bytes[payload_start..payload_end];
        let actual_crc = crc32fast::hash(payload);
        if actual_crc != crc {
            // CRC mismatch — corruption. Stop here. (Could try to
            // skip and resume; v1 is conservative.)
            break;
        }
        match bincode::deserialize::<WalRecord>(payload) {
            Ok(rec) => {
                records.push(rec);
                pos = payload_end;
            }
            Err(_) => {
                // Shouldn't happen if CRC matched — but if bincode
                // disagrees we stop, same as torn.
                break;
            }
        }
    }
    (records, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::{CommittedLeaderId, EntryPayload};

    fn make_entry(idx: u64) -> Entry<VedaTypeConfig> {
        Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 1), idx),
            payload: EntryPayload::Blank,
        }
    }

    #[tokio::test]
    async fn append_and_replay_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let wal = WalLog::open(tmp.path()).unwrap();
        for i in 1..=10 {
            wal.append(&WalRecord::Entry(make_entry(i))).await.unwrap();
        }
        wal.append(&WalRecord::Truncate(LogId::new(
            CommittedLeaderId::new(1, 1),
            8,
        )))
        .await
        .unwrap();
        let records = wal.replay().await.unwrap();
        assert_eq!(records.len(), 11);
        match &records[10] {
            WalRecord::Truncate(lid) => assert_eq!(lid.index, 8),
            _ => panic!("expected Truncate"),
        }
    }

    #[tokio::test]
    async fn replay_survives_torn_last_record() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("raft_log.wal");
        {
            let wal = WalLog::open(tmp.path()).unwrap();
            for i in 1..=5 {
                wal.append(&WalRecord::Entry(make_entry(i))).await.unwrap();
            }
        }
        // Append garbage to simulate a torn write (header says
        // 1000 bytes follow but only 4 do).
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&0xdeadbeef_u32.to_le_bytes()).unwrap(); // crc
            f.write_all(&1000_u32.to_le_bytes()).unwrap(); // len
            f.write_all(b"abc!").unwrap(); // 4 bytes of "payload"
        }
        let wal = WalLog::open(tmp.path()).unwrap();
        let records = wal.replay().await.unwrap();
        assert_eq!(records.len(), 5, "torn final record should be discarded");
        // After replay, the file should have been truncated to the
        // clean length.
        let len_after = std::fs::metadata(&path).unwrap().len();
        // Append a 6th entry, replay should give 6 records.
        wal.append(&WalRecord::Entry(make_entry(6))).await.unwrap();
        let records2 = wal.replay().await.unwrap();
        assert_eq!(records2.len(), 6);
        let len_final = std::fs::metadata(&path).unwrap().len();
        assert!(
            len_final > len_after,
            "appending after recovery should grow the file",
        );
    }

    #[tokio::test]
    async fn replay_survives_crc_corruption() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("raft_log.wal");
        {
            let wal = WalLog::open(tmp.path()).unwrap();
            for i in 1..=3 {
                wal.append(&WalRecord::Entry(make_entry(i))).await.unwrap();
            }
        }
        // Flip a byte in the payload of the last record (the CRC
        // header is the first 4 bytes after entry 2's bytes; we
        // hit a payload byte well past it).
        {
            let mut all = std::fs::read(&path).unwrap();
            *all.last_mut().unwrap() ^= 0xff;
            std::fs::write(&path, all).unwrap();
        }
        let wal = WalLog::open(tmp.path()).unwrap();
        let records = wal.replay().await.unwrap();
        assert_eq!(records.len(), 2, "corrupted last record should be dropped");
    }
}
