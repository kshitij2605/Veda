//! Phase F MVCC durability: append-only WAL for [`MvccStore`].
//!
//! v1 ships *just* the WAL + replay path. Every mutator on
//! `MvccStore` produces one record (`Put` / `Delete` /
//! `SetSafeTs`); recovery replays them in order to rebuild the
//! in-memory btree on startup.
//!
//! Periodic checkpointing (write the full btree to a snapshot
//! file + truncate the WAL) is a follow-up. The WAL alone is
//! correct but unbounded; operators with heavy MVCC churn need
//! the checkpoint loop, but for the rollout the WAL is the
//! correctness foundation.
//!
//! Wire format mirrors `veda-log`: each record is framed as
//! `[crc32 u32 LE][len u32 LE][bincode(MvccWalEntry)]`. CRC
//! covers `[len || payload]`. Torn-tail (partial write at
//! shutdown) is recovered by stopping at the first frame whose
//! CRC fails — same recovery shape as `crates/veda-coordinator/
//! src/raft_wal.rs`.
//!
//! Cursor: not maintained explicitly. Replay re-applies every
//! valid frame from the start. Idempotence isn't needed because
//! `MvccStore::put` overwrites the same `(user_key, commit_ts)`
//! cleanly.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::mvcc::{MvccStore, MvccValue, Ts};

const FRAME_HEADER_BYTES: usize = 8; // crc32(u32) + len(u32)

#[derive(Debug, thiserror::Error)]
pub enum MvccWalError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
    #[error("store: {0}")]
    Store(String),
}

/// One persisted operation. The variants mirror `MvccStore`'s
/// public mutators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MvccWalEntry {
    Put {
        user_key: Vec<u8>,
        commit_ts: Ts,
        payload: Bytes,
    },
    Delete {
        user_key: Vec<u8>,
        commit_ts: Ts,
    },
    SetSafeTs {
        ts: Ts,
    },
}

/// Append-only WAL backing an `MvccStore`. One file per
/// `(collection, shard_id)` — caller decides the path.
pub struct MvccWal {
    path: PathBuf,
    file: BufWriter<File>,
}

impl MvccWal {
    /// Open (or create) the WAL at `path`. Existing contents
    /// are kept; new writes append.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MvccWalError> {
        if let Some(parent) = path.as_ref().parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
        }
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.as_ref())?;
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            file: BufWriter::new(f),
        })
    }

    /// Append a record + fsync. Caller is responsible for
    /// throttling — every call is one fsync.
    pub fn append(&mut self, entry: &MvccWalEntry) -> Result<(), MvccWalError> {
        let payload = bincode::serialize(entry)
            .map_err(|e| MvccWalError::Encode(format!("{e}")))?;
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);
        // Format: [crc32 LE][len LE][payload]
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.flush()?;
        self.file.get_ref().sync_data()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replay every record from a WAL file into `store`. Stops
    /// (cleanly) at the first frame whose CRC fails — that's a
    /// torn write at shutdown and the entries past it are
    /// dropped.
    ///
    /// Returns `(applied, dropped_tail_bytes)` so the caller can
    /// log / metric the recovery outcome.
    pub fn replay_into(
        path: impl AsRef<Path>,
        store: &mut MvccStore,
    ) -> Result<(usize, u64), MvccWalError> {
        let mut f = match File::open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((0, 0)),
            Err(e) => return Err(e.into()),
        };
        let mut applied = 0usize;
        let file_len = f.metadata()?.len();
        let mut cursor = 0u64;
        loop {
            if file_len.saturating_sub(cursor) < FRAME_HEADER_BYTES as u64 {
                break;
            }
            let mut header = [0u8; FRAME_HEADER_BYTES];
            f.seek(SeekFrom::Start(cursor))?;
            if f.read_exact(&mut header).is_err() {
                break;
            }
            let crc = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            if cursor + FRAME_HEADER_BYTES as u64 + len as u64 > file_len {
                // Torn tail: ignore.
                break;
            }
            let mut payload = vec![0u8; len];
            if f.read_exact(&mut payload).is_err() {
                break;
            }
            if crc32fast::hash(&payload) != crc {
                // CRC mismatch — torn write. Stop here.
                break;
            }
            let entry: MvccWalEntry = match bincode::deserialize(&payload) {
                Ok(e) => e,
                Err(_) => break,
            };
            apply_entry(store, entry).map_err(MvccWalError::Store)?;
            applied += 1;
            cursor += FRAME_HEADER_BYTES as u64 + len as u64;
        }
        let dropped = file_len - cursor;
        Ok((applied, dropped))
    }
}

fn apply_entry(store: &mut MvccStore, entry: MvccWalEntry) -> Result<(), String> {
    match entry {
        MvccWalEntry::Put {
            user_key,
            commit_ts,
            payload,
        } => store
            .put_payload(&user_key, commit_ts, payload)
            .map_err(|e| format!("put: {e}")),
        MvccWalEntry::Delete {
            user_key,
            commit_ts,
        } => store
            .delete(&user_key, commit_ts)
            .map_err(|e| format!("delete: {e}")),
        MvccWalEntry::SetSafeTs { ts } => {
            store.set_safe_ts(ts);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_entry(k: &[u8], ts: Ts, val: &[u8]) -> MvccWalEntry {
        MvccWalEntry::Put {
            user_key: k.to_vec(),
            commit_ts: ts,
            payload: Bytes::copy_from_slice(val),
        }
    }

    #[test]
    fn append_then_replay_round_trips_three_versions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mvcc.wal");
        {
            let mut wal = MvccWal::open(&path).unwrap();
            wal.append(&put_entry(b"k1", 100, b"v1")).unwrap();
            wal.append(&put_entry(b"k1", 200, b"v2")).unwrap();
            wal.append(&MvccWalEntry::Delete {
                user_key: b"k1".to_vec(),
                commit_ts: 300,
            })
            .unwrap();
        }
        let mut store = MvccStore::new();
        let (applied, dropped) = MvccWal::replay_into(&path, &mut store).unwrap();
        assert_eq!(applied, 3);
        assert_eq!(dropped, 0);
        // v1 visible at ts=150, v2 visible at ts=250, tombstone at 350.
        assert_eq!(store.get_at(b"k1", 150).as_deref(), Some(b"v1".as_slice()));
        assert_eq!(store.get_at(b"k1", 250).as_deref(), Some(b"v2".as_slice()));
        assert!(store.get_at(b"k1", 350).is_none());
    }

    #[test]
    fn replay_stops_at_torn_tail_crc_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mvcc.wal");
        {
            let mut wal = MvccWal::open(&path).unwrap();
            wal.append(&put_entry(b"k", 100, b"good")).unwrap();
            wal.append(&put_entry(b"k", 200, b"good")).unwrap();
        }
        // Corrupt the tail: append garbage that won't pass CRC.
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&[0xff; 32]).unwrap();
            f.sync_all().unwrap();
        }
        let mut store = MvccStore::new();
        let (applied, dropped) = MvccWal::replay_into(&path, &mut store).unwrap();
        // The 2 good entries replayed; the garbage tail was
        // dropped without error.
        assert_eq!(applied, 2);
        assert!(dropped >= 32, "expected ≥32 bytes dropped, got {dropped}");
        assert_eq!(store.get_at(b"k", 250).as_deref(), Some(b"good".as_slice()));
    }

    #[test]
    fn replay_of_missing_file_is_clean_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no-such.wal");
        let mut store = MvccStore::new();
        let (applied, dropped) = MvccWal::replay_into(&path, &mut store).unwrap();
        assert_eq!(applied, 0);
        assert_eq!(dropped, 0);
        assert_eq!(store.total_versions(), 0);
    }

    #[test]
    fn set_safe_ts_persists_through_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mvcc.wal");
        {
            let mut wal = MvccWal::open(&path).unwrap();
            wal.append(&put_entry(b"x", 100, b"a")).unwrap();
            wal.append(&MvccWalEntry::SetSafeTs { ts: 500 }).unwrap();
        }
        let mut store = MvccStore::new();
        let _ = MvccWal::replay_into(&path, &mut store).unwrap();
        assert_eq!(store.safe_ts(), 500);
    }

    /// Truncate a frame mid-payload (torn write). Replay must
    /// drop the truncated frame and any earlier frames stay.
    #[test]
    fn truncated_payload_is_dropped_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mvcc.wal");
        {
            let mut wal = MvccWal::open(&path).unwrap();
            wal.append(&put_entry(b"k1", 100, b"alpha")).unwrap();
            wal.append(&put_entry(b"k2", 100, b"beta")).unwrap();
        }
        // Truncate to lose the last byte of the second frame's
        // payload.
        let len = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap();
        f.set_len(len - 1).unwrap();
        let mut store = MvccStore::new();
        let (applied, dropped) = MvccWal::replay_into(&path, &mut store).unwrap();
        assert_eq!(applied, 1, "only the first frame should survive");
        assert!(dropped > 0, "tail should report bytes dropped");
        assert_eq!(store.get_at(b"k1", 150).as_deref(), Some(b"alpha".as_slice()));
        assert!(store.get_at(b"k2", 150).is_none());
    }
}
