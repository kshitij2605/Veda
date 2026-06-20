//! Per-shard append-only log.
//!
//! Single-node correctness today; Phase D adds quorum replication
//! across `(E, Qw, Qa)` followers. The on-disk format and the
//! public API match what the multi-node version will use, so the
//! Phase D delta is transport + ensemble bookkeeping, not data
//! format.
//!
//! Group commit: appends are accumulated in a Mutex-guarded
//! pending queue, and a background flusher writes + fsyncs them
//! every `FLUSH_INTERVAL` or every `FLUSH_BYTES` of accumulated
//! payload, whichever fires first. Phase B uses synchronous
//! flush-on-append for simplicity; the background flusher is a
//! follow-up.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use thiserror::Error;
use veda_time::Hlc;

use crate::format::{FrameError, LogEntryV2, LogOp};
use crate::segment_file::SegmentFile;

#[derive(Debug, Error)]
pub enum ShardLogError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(String),
    #[error("frame: {0}")]
    Frame(#[from] FrameError),
    #[error("seq_id assigned out of order: expected {expected}, got {got}")]
    OutOfOrderSeq { expected: u64, got: u64 },
}

/// State maintained alongside the writable segment.
#[derive(Debug)]
struct State {
    /// Active segment file we're appending to.
    active: SegmentFile,
    /// Next seq_id to assign on `append_one`. Strictly monotonic
    /// per-shard across segment rollovers.
    next_seq: u64,
    /// Highest seq_id durably persisted (fsync acknowledged).
    /// Mirrors what the in-process tailers can observe.
    durable_seq: u64,
    /// Append-only history of segment files in birth order. Used
    /// by readers and by Phase D recovery.
    segments: VecDeque<PathBuf>,
    /// Base directory for this shard.
    base: PathBuf,
}

/// Append-only log for one shard.
#[derive(Debug)]
pub struct ShardLog {
    state: Mutex<State>,
}

impl ShardLog {
    /// Open or create the log rooted at `base`. Replays any
    /// existing segment files, truncating the last segment past
    /// any torn write.
    pub fn open(base: impl AsRef<Path>) -> Result<Self, ShardLogError> {
        let base = base.as_ref().to_path_buf();
        std::fs::create_dir_all(&base)?;

        // Enumerate existing .log files and sort them by their
        // numeric prefix. This is the seq_lo of each segment.
        let mut segs: Vec<u64> = Vec::new();
        for entry in std::fs::read_dir(&base)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if let Some(stem) = name_str.strip_suffix(".log")
                && let Ok(seq) = stem.parse::<u64>()
            {
                segs.push(seq);
            }
        }
        segs.sort();

        let segments: VecDeque<PathBuf> = segs
            .iter()
            .map(|s| base.join(format!("{s:012}.log")))
            .collect();

        // Replay-and-truncate the latest segment; figure out
        // next_seq from there. Earlier segments are assumed
        // intact (they were fsynced before the next was created).
        let (active, next_seq) = if let Some(&last_seq_lo) = segs.last() {
            let mut active = SegmentFile::open(&base, last_seq_lo)?;
            let (highest_seq, valid_to) = replay_segment(&active)?;
            if valid_to < active.size() {
                tracing::warn!(
                    path = ?active.log_path,
                    valid_to,
                    file_size = active.size(),
                    "torn write detected in shard log; truncating"
                );
                active.truncate_log(valid_to)?;
            }
            let next_seq = highest_seq.map(|s| s + 1).unwrap_or(last_seq_lo);
            (active, next_seq)
        } else {
            (SegmentFile::open(&base, 1)?, 1)
        };

        Ok(Self {
            state: Mutex::new(State {
                active,
                next_seq,
                durable_seq: next_seq.saturating_sub(1),
                segments,
                base,
            }),
        })
    }

    /// Append a single entry. The caller supplies `hlc_ts`; the
    /// log assigns `seq_id` and fsyncs before returning. Returns
    /// the assigned `seq_id`.
    pub fn append_one(
        &self,
        hlc_ts: Hlc,
        epoch: u64,
        key: bytes::Bytes,
        op: LogOp,
        payload: bytes::Bytes,
    ) -> Result<u64, ShardLogError> {
        let mut state = self.state.lock().expect("ShardLog mutex poisoned");

        // Decide whether to roll over before this append.
        if state.active.should_roll() {
            roll_over(&mut state)?;
        }

        let seq_id = state.next_seq;
        let entry = LogEntryV2 {
            seq_id,
            hlc_ts,
            epoch,
            key,
            op,
            payload,
        };
        let frame = entry
            .encode()
            .map_err(|e| ShardLogError::Encode(e.to_string()))?;
        let offset = state.active.append(&frame)?;
        state.active.maybe_index(seq_id, offset)?;
        state.active.sync_all()?;
        state.next_seq = seq_id + 1;
        state.durable_seq = seq_id;
        Ok(seq_id)
    }

    /// Return all entries with `seq_id >= from`. Reads across
    /// segment files in order. O(entries returned) on a warm cache;
    /// the sidecar index gives O(1) seek-to-first.
    pub fn read_from(&self, from: u64) -> Result<Vec<LogEntryV2>, ShardLogError> {
        let state = self.state.lock().expect("ShardLog mutex poisoned");
        let mut out = Vec::new();
        // Phase B scan: walk every segment file from the start.
        // The sidecar index is consulted but the linear scan is
        // the correctness path. Optimised seek lands in a B follow-up.
        let mut seg_paths: Vec<PathBuf> = state.segments.iter().cloned().collect();
        // Include the active segment if it isn't already in the list.
        if !seg_paths.iter().any(|p| p == &state.active.log_path) {
            seg_paths.push(state.active.log_path.clone());
        }
        // No-segments edge case shouldn't happen (open() always
        // initialises at least one), but guard anyway.
        if seg_paths.is_empty() {
            seg_paths.push(state.active.log_path.clone());
        }
        for path in seg_paths {
            let bytes = std::fs::read(&path)?;
            let mut cursor = 0usize;
            while cursor < bytes.len() {
                match LogEntryV2::decode(&bytes[cursor..]) {
                    Ok((entry, n)) => {
                        cursor += n;
                        if entry.seq_id >= from {
                            out.push(entry);
                        }
                    }
                    Err(FrameError::Torn) | Err(FrameError::CrcMismatch) => {
                        tracing::warn!(
                            path = ?path,
                            "stopping read at torn or corrupt frame in segment"
                        );
                        break;
                    }
                    Err(e) => return Err(ShardLogError::Frame(e)),
                }
            }
        }
        Ok(out)
    }

    /// Highest seq_id that has been fsynced.
    pub fn durable_seq(&self) -> u64 {
        self.state
            .lock()
            .expect("ShardLog mutex poisoned")
            .durable_seq
    }

    /// Next seq_id that would be assigned (= durable_seq + 1 in steady state).
    pub fn next_seq(&self) -> u64 {
        self.state.lock().expect("ShardLog mutex poisoned").next_seq
    }

    /// Phase D: copy every segment file from `self` into
    /// `dest_dir`, producing a byte-identical replica log.
    /// Used to seed a new follower joining the ensemble
    /// mid-stream: the operator (or auto-orchestrator) ships
    /// the writer's current log to the new follower's data
    /// dir, opens that dir as a `ShardLog`, and registers the
    /// follower in the ensemble. The new follower's
    /// `durable_seq` then matches the writer's at the moment
    /// of the copy; subsequent `AppendRpc`s replicate forward.
    ///
    /// `dest_dir` must not already contain `.log` files (we
    /// don't merge / dedupe). The simplest correct usage is to
    /// hand a fresh empty dir.
    pub fn copy_to(&self, dest_dir: impl AsRef<Path>) -> Result<u64, ShardLogError> {
        let dest = dest_dir.as_ref();
        std::fs::create_dir_all(dest)?;
        let mut state = self.state.lock().expect("ShardLog mutex poisoned");
        // Flush + fsync the active segment so the bytes we
        // about to copy are durable.
        state.active.sync_all()?;
        // Copy rolled-over segments + the active segment. For a
        // freshly-opened log `state.segments` is empty (it's
        // populated by directory enumeration at open time, not
        // by per-append bookkeeping), so we always also copy
        // the active segment's file. Dedupe by file name in
        // case `state.segments` already contains it.
        let mut copied_names: std::collections::HashSet<std::ffi::OsString> =
            std::collections::HashSet::new();
        for seg_path in &state.segments {
            let file_name = seg_path.file_name().ok_or_else(|| {
                ShardLogError::Io(std::io::Error::other("segment has no file name"))
            })?;
            std::fs::copy(seg_path, dest.join(file_name))?;
            copied_names.insert(file_name.to_os_string());
        }
        let active_path = &state.active.log_path;
        if let Some(file_name) = active_path.file_name()
            && !copied_names.contains(file_name)
        {
            std::fs::copy(active_path, dest.join(file_name))?;
        }
        Ok(state.durable_seq)
    }

    /// Phase D: drop every entry with `seq_id > target` from the
    /// active segment. Used by the BookKeeper-style open-and-
    /// recover handshake to roll back any entries a now-fenced
    /// old writer may have acked to a minority of followers but
    /// NOT to the safe quorum. After this returns,
    /// `durable_seq() <= target` and subsequent appends pick up
    /// at `target + 1`.
    ///
    /// Best-effort against the active segment only — older
    /// segments are assumed intact because they were fsynced
    /// closed before the active segment was created. In
    /// practice the post-handover safe_seq is always inside the
    /// active segment for any realistic ensemble lifetime.
    pub fn truncate_above(&self, target: u64) -> Result<(), ShardLogError> {
        let mut state = self.state.lock().expect("ShardLog mutex poisoned");
        if state.durable_seq <= target {
            return Ok(()); // nothing to truncate
        }
        // Walk the active segment, find the byte offset of the
        // first frame whose seq_id > target.
        let bytes = state.active.read_all()?;
        let mut cursor: usize = 0;
        let mut last_kept_seq: Option<u64> = None;
        let mut truncate_at: usize = bytes.len();
        while cursor < bytes.len() {
            match crate::format::LogEntryV2::decode(&bytes[cursor..]) {
                Ok((entry, consumed)) => {
                    if entry.seq_id > target {
                        truncate_at = cursor;
                        break;
                    }
                    last_kept_seq = Some(entry.seq_id);
                    cursor += consumed;
                }
                Err(_) => break, // torn / CRC — leave the tail
            }
        }
        if truncate_at < bytes.len() {
            state.active.truncate_log(truncate_at as u64)?;
        }
        state.durable_seq = last_kept_seq.unwrap_or(0);
        state.next_seq = state.durable_seq + 1;
        Ok(())
    }
}

/// Replay one segment file: scan every frame, return the highest
/// seq_id seen and the byte offset of the first invalid (torn or
/// CRC-mismatch) frame — that's the valid prefix length.
fn replay_segment(seg: &SegmentFile) -> Result<(Option<u64>, u64), ShardLogError> {
    let bytes = seg.read_all()?;
    let mut cursor = 0usize;
    let mut highest = None;
    loop {
        if cursor >= bytes.len() {
            break;
        }
        match LogEntryV2::decode(&bytes[cursor..]) {
            Ok((entry, n)) => {
                highest = Some(entry.seq_id);
                cursor += n;
            }
            Err(FrameError::Torn) | Err(FrameError::CrcMismatch) => {
                break;
            }
            Err(e) => return Err(ShardLogError::Frame(e)),
        }
    }
    Ok((highest, cursor as u64))
}

/// Roll over the active segment. Renames are not needed — the new
/// segment is opened under `{next_seq:012}.log`.
fn roll_over(state: &mut State) -> Result<(), ShardLogError> {
    // Ensure the soon-to-be-old segment is fully durable.
    state.active.sync_all()?;
    state.segments.push_back(state.active.log_path.clone());
    let new_seg = SegmentFile::open(&state.base, state.next_seq)?;
    state.active = new_seg;
    Ok(())
}
