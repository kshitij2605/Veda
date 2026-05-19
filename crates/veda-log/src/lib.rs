//! Append-only sharded log layer for Veda.
//!
//! Phase B of the FoundationDB-shape rearchitecture
//! (`.claude/plans/structured-mixing-turing.md`). The crate ships a
//! single-node `ShardLog` with the on-disk format the eventual
//! multi-node version will use; replication lands in Phase D.
//!
//! ## What's here
//!
//! - [`format::LogEntryV2`] — the wire + on-disk record. Carries
//!   `(seq_id, hlc_ts, epoch, key, op, payload)` framed with the
//!   `[crc32][len][bincode_payload]` scheme inherited from
//!   `veda-coordinator::raft_wal`.
//! - [`shard_log::ShardLog`] — per-shard append-only log. Group
//!   commit (1ms / 64KB batches), 256MB segment rollover, 4KB
//!   sidecar index for O(1) tail-from.
//! - [`reader::LogTailer`] — yields gapless ordered entries from
//!   the log, surfacing a `lag_seq` watermark for consumers.
//!
//! ## What's *not* here yet
//!
//! - Multi-node replication (Phase D — BookKeeper-style ensembles).
//! - Snapshot / truncation orchestration with metadata Raft
//!   (Phase E).
//! - WAN log shipping (Phase H).
//!
//! ## Reuse from elsewhere in the tree
//!
//! - The CRC32 + length framing in [`format::FRAME_HEADER_BYTES`]
//!   is verbatim from `crates/veda-coordinator/src/raft_wal.rs:68`.
//!   `ShardLog::recover` mirrors `WalLog::replay` lines 130-228.

pub mod format;
pub mod reader;
pub mod replication;
pub mod segment_file;
pub mod shard_log;
pub mod wan;

pub use format::{LogEntryV2, LogOp};
pub use reader::LogTailer;
pub use replication::{
    AppendAck, AppendRpc, EnsembleError, LocalFollower, LogEnsemble, LogFollower, LogWriter, NodeId,
};
pub use shard_log::{ShardLog, ShardLogError};
