//! On-disk + wire format for `LogEntryV2`.
//!
//! Framing: `[u32 LE crc32][u32 LE bincode_len][bincode payload]`
//! repeated. Identical envelope to `WalLog`'s format in
//! `crates/veda-coordinator/src/raft_wal.rs:68`, so the recovery
//! logic ported there applies here too.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use veda_time::Hlc;

/// Size in bytes of the per-entry frame header (crc32 + len).
pub const FRAME_HEADER_BYTES: usize = 8;

/// Operation on a key in the log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogOp {
    /// Insert / upsert. `payload` is the bincode-serialised record;
    /// concrete record type lives in `veda-core`.
    Put,
    /// Tombstone. `payload` is empty.
    Delete,
    /// Transaction commit marker. `payload` is the txn_id.
    /// Phase G uses this; Phase B emits the variant but the
    /// applier currently ignores it.
    TxnCommit,
    /// Reserved for future control-plane markers (e.g. shard
    /// rollover, ensemble change). Payload is opaque.
    Marker,
}

/// A single log entry.
///
/// `seq_id` is strictly monotonic per shard log. It is assigned by
/// the writer at append time, NOT by the caller — the caller asks
/// for an Hlc commit_ts up front, the log stamps seq_id when the
/// entry is committed durably.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntryV2 {
    pub seq_id: u64,
    pub hlc_ts: Hlc,
    /// Epoch of the writer-leader that emitted this entry. Phase D
    /// uses this to fence stale writers; Phase B emits a constant
    /// `0` until ensembles exist.
    pub epoch: u64,
    /// User-supplied key — typically a UUID byte-encoded.
    pub key: Bytes,
    pub op: LogOp,
    /// Op payload (record bytes for `Put`, txn_id bytes for
    /// `TxnCommit`, empty for `Delete`, opaque for `Marker`).
    pub payload: Bytes,
}

impl LogEntryV2 {
    /// Serialise to the on-disk frame format.
    pub fn encode(&self) -> Result<Vec<u8>, bincode::Error> {
        let body = bincode::serialize(self)?;
        let len = body.len() as u32;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&body);
        let crc = hasher.finalize();
        let mut out = Vec::with_capacity(FRAME_HEADER_BYTES + body.len());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Decode one frame at the start of `buf`. Returns the entry
    /// and the number of bytes consumed.
    ///
    /// Returns `Err(FrameError::Torn)` if the frame is truncated
    /// (likely a torn write — the caller should treat everything
    /// from here on as lost and stop the scan).
    /// Returns `Err(FrameError::CrcMismatch)` if the CRC doesn't
    /// validate (same recovery treatment as Torn).
    pub fn decode(buf: &[u8]) -> Result<(LogEntryV2, usize), FrameError> {
        if buf.len() < FRAME_HEADER_BYTES {
            return Err(FrameError::Torn);
        }
        let crc = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        if buf.len() < FRAME_HEADER_BYTES + len {
            return Err(FrameError::Torn);
        }
        let body = &buf[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + len];
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(body);
        if hasher.finalize() != crc {
            return Err(FrameError::CrcMismatch);
        }
        let entry: LogEntryV2 =
            bincode::deserialize(body).map_err(|e| FrameError::Decode(e.to_string()))?;
        Ok((entry, FRAME_HEADER_BYTES + len))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("frame is truncated (torn write)")]
    Torn,
    #[error("crc32 mismatch — entry corrupt")]
    CrcMismatch,
    #[error("bincode decode failed: {0}")]
    Decode(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(seq_id: u64) -> LogEntryV2 {
        LogEntryV2 {
            seq_id,
            hlc_ts: Hlc::new(1_700_000_000_000_000_000 + seq_id, 0),
            epoch: 0,
            key: Bytes::from_static(b"k"),
            op: LogOp::Put,
            payload: Bytes::from_static(b"{\"v\":1}"),
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let e = sample_entry(42);
        let bytes = e.encode().unwrap();
        let (decoded, n) = LogEntryV2::decode(&bytes).unwrap();
        assert_eq!(decoded, e);
        assert_eq!(n, bytes.len());
    }

    #[test]
    fn decode_detects_truncated_header() {
        let bytes = sample_entry(1).encode().unwrap();
        let result = LogEntryV2::decode(&bytes[..FRAME_HEADER_BYTES - 1]);
        assert!(matches!(result, Err(FrameError::Torn)));
    }

    #[test]
    fn decode_detects_truncated_body() {
        let bytes = sample_entry(1).encode().unwrap();
        let result = LogEntryV2::decode(&bytes[..bytes.len() - 1]);
        assert!(matches!(result, Err(FrameError::Torn)));
    }

    #[test]
    fn decode_detects_crc_mismatch() {
        let mut bytes = sample_entry(1).encode().unwrap();
        // Flip a byte in the payload (after the header) — body CRC fails.
        bytes[FRAME_HEADER_BYTES + 1] ^= 0xFF;
        let result = LogEntryV2::decode(&bytes);
        assert!(matches!(result, Err(FrameError::CrcMismatch)));
    }

    #[test]
    fn frames_concatenate_cleanly() {
        let a = sample_entry(1).encode().unwrap();
        let b = sample_entry(2).encode().unwrap();
        let mut buf = a.clone();
        buf.extend_from_slice(&b);
        let (decoded_a, na) = LogEntryV2::decode(&buf).unwrap();
        let (decoded_b, nb) = LogEntryV2::decode(&buf[na..]).unwrap();
        assert_eq!(decoded_a.seq_id, 1);
        assert_eq!(decoded_b.seq_id, 2);
        assert_eq!(na + nb, buf.len());
    }
}
