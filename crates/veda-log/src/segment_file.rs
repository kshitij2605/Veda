//! On-disk segment file for a `ShardLog`.
//!
//! Layout: a directory `{base}/{shard_id}/` holds:
//!   - `000000000001.log` — append-only entries (CRC32 framed)
//!   - `000000000001.idx` — sidecar: u64 LE pairs `(seq_id, byte_offset)`
//!     written every `INDEX_INTERVAL_BYTES` of the log
//!   - rolled over at `MAX_SEGMENT_BYTES` (default 256MB), at
//!     which point a new `000000000002.{log,idx}` is started.
//!
//! Phase B ships a single active segment file. Multi-segment
//! rollover is implemented but lands in the same Phase B PR as
//! Tests will live in `shard_log.rs` against the full append/read
//! API surface, since the segment file isn't directly part of the
//! public surface.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Roll over to a new segment file when the current one exceeds
/// this size. Configurable in tests via [`SegmentFile::with_max`].
pub const MAX_SEGMENT_BYTES: u64 = 256 * 1024 * 1024;

/// Write one index entry every N bytes of log. 4KB matches the
/// FoundationDB / RocksDB default; bigger values save sidecar
/// space at the cost of O(N/INTERVAL) scan per tail-from.
pub const INDEX_INTERVAL_BYTES: u64 = 4 * 1024;

/// Single segment file pair (log + sidecar index).
///
/// Phase B uses one of these as the "active" file; rollover
/// produces a new one. Readers open the file series in order via
/// [`SegmentSet`].
#[derive(Debug)]
pub struct SegmentFile {
    pub log_path: PathBuf,
    pub idx_path: PathBuf,
    pub seq_lo: u64,
    log_file: File,
    idx_file: File,
    /// Byte position in the log file. == log_file.metadata().len()
    /// after construction; tracked in-memory so we don't syscall
    /// on every append.
    write_pos: u64,
    /// Byte offset of the last index entry written. Used to
    /// decide when to emit a new index entry.
    last_index_pos: u64,
    max_bytes: u64,
}

impl SegmentFile {
    /// Open or create the segment whose first seq_id is `seq_lo`.
    /// If the file exists it's opened for append; if not, it's
    /// created empty.
    pub fn open(base: &Path, seq_lo: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(base)?;
        let log_path = base.join(format!("{seq_lo:012}.log"));
        let idx_path = base.join(format!("{seq_lo:012}.idx"));
        let log_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&log_path)?;
        let idx_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&idx_path)?;
        let write_pos = log_file.metadata()?.len();
        Ok(Self {
            log_path,
            idx_path,
            seq_lo,
            log_file,
            idx_file,
            write_pos,
            last_index_pos: 0,
            max_bytes: MAX_SEGMENT_BYTES,
        })
    }

    pub fn with_max(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Append a frame to the log. Returns the byte offset of the
    /// frame's first byte. Caller is responsible for emitting the
    /// index entry by calling [`Self::maybe_index`] with the
    /// entry's `seq_id`.
    pub fn append(&mut self, frame: &[u8]) -> std::io::Result<u64> {
        let offset = self.write_pos;
        self.log_file.write_all(frame)?;
        self.write_pos += frame.len() as u64;
        Ok(offset)
    }

    /// Emit a sidecar index entry if enough log bytes have
    /// accumulated since the last one.
    pub fn maybe_index(&mut self, seq_id: u64, frame_offset: u64) -> std::io::Result<()> {
        if self.write_pos.saturating_sub(self.last_index_pos) < INDEX_INTERVAL_BYTES {
            return Ok(());
        }
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&seq_id.to_le_bytes());
        buf[8..].copy_from_slice(&frame_offset.to_le_bytes());
        self.idx_file.write_all(&buf)?;
        self.last_index_pos = self.write_pos;
        Ok(())
    }

    /// fsync both log + index. Called after a batch is flushed
    /// from the writer.
    pub fn sync_all(&mut self) -> std::io::Result<()> {
        self.log_file.sync_all()?;
        self.idx_file.sync_all()?;
        Ok(())
    }

    /// Current size of the log file in bytes.
    pub fn size(&self) -> u64 {
        self.write_pos
    }

    /// True if the segment has grown past its rollover threshold.
    pub fn should_roll(&self) -> bool {
        self.write_pos >= self.max_bytes
    }

    /// Read the entire log file body. Used by replay paths in
    /// [`crate::shard_log::ShardLog::recover`]. Returns the bytes
    /// and the file's length (so the caller knows where the last
    /// successfully-decoded frame should end).
    pub fn read_all(&self) -> std::io::Result<Vec<u8>> {
        let mut file = File::open(&self.log_path)?;
        let len = file.metadata()?.len() as usize;
        let mut buf = Vec::with_capacity(len);
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }

    /// Read the sidecar index entries. Returns `Vec<(seq_id,
    /// byte_offset)>` sorted by seq_id.
    pub fn read_index(&self) -> std::io::Result<Vec<(u64, u64)>> {
        let mut file = File::open(&self.idx_path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut out = Vec::with_capacity(bytes.len() / 16);
        for chunk in bytes.chunks_exact(16) {
            let seq = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let off = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
            out.push((seq, off));
        }
        Ok(out)
    }

    /// Truncate the log file in-place. Used by recovery when the
    /// scan finds a torn write at byte `valid_to` — everything
    /// past that is dropped so the next append starts at a known
    /// good position.
    pub fn truncate_log(&mut self, valid_to: u64) -> std::io::Result<()> {
        self.log_file.flush()?;
        let f = OpenOptions::new().write(true).open(&self.log_path)?;
        f.set_len(valid_to)?;
        // Reopen the append handle so the in-memory write_pos
        // matches the on-disk length.
        let log_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.log_path)?;
        // Seek to end to make sure append() writes after the
        // truncation point.
        let mut log_file = log_file;
        log_file.seek(SeekFrom::End(0))?;
        self.log_file = log_file;
        self.write_pos = valid_to;
        Ok(())
    }
}
