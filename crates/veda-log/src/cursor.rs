//! Per-shard applier cursor: the highest log `seq_id` that is
//! durably reflected in the local SegmentManager. Written
//! atomically (tmp + fsync + rename), framed `[crc32][seq u64]`.
//! A missing or torn/corrupt file reads as 0 (full replay — safe,
//! because apply is idempotent on `(key, commit_ts)`).

use std::io::Write;
use std::path::Path;

/// Read the persisted cursor, or 0 if absent/short/corrupt.
pub fn read_cursor(path: impl AsRef<Path>) -> u64 {
    let bytes = match std::fs::read(path.as_ref()) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    if bytes.len() != 12 {
        return 0;
    }
    let crc_stored = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let payload = &bytes[4..12];
    if crc32fast::hash(payload) != crc_stored {
        return 0;
    }
    let mut seq = [0u8; 8];
    seq.copy_from_slice(payload);
    u64::from_le_bytes(seq)
}

/// Write the cursor atomically: tmp file + fsync + rename.
pub fn write_cursor(path: impl AsRef<Path>, seq: u64) -> std::io::Result<()> {
    let path = path.as_ref();
    let payload = seq.to_le_bytes();
    let crc = crc32fast::hash(&payload);
    let mut frame = Vec::with_capacity(12);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(&payload);

    let tmp = path.with_extension("cursor.tmp");
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(&frame)?;
        f.flush()?;
        f.sync_data()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_a_seq() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 42).unwrap();
        assert_eq!(read_cursor(&p), 42);
    }

    #[test]
    fn missing_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(read_cursor(tmp.path().join("nope.cursor")), 0);
    }

    #[test]
    fn corrupt_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 99).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(read_cursor(&p), 0, "CRC mismatch falls back to 0");
    }

    #[test]
    fn truncated_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 7).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() - 3]).unwrap();
        assert_eq!(read_cursor(&p), 0);
    }
}
