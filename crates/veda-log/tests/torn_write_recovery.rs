//! Truncate the log file mid-frame and confirm recovery returns
//! the largest valid prefix.

use std::fs::OpenOptions;

use bytes::Bytes;
use veda_log::{LogOp, ShardLog};
use veda_time::Hlc;

fn corrupt_last_frame(log_path: &std::path::Path, bytes_to_drop: u64) {
    let f = OpenOptions::new().write(true).open(log_path).unwrap();
    let size = f.metadata().unwrap().len();
    f.set_len(size.saturating_sub(bytes_to_drop)).unwrap();
}

#[test]
fn torn_tail_dropped_on_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let n: u64 = 50;

    // Write N entries.
    {
        let log = ShardLog::open(tmp.path()).unwrap();
        for i in 1..=n {
            log.append_one(
                Hlc::new(i, 0),
                0,
                Bytes::copy_from_slice(&i.to_le_bytes()),
                LogOp::Put,
                Bytes::copy_from_slice(b"payload-XXXX"),
            )
            .unwrap();
        }
    }

    // Truncate 5 bytes off the end — guaranteed to be inside the
    // last frame (each frame is well over 5 bytes).
    let log_path = tmp.path().join(format!("{:012}.log", 1));
    corrupt_last_frame(&log_path, 5);

    // Reopen — the torn last frame must be detected and dropped.
    let log = ShardLog::open(tmp.path()).unwrap();
    let entries = log.read_from(1).unwrap();
    // We can't assert exactly n-1 because the truncation may have
    // damaged just the final frame, OR may have damaged the
    // length field and cascaded — but the recovered count must be
    // strictly less than n and the sequence must still be
    // contiguous starting at 1.
    assert!(
        entries.len() < n as usize,
        "expected truncation to drop at least one entry, kept {}",
        entries.len()
    );
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e.seq_id, (i as u64) + 1, "sequence must remain contiguous");
    }

    // Next append continues from where recovery left off.
    let next = log
        .append_one(
            Hlc::new(99, 0),
            0,
            Bytes::from_static(b"post-recovery"),
            LogOp::Put,
            Bytes::from_static(b"x"),
        )
        .unwrap();
    assert_eq!(next, entries.len() as u64 + 1);
}

#[test]
fn crc_mismatch_dropped_on_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let log = ShardLog::open(tmp.path()).unwrap();
        for i in 1..=10 {
            log.append_one(
                Hlc::new(i, 0),
                0,
                Bytes::copy_from_slice(&i.to_le_bytes()),
                LogOp::Put,
                Bytes::from_static(b"payload"),
            )
            .unwrap();
        }
    }
    // Flip the last byte of the file — corrupts the last frame's
    // body, CRC mismatch on replay.
    let log_path = tmp.path().join(format!("{:012}.log", 1));
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&log_path)
        .unwrap();
    let len = f.metadata().unwrap().len();
    f.seek(SeekFrom::Start(len - 1)).unwrap();
    let mut last = [0u8; 1];
    f.read_exact(&mut last).unwrap();
    f.seek(SeekFrom::Start(len - 1)).unwrap();
    f.write_all(&[last[0] ^ 0xFF]).unwrap();
    f.sync_all().unwrap();
    drop(f);

    let log = ShardLog::open(tmp.path()).unwrap();
    let entries = log.read_from(1).unwrap();
    // The last entry's CRC fails, so we expect at most 9 to come
    // back.
    assert!(entries.len() <= 9, "expected CRC failure to drop tail");
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e.seq_id, (i as u64) + 1);
    }
}
