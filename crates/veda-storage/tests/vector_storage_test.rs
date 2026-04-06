use std::io::Write;

use tempfile::TempDir;
use veda_storage::vector_storage::{VectorStorageReader, VectorStorageWriter};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a deterministic pseudo-random vector of length `dim`.
fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let val = ((state >> 16) as f32 / 32768.0) - 1.0;
        v.push(val);
    }
    v
}

/// Write `count` vectors of `dim` dimensions to a file, returning the generated vectors.
fn write_vectors(dir: &TempDir, dim: usize, count: usize) -> Vec<Vec<f32>> {
    let path = dir.path().join("vectors.veda");
    let mut writer = VectorStorageWriter::create(&path, dim as u32).unwrap();
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let v = make_vector(dim, i as u32);
        let id = writer.append(&v).unwrap();
        assert_eq!(id, i as u32);
        vectors.push(v);
    }
    writer.close().unwrap();
    vectors
}

// ── Basic round-trip ────────────────────────────────────────────────────────

#[test]
fn test_write_and_read_1000_vectors_dim_768() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 768, 1000);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    assert_eq!(reader.count(), 1000);
    assert_eq!(reader.dimension(), 768);

    for (i, expected) in vectors.iter().enumerate() {
        let actual = reader.get(i as u32).unwrap();
        assert_eq!(actual, expected.as_slice(), "mismatch at vector {i}");
    }
}

// ── get_batch ───────────────────────────────────────────────────────────────

#[test]
fn test_get_batch_with_selected_ids() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 128, 500);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    let ids: Vec<u32> = vec![0, 42, 99, 256, 499];
    let batch = reader.get_batch(&ids).unwrap();

    assert_eq!(batch.len(), ids.len());
    for (&id, actual) in ids.iter().zip(batch.iter()) {
        assert_eq!(*actual, vectors[id as usize].as_slice());
    }
}

#[test]
fn test_get_batch_empty_ids() {
    let dir = TempDir::new().unwrap();
    write_vectors(&dir, 128, 10);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    let batch = reader.get_batch(&[]).unwrap();
    assert!(batch.is_empty());
}

// ── iter ────────────────────────────────────────────────────────────────────

#[test]
fn test_iter_all_vectors_in_order() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 256, 200);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    for (i, actual) in reader.iter().enumerate() {
        assert_eq!(actual, vectors[i].as_slice(), "iter mismatch at {i}");
    }

    // Also check that iter yields the right count.
    assert_eq!(reader.iter().count(), 200);
}

// ── Dimension mismatch on append ────────────────────────────────────────────

#[test]
fn test_dimension_mismatch_on_append() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.veda");
    let mut writer = VectorStorageWriter::create(&path, 128).unwrap();

    // Correct dimension works.
    let v = make_vector(128, 0);
    assert!(writer.append(&v).is_ok());

    // Wrong dimension fails.
    let bad = make_vector(256, 1);
    let err = writer.append(&bad);
    assert!(err.is_err());
    let err_msg = format!("{}", err.unwrap_err());
    assert!(
        err_msg.contains("mismatch"),
        "expected dimension mismatch error, got: {err_msg}"
    );

    writer.close().unwrap();
}

// ── Invalid file errors ─────────────────────────────────────────────────────

#[test]
fn test_open_invalid_magic_bytes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad.veda");

    // Write a file with invalid magic.
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"BAAD").unwrap();
    f.write_all(&[0u8; 28]).unwrap(); // rest of "header"
    f.flush().unwrap();

    let result = VectorStorageReader::open(&path);
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("magic"),
        "expected magic bytes error, got: {err_msg}"
    );
}

#[test]
fn test_open_file_too_small() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tiny.veda");

    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"VEDA").unwrap(); // only 4 bytes, need 32
    f.flush().unwrap();

    let result = VectorStorageReader::open(&path);
    assert!(result.is_err());
}

// ── Empty storage ───────────────────────────────────────────────────────────

#[test]
fn test_empty_storage_zero_vectors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.veda");

    let writer = VectorStorageWriter::create(&path, 128).unwrap();
    assert_eq!(writer.count(), 0);
    writer.close().unwrap();

    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.count(), 0);
    assert_eq!(reader.dimension(), 128);
    assert_eq!(reader.iter().count(), 0);

    // Out of range access.
    let err = reader.get(0);
    assert!(err.is_err());
}

// ── Various dimensions ──────────────────────────────────────────────────────

#[test]
fn test_dimension_128() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 128, 50);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.dimension(), 128);
    assert_eq!(reader.count(), 50);

    for (i, expected) in vectors.iter().enumerate() {
        assert_eq!(reader.get(i as u32).unwrap(), expected.as_slice());
    }
}

#[test]
fn test_dimension_512() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 512, 50);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.dimension(), 512);
    assert_eq!(reader.count(), 50);

    for (i, expected) in vectors.iter().enumerate() {
        assert_eq!(reader.get(i as u32).unwrap(), expected.as_slice());
    }
}

#[test]
fn test_dimension_1536() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 1536, 50);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.dimension(), 1536);
    assert_eq!(reader.count(), 50);

    for (i, expected) in vectors.iter().enumerate() {
        assert_eq!(reader.get(i as u32).unwrap(), expected.as_slice());
    }
}

// ── Large batch ─────────────────────────────────────────────────────────────

#[test]
fn test_large_batch_10000_vectors_dim_128() {
    let dir = TempDir::new().unwrap();
    let vectors = write_vectors(&dir, 128, 10_000);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.count(), 10_000);
    assert_eq!(reader.dimension(), 128);

    // Spot-check first, last, and some in the middle.
    assert_eq!(reader.get(0).unwrap(), vectors[0].as_slice());
    assert_eq!(reader.get(9999).unwrap(), vectors[9999].as_slice());
    assert_eq!(reader.get(5000).unwrap(), vectors[5000].as_slice());

    // Verify all via iterator.
    for (i, actual) in reader.iter().enumerate() {
        assert_eq!(actual, vectors[i].as_slice(), "mismatch at vector {i}");
    }
}

// ── Out of range access ─────────────────────────────────────────────────────

#[test]
fn test_get_out_of_range() {
    let dir = TempDir::new().unwrap();
    write_vectors(&dir, 128, 10);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    assert!(reader.get(10).is_err());
    assert!(reader.get(100).is_err());
    assert!(reader.get(u32::MAX).is_err());
}

// ── Batch with out of range ID ──────────────────────────────────────────────

#[test]
fn test_get_batch_with_invalid_id() {
    let dir = TempDir::new().unwrap();
    write_vectors(&dir, 128, 10);

    let path = dir.path().join("vectors.veda");
    let reader = VectorStorageReader::open(&path).unwrap();

    let result = reader.get_batch(&[0, 5, 10]); // id 10 is out of range
    assert!(result.is_err());
}

// ── Writer count tracks correctly ───────────────────────────────────────────

#[test]
fn test_writer_count_tracks_appends() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.veda");
    let mut writer = VectorStorageWriter::create(&path, 64).unwrap();

    assert_eq!(writer.count(), 0);

    for i in 0..5 {
        let v = make_vector(64, i);
        writer.append(&v).unwrap();
        assert_eq!(writer.count(), i as u64 + 1);
    }

    writer.close().unwrap();
}

// ── Flush mid-write ─────────────────────────────────────────────────────────

#[test]
fn test_flush_does_not_corrupt() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vectors.veda");
    let mut writer = VectorStorageWriter::create(&path, 128).unwrap();

    for i in 0..50 {
        let v = make_vector(128, i);
        writer.append(&v).unwrap();
    }
    writer.flush().unwrap();

    for i in 50..100 {
        let v = make_vector(128, i);
        writer.append(&v).unwrap();
    }
    writer.close().unwrap();

    let reader = VectorStorageReader::open(&path).unwrap();
    assert_eq!(reader.count(), 100);

    for i in 0..100u32 {
        let expected = make_vector(128, i);
        assert_eq!(reader.get(i).unwrap(), expected.as_slice());
    }
}
