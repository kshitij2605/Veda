use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;
use veda_core::error::{Result, VedaError};

// ── File format constants ───────────────────────────────────────────────────

const MAGIC: [u8; 4] = *b"VEDA";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

// Data type codes
const DATA_TYPE_F32: u32 = 0;

// ── Header layout ───────────────────────────────────────────────────────────
//
// Offset  Size  Field
// 0       4     magic: [u8; 4] = b"VEDA"
// 4       4     version: u32 = 1
// 8       4     dimension: u32
// 12      8     count: u64
// 20      4     data_type: u32 (0 = f32)
// 24      8     reserved: [u8; 8]
// ─────────────────────────────────────────────────────────────────────────────

fn write_header(
    writer: &mut impl Write,
    dimension: u32,
    count: u64,
    data_type: u32,
) -> std::io::Result<()> {
    writer.write_all(&MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&dimension.to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    writer.write_all(&data_type.to_le_bytes())?;
    writer.write_all(&[0u8; 8])?; // reserved
    Ok(())
}

fn parse_header(buf: &[u8]) -> Result<(u32, u64, u32)> {
    if buf.len() < HEADER_SIZE {
        return Err(VedaError::Storage("file too small for header".into()));
    }
    if buf[0..4] != MAGIC {
        return Err(VedaError::Storage(format!(
            "invalid magic bytes: expected {:?}, got {:?}",
            MAGIC,
            &buf[0..4]
        )));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(VedaError::Storage(format!(
            "unsupported version: expected {VERSION}, got {version}"
        )));
    }
    let dimension = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let count = u64::from_le_bytes(buf[12..20].try_into().unwrap());
    let data_type = u32::from_le_bytes(buf[20..24].try_into().unwrap());
    Ok((dimension, count, data_type))
}

// ── VectorStorageWriter ─────────────────────────────────────────────────────

/// Writes vectors to a file in the VEDA vector storage format.
///
/// Vectors are appended one at a time using buffered I/O. The header is
/// finalized with the correct count when [`close`] is called.
pub struct VectorStorageWriter {
    writer: BufWriter<File>,
    dimension: u32,
    count: u64,
}

impl VectorStorageWriter {
    /// Create a new vector storage file at `path` with the given vector dimension.
    pub fn create(path: &Path, dimension: u32) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        // Write initial header with count = 0; we will update it on close.
        write_header(&mut writer, dimension, 0, DATA_TYPE_F32)?;

        Ok(Self {
            writer,
            dimension,
            count: 0,
        })
    }

    /// Append a single vector. Returns the 0-indexed vector ID.
    ///
    /// # Errors
    /// Returns [`VedaError::DimensionMismatch`] if the vector length does not
    /// match the storage dimension.
    pub fn append(&mut self, vector: &[f32]) -> Result<u32> {
        if vector.len() != self.dimension as usize {
            return Err(VedaError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len() as u32,
            });
        }

        // Write raw f32 bytes (little-endian on LE platforms, which is the
        // standard target for this project).
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                std::mem::size_of_val(vector),
            )
        };
        self.writer.write_all(byte_slice)?;

        let id = self.count as u32;
        self.count += 1;
        Ok(id)
    }

    /// Flush buffered data to the underlying file.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Return the number of vectors written so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Finalize the file: update the header with the final count and close.
    pub fn close(mut self) -> Result<()> {
        self.writer.flush()?;

        // Seek back to the beginning and rewrite the header with the correct count.
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        write_header(file, self.dimension, self.count, DATA_TYPE_F32)?;
        file.flush()?;

        Ok(())
    }
}

// ── VectorStorageReader ─────────────────────────────────────────────────────

/// Memory-mapped reader for VEDA vector storage files.
///
/// Vectors are accessed as `&[f32]` slices pointing directly into mmap'd
/// memory, enabling zero-copy reads.
pub struct VectorStorageReader {
    mmap: Mmap,
    dimension: u32,
    count: u64,
}

impl VectorStorageReader {
    /// Open an existing vector storage file for reading.
    ///
    /// Validates magic bytes and version. The file body is memory-mapped.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;

        // Read header first to validate before mmapping.
        let metadata = file.metadata()?;
        if metadata.len() < HEADER_SIZE as u64 {
            return Err(VedaError::Storage("file too small for header".into()));
        }

        // SAFETY: The file is opened read-only and we validate the header
        // before exposing any data.
        let mmap = unsafe { Mmap::map(&file)? };
        let (dimension, count, _data_type) = parse_header(&mmap[..HEADER_SIZE])?;

        // Verify file size matches expected data.
        let expected_body = count * dimension as u64 * std::mem::size_of::<f32>() as u64;
        let actual_body = mmap.len() as u64 - HEADER_SIZE as u64;
        if actual_body < expected_body {
            return Err(VedaError::Storage(format!(
                "file body too small: expected {expected_body} bytes, got {actual_body}"
            )));
        }

        Ok(Self {
            mmap,
            dimension,
            count,
        })
    }

    /// The vector dimension.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// The number of vectors stored.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get a single vector by its 0-indexed ID, returning a slice into mmap'd memory.
    pub fn get(&self, id: u32) -> Result<&[f32]> {
        if id as u64 >= self.count {
            return Err(VedaError::Storage(format!(
                "vector id {id} out of range (count={})",
                self.count
            )));
        }
        Ok(self.get_unchecked(id))
    }

    /// Get multiple vectors by their IDs.
    pub fn get_batch(&self, ids: &[u32]) -> Result<Vec<&[f32]>> {
        ids.iter().map(|&id| self.get(id)).collect()
    }

    /// Iterate over all vectors in order.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.count as u32).map(move |id| self.get_unchecked(id))
    }

    // ── private ─────────────────────────────────────────────────────────

    fn get_unchecked(&self, id: u32) -> &[f32] {
        let dim = self.dimension as usize;
        let float_offset = id as usize * dim;
        let byte_offset = HEADER_SIZE + float_offset * std::mem::size_of::<f32>();
        let byte_len = dim * std::mem::size_of::<f32>();

        let bytes = &self.mmap[byte_offset..byte_offset + byte_len];

        // SAFETY: The VEDA format stores tightly packed f32 arrays and the
        // header starts at an aligned offset (32 bytes). f32 requires 4-byte
        // alignment; we verify the pointer alignment at runtime.
        let ptr = bytes.as_ptr();
        assert!(
            (ptr as usize).is_multiple_of(std::mem::align_of::<f32>()),
            "mmap data not aligned for f32"
        );
        unsafe { std::slice::from_raw_parts(ptr as *const f32, dim) }
    }
}
