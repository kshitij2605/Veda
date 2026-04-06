use thiserror::Error;

#[derive(Error, Debug)]
pub enum VedaError {
    #[error("Invalid schema: {0}")]
    InvalidSchema(String),

    #[error("Field not found: {0}")]
    FieldNotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Modality not supported: model '{model}' does not support '{modality}'")]
    ModalityNotSupported { model: String, modality: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, VedaError>;
