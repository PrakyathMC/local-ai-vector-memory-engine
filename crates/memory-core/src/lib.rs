//! Domain types for the Local AI Vector Memory Engine.
//!
//! This crate intentionally contains no database, model, HTTP, or Python code.
//! It defines the contracts that those layers will share.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A stable identifier for one memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(String);

impl MemoryId {
    pub fn new(value: impl Into<String>) -> Result<Self, MemoryValidationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(MemoryValidationError::EmptyId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The application meaning assigned to a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    Document,
    Fact,
    Decision,
    Event,
    Note,
    Procedure,
}

/// Whether a memory participates in normal retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryStatus {
    Active,
    Archived,
    Deleted,
}

/// Information needed to interpret an embedding correctly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProvenance {
    pub model: String,
    pub dimensions: usize,
    pub normalized: bool,
}

/// One piece of information that the engine can persist and retrieve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub content: String,
    pub embedding: Vec<f32>,
    pub source: Option<String>,
    pub memory_type: MemoryType,
    pub namespace: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub status: MemoryStatus,
    pub metadata: BTreeMap<String, String>,
    pub embedding_provenance: EmbeddingProvenance,
}

impl MemoryRecord {
    pub fn validate(&self) -> Result<(), MemoryValidationError> {
        if self.content.trim().is_empty() {
            return Err(MemoryValidationError::EmptyContent);
        }
        if self.namespace.trim().is_empty() {
            return Err(MemoryValidationError::EmptyNamespace);
        }
        if self.embedding.is_empty() {
            return Err(MemoryValidationError::EmptyEmbedding);
        }
        if self.embedding.len() != self.embedding_provenance.dimensions {
            return Err(MemoryValidationError::EmbeddingDimensionMismatch {
                expected: self.embedding_provenance.dimensions,
                actual: self.embedding.len(),
            });
        }
        if self.updated_at_ms < self.created_at_ms {
            return Err(MemoryValidationError::InvalidTimestampOrder);
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MemoryValidationError {
    #[error("memory ID cannot be empty")]
    EmptyId,
    #[error("memory content cannot be empty")]
    EmptyContent,
    #[error("memory namespace cannot be empty")]
    EmptyNamespace,
    #[error("memory embedding cannot be empty")]
    EmptyEmbedding,
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
    #[error("updated_at_ms cannot be earlier than created_at_ms")]
    InvalidTimestampOrder,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_memory() -> MemoryRecord {
        MemoryRecord {
            id: MemoryId::new("memory-1").unwrap(),
            content: "We use Rust for the retrieval core.".into(),
            embedding: vec![0.1, 0.2, 0.3],
            source: Some("docs/decision.md".into()),
            memory_type: MemoryType::Decision,
            namespace: "default".into(),
            created_at_ms: 1,
            updated_at_ms: 1,
            status: MemoryStatus::Active,
            metadata: BTreeMap::new(),
            embedding_provenance: EmbeddingProvenance {
                model: "test-embedder".into(),
                dimensions: 3,
                normalized: false,
            },
        }
    }

    #[test]
    fn valid_memory_passes_validation() {
        assert_eq!(valid_memory().validate(), Ok(()));
    }

    #[test]
    fn empty_id_is_rejected() {
        assert_eq!(
            MemoryId::new("  "),
            Err(MemoryValidationError::EmptyId)
        );
    }

    #[test]
    fn embedding_dimension_mismatch_is_rejected() {
        let mut memory = valid_memory();
        memory.embedding.pop();

        assert_eq!(
            memory.validate(),
            Err(MemoryValidationError::EmbeddingDimensionMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn timestamps_must_be_ordered() {
        let mut memory = valid_memory();
        memory.updated_at_ms = 0;

        assert_eq!(
            memory.validate(),
            Err(MemoryValidationError::InvalidTimestampOrder)
        );
    }
}
