//! The first Python boundary for the memory engine.
//!
//! This module intentionally exposes only a small validated object. Storage,
//! embeddings, and retrieval will be added in later sprints.

use std::collections::BTreeMap;

use memory_core::{
    EmbeddingProvenance, MemoryId, MemoryRecord, MemoryStatus, MemoryType,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct Memory {
    record: MemoryRecord,
}

#[pymethods]
impl Memory {
    #[new]
    #[pyo3(signature = (id, content, embedding, namespace="default".to_string(), model="test-embedder".to_string()))]
    fn new(
        id: String,
        content: String,
        embedding: Vec<f32>,
        namespace: String,
        model: String,
    ) -> PyResult<Self> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let dimensions = embedding.len();
        let record = MemoryRecord {
            id,
            content,
            embedding,
            source: None,
            memory_type: MemoryType::Note,
            namespace,
            created_at_ms: 0,
            updated_at_ms: 0,
            status: MemoryStatus::Active,
            metadata: BTreeMap::new(),
            embedding_provenance: EmbeddingProvenance {
                model,
                dimensions,
                normalized: false,
            },
        };

        record
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self { record })
    }

    #[getter]
    fn id(&self) -> &str {
        self.record.id.as_str()
    }

    #[getter]
    fn content(&self) -> &str {
        &self.record.content
    }

    #[getter]
    fn namespace(&self) -> &str {
        &self.record.namespace
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.record.embedding.len()
    }

    #[getter]
    fn embedding_model(&self) -> &str {
        &self.record.embedding_provenance.model
    }

    fn validate(&self) -> PyResult<()> {
        self.record
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Memory(id={:?}, namespace={:?}, dimensions={})",
            self.id(),
            self.namespace(),
            self.dimensions()
        )
    }
}

#[pymodule]
fn local_ai_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    Ok(())
}
