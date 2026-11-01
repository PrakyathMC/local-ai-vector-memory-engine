//! The first Python boundary for the memory engine.
//!
//! This module intentionally exposes only a small validated object. Storage,
//! embeddings, and retrieval will be added in later sprints.

use std::collections::BTreeMap;

use memory_core::{EmbeddingProvenance, MemoryId, MemoryRecord, MemoryStatus, MemoryType};
use memory_retrieval::search;
use memory_store::MemoryStore;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct Memory {
    record: MemoryRecord,
}

impl Memory {
    fn from_record(record: MemoryRecord) -> Self {
        Self { record }
    }
}

#[pymethods]
impl Memory {
    #[new]
    #[pyo3(signature = (id, content, embedding, namespace="default".to_string(), model="test-embedder".to_string(), source=None, metadata=None, created_at_ms=0, updated_at_ms=0, normalized=false))]
    fn new(
        id: String,
        content: String,
        embedding: Vec<f32>,
        namespace: String,
        model: String,
        source: Option<String>,
        metadata: Option<BTreeMap<String, String>>,
        created_at_ms: u64,
        updated_at_ms: u64,
        normalized: bool,
    ) -> PyResult<Self> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let dimensions = embedding.len();
        let record = MemoryRecord {
            id,
            content,
            embedding,
            source,
            memory_type: MemoryType::Note,
            namespace,
            created_at_ms,
            updated_at_ms,
            status: MemoryStatus::Active,
            metadata: metadata.unwrap_or_default(),
            embedding_provenance: EmbeddingProvenance {
                model,
                dimensions,
                normalized,
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
    fn source(&self) -> Option<&str> {
        self.record.source.as_deref()
    }

    #[getter]
    fn dimensions(&self) -> usize {
        self.record.embedding.len()
    }

    #[getter]
    fn embedding(&self) -> Vec<f32> {
        self.record.embedding.clone()
    }

    #[getter]
    fn metadata(&self) -> BTreeMap<String, String> {
        self.record.metadata.clone()
    }

    #[getter]
    fn created_at_ms(&self) -> u64 {
        self.record.created_at_ms
    }

    #[getter]
    fn updated_at_ms(&self) -> u64 {
        self.record.updated_at_ms
    }

    #[getter]
    fn normalized(&self) -> bool {
        self.record.embedding_provenance.normalized
    }

    #[getter]
    fn embedding_model(&self) -> &str {
        &self.record.embedding_provenance.model
    }

    #[getter]
    fn status(&self) -> &'static str {
        match self.record.status {
            MemoryStatus::Active => "active",
            MemoryStatus::Archived => "archived",
            MemoryStatus::Deleted => "deleted",
        }
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

/// Python access to the Rust-backed SQLite memory store.
#[pyclass(name = "MemoryStore", unsendable)]
pub struct Store {
    inner: MemoryStore,
}

#[pymethods]
impl Store {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let inner =
            MemoryStore::open(path).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn insert(&self, memory: PyRef<'_, Memory>) -> PyResult<()> {
        self.inner
            .insert(&memory.record)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn get(&self, id: String) -> PyResult<Option<Memory>> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let record = self
            .inner
            .get(&id)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(record.map(Memory::from_record))
    }

    fn list(&self, namespace: Option<String>) -> PyResult<Vec<Memory>> {
        let records = self
            .inner
            .list(namespace.as_deref())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(records.into_iter().map(Memory::from_record).collect())
    }

    #[pyo3(signature = (query, top_k=5, namespace=None))]
    fn search(
        &self,
        query: Vec<f32>,
        top_k: usize,
        namespace: Option<String>,
    ) -> PyResult<Vec<SearchResult>> {
        let hits = search(&self.inner, &query, top_k, namespace.as_deref())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(hits.into_iter().map(SearchResult::from_hit).collect())
    }

    fn update(&self, memory: PyRef<'_, Memory>) -> PyResult<()> {
        self.inner
            .update(&memory.record)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn archive(&self, id: String) -> PyResult<()> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .archive(&id)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn delete(&self, id: String) -> PyResult<()> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .delete(&id)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn restore(&self, id: String) -> PyResult<()> {
        let id = MemoryId::new(id).map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.inner
            .restore(&id)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyclass]
pub struct SearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub namespace: String,
    #[pyo3(get)]
    pub source: Option<String>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub embedding_model: String,
    #[pyo3(get)]
    pub dimensions: usize,
    #[pyo3(get)]
    pub metadata: BTreeMap<String, String>,
}

impl SearchResult {
    fn from_hit(hit: memory_retrieval::SearchHit) -> Self {
        Self {
            id: hit.memory.id.as_str().to_owned(),
            content: hit.memory.content,
            score: hit.score,
            namespace: hit.memory.namespace,
            source: hit.memory.source,
            status: match hit.memory.status {
                MemoryStatus::Active => "active".into(),
                MemoryStatus::Archived => "archived".into(),
                MemoryStatus::Deleted => "deleted".into(),
            },
            embedding_model: hit.memory.embedding_provenance.model,
            dimensions: hit.memory.embedding_provenance.dimensions,
            metadata: hit.memory.metadata,
        }
    }
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id={:?}, score={:.4}, namespace={:?})",
            self.id, self.score, self.namespace
        )
    }
}

#[pymodule]
fn local_ai_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<Store>()?;
    m.add_class::<SearchResult>()?;
    Ok(())
}
