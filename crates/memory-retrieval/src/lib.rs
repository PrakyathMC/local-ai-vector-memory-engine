//! Exact vector retrieval for the local memory engine.
//!
//! This is intentionally a brute-force baseline: every active memory is
//! compared with the query. It is easy to understand and gives us a trusted
//! result before we introduce an approximate index such as HNSW.

use memory_core::{MemoryRecord, MemoryStatus};
use memory_store::{MemoryStore, StoreError};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub memory: MemoryRecord,
    pub score: f32,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("query embedding cannot be empty")]
    EmptyQuery,
    #[error("query embedding cannot be all zeros")]
    ZeroQuery,
    #[error("query and memory embeddings have different dimensions: {query} vs {memory}")]
    DimensionMismatch { query: usize, memory: usize },
    #[error("memory embedding cannot be all zeros: {0}")]
    ZeroMemory(String),
    #[error("memory store error: {0}")]
    Store(#[from] StoreError),
}

/// Calculate cosine similarity between two embeddings.
pub fn cosine_similarity(query: &[f32], memory: &[f32]) -> Result<f32, SearchError> {
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }
    if query.len() != memory.len() {
        return Err(SearchError::DimensionMismatch {
            query: query.len(),
            memory: memory.len(),
        });
    }

    let mut dot = 0.0_f64;
    let mut query_norm = 0.0_f64;
    let mut memory_norm = 0.0_f64;

    for (&query_value, &memory_value) in query.iter().zip(memory) {
        let query_value = f64::from(query_value);
        let memory_value = f64::from(memory_value);
        dot += query_value * memory_value;
        query_norm += query_value * query_value;
        memory_norm += memory_value * memory_value;
    }

    if query_norm == 0.0 {
        return Err(SearchError::ZeroQuery);
    }
    if memory_norm == 0.0 {
        return Err(SearchError::ZeroMemory("unknown".into()));
    }

    Ok((dot / (query_norm.sqrt() * memory_norm.sqrt())) as f32)
}

/// Search active memories and return the highest-scoring results first.
pub fn search(
    store: &MemoryStore,
    query: &[f32],
    top_k: usize,
    namespace: Option<&str>,
) -> Result<Vec<SearchHit>, SearchError> {
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }
    if top_k == 0 {
        return Ok(Vec::new());
    }

    let memories = store.list(namespace)?;
    let mut hits = Vec::new();

    for memory in memories {
        if memory.status != MemoryStatus::Active {
            continue;
        }

        let score = cosine_similarity(query, &memory.embedding).map_err(|error| match error {
            SearchError::ZeroMemory(_) => SearchError::ZeroMemory(memory.id.as_str().to_owned()),
            other => other,
        })?;
        hits.push(SearchHit { memory, score });
    }

    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.memory.id.as_str().cmp(right.memory.id.as_str()))
    });
    hits.truncate(top_k);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_core::{EmbeddingProvenance, MemoryId, MemoryType};
    use std::collections::BTreeMap;

    fn memory(id: &str, embedding: Vec<f32>) -> MemoryRecord {
        MemoryRecord {
            id: MemoryId::new(id).unwrap(),
            content: format!("Content for {id}"),
            embedding: embedding.clone(),
            source: None,
            memory_type: MemoryType::Note,
            namespace: "default".into(),
            created_at_ms: 0,
            updated_at_ms: 0,
            status: MemoryStatus::Active,
            metadata: BTreeMap::new(),
            embedding_provenance: EmbeddingProvenance {
                model: "test".into(),
                dimensions: embedding.len(),
                normalized: false,
            },
        }
    }

    #[test]
    fn cosine_similarity_handles_identical_and_orthogonal_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap().abs() < 1e-6);
    }

    #[test]
    fn search_ranks_results_and_excludes_archived_memories() {
        let store = MemoryStore::in_memory().unwrap();
        store.insert(&memory("exact", vec![1.0, 0.0])).unwrap();
        store.insert(&memory("different", vec![0.0, 1.0])).unwrap();
        let mut archived = memory("archived", vec![1.0, 0.0]);
        archived.status = MemoryStatus::Archived;
        store.insert(&archived).unwrap();

        let hits = search(&store, &[1.0, 0.0], 5, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory.id.as_str(), "exact");
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].memory.id.as_str(), "different");
    }
}
