//! Exact vector retrieval for the local memory engine.
//!
//! This is intentionally a brute-force baseline: every active memory is
//! compared with the query. It is easy to understand and gives us a trusted
//! result before we introduce an approximate index such as HNSW.

use hnsw_rs::prelude::{DistCosine, Hnsw};
use memory_core::{MemoryRecord, MemoryStatus, MemoryType};
use memory_store::{MemoryStore, StoreError};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub memory: MemoryRecord,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub top_k: usize,
    pub namespace: Option<String>,
    pub source: Option<String>,
    pub memory_type: Option<MemoryType>,
    pub status: Option<MemoryStatus>,
    pub min_created_at_ms: Option<u64>,
    pub max_created_at_ms: Option<u64>,
    pub query_model: Option<String>,
}

/// An in-memory HNSW index over a snapshot of active memory records.
///
/// SQLite remains the source of truth. This index is a derived acceleration
/// structure and must be rebuilt after records change in this first version.
pub struct HnswIndex {
    index: Hnsw<'static, f32, DistCosine>,
    memories: Vec<MemoryRecord>,
    dimensions: usize,
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
    #[error("query embedding dimensions do not match the database: expected {expected}, received {actual}")]
    QueryDimensionMismatch { expected: usize, actual: usize },
    #[error(
        "query embedding model does not match the database: expected {expected}, received {actual}"
    )]
    QueryModelMismatch { expected: String, actual: String },
    #[error("memory store error: {0}")]
    Store(#[from] StoreError),
}

#[derive(Debug, Error)]
pub enum HnswError {
    #[error("query embedding cannot be empty")]
    EmptyQuery,
    #[error("query and index embeddings have different dimensions: {query} vs {index}")]
    DimensionMismatch { query: usize, index: usize },
    #[error("query embedding cannot be all zeros")]
    ZeroQuery,
    #[error("memory store error: {0}")]
    Store(#[from] StoreError),
}

/// Common result type for interchangeable exact and approximate indexes.
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("exact index error: {0}")]
    Exact(#[from] SearchError),
    #[error("HNSW index error: {0}")]
    Hnsw(#[from] HnswError),
}

/// Small common interface used to compare index implementations.
///
/// The exact implementation remains the correctness oracle. HNSW implements
/// the same minimal query shape so benchmark and evaluation code can compare
/// them without depending on their internal data structures.
pub trait VectorIndex {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, IndexError>;
}

/// Exact index adapter over the SQLite source of truth.
pub struct ExactIndex<'a> {
    store: &'a MemoryStore,
}

impl<'a> ExactIndex<'a> {
    pub fn new(store: &'a MemoryStore) -> Self {
        Self { store }
    }
}

impl VectorIndex for ExactIndex<'_> {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, IndexError> {
        Ok(search_with_options(
            self.store,
            query,
            SearchOptions {
                top_k,
                ..SearchOptions::default()
            },
        )?)
    }
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
    search_with_options(
        store,
        query,
        SearchOptions {
            top_k,
            namespace: namespace.map(str::to_owned),
            ..SearchOptions::default()
        },
    )
}

/// Search using candidate filters and embedding configuration validation.
pub fn search_with_options(
    store: &MemoryStore,
    query: &[f32],
    options: SearchOptions,
) -> Result<Vec<SearchHit>, SearchError> {
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }
    if options.top_k == 0 {
        return Ok(Vec::new());
    }

    if let Some((expected_model, expected_dimensions)) = store.embedding_configuration()? {
        if query.len() != expected_dimensions {
            return Err(SearchError::QueryDimensionMismatch {
                expected: expected_dimensions,
                actual: query.len(),
            });
        }
        if let Some(actual_model) = options.query_model.as_deref() {
            if actual_model != expected_model {
                return Err(SearchError::QueryModelMismatch {
                    expected: expected_model,
                    actual: actual_model.to_owned(),
                });
            }
        }
    }

    let expected_status = options.status.unwrap_or(MemoryStatus::Active);
    let memories = if expected_status == MemoryStatus::Active {
        store.list_retrievable(options.namespace.as_deref(), now_ms())?
    } else {
        store.list(options.namespace.as_deref())?
    };
    let mut hits = Vec::new();

    for memory in memories {
        if memory.status != expected_status {
            continue;
        }
        if let Some(source) = options.source.as_deref() {
            if memory.source.as_deref() != Some(source) {
                continue;
            }
        }
        if let Some(memory_type) = options.memory_type {
            if memory.memory_type != memory_type {
                continue;
            }
        }
        if let Some(min_created_at_ms) = options.min_created_at_ms {
            if memory.created_at_ms < min_created_at_ms {
                continue;
            }
        }
        if let Some(max_created_at_ms) = options.max_created_at_ms {
            if memory.created_at_ms > max_created_at_ms {
                continue;
            }
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
    hits.truncate(options.top_k);
    Ok(hits)
}

impl HnswIndex {
    pub fn build(records: &[MemoryRecord]) -> Result<Self, HnswError> {
        let memories: Vec<MemoryRecord> = records
            .iter()
            .filter(|record| record.status == MemoryStatus::Active)
            .cloned()
            .collect();
        let dimensions = memories
            .first()
            .map(|memory| memory.embedding.len())
            .unwrap_or(0);

        for memory in &memories {
            if memory.embedding.len() != dimensions {
                return Err(HnswError::DimensionMismatch {
                    query: memory.embedding.len(),
                    index: dimensions,
                });
            }
        }

        let index = Hnsw::new(16, memories.len().max(1), 16, 200, DistCosine {});
        for (index_id, memory) in memories.iter().enumerate() {
            index.insert((&memory.embedding, index_id));
        }

        Ok(Self {
            index,
            memories,
            dimensions,
        })
    }

    pub fn build_from_store(store: &MemoryStore) -> Result<Self, HnswError> {
        Ok(Self::build(&store.list_retrievable(None, now_ms())?)?)
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Search the snapshot. `ef_search` controls the accuracy/speed trade-off.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchHit>, HnswError> {
        if query.is_empty() {
            return Err(HnswError::EmptyQuery);
        }
        if self.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimensions {
            return Err(HnswError::DimensionMismatch {
                query: query.len(),
                index: self.dimensions,
            });
        }

        let norm = query
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        if norm == 0.0 {
            return Err(HnswError::ZeroQuery);
        }

        let neighbours =
            self.index
                .search(query, top_k.min(self.memories.len()), ef_search.max(top_k));
        let mut hits = neighbours
            .into_iter()
            .filter_map(|neighbour| {
                self.memories
                    .get(neighbour.d_id)
                    .cloned()
                    .map(|memory| SearchHit {
                        memory,
                        score: 1.0 - neighbour.distance,
                    })
            })
            .collect::<Vec<_>>();

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.memory.id.as_str().cmp(right.memory.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

impl VectorIndex for HnswIndex {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, IndexError> {
        Ok(self.search(query, top_k, 64)?)
    }
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
        let mut deleted = memory("deleted", vec![1.0, 0.0]);
        deleted.status = MemoryStatus::Deleted;
        store.insert(&deleted).unwrap();

        let hits = search(&store, &[1.0, 0.0], 5, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory.id.as_str(), "exact");
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[1].memory.id.as_str(), "different");
    }

    #[test]
    fn empty_store_returns_no_hits_and_ties_are_deterministic() {
        let empty_store = MemoryStore::in_memory().unwrap();
        assert!(search(&empty_store, &[1.0, 0.0], 5, None)
            .unwrap()
            .is_empty());

        let store = MemoryStore::in_memory().unwrap();
        store.insert(&memory("b", vec![1.0, 0.0])).unwrap();
        store.insert(&memory("a", vec![1.0, 0.0])).unwrap();
        let hits = search(&store, &[1.0, 0.0], 5, None).unwrap();
        assert_eq!(hits[0].memory.id.as_str(), "a");
        assert_eq!(hits[1].memory.id.as_str(), "b");
    }

    #[test]
    fn search_applies_namespace_source_type_and_time_filters() {
        let store = MemoryStore::in_memory().unwrap();
        let mut target = memory("target", vec![1.0, 0.0]);
        target.source = Some("notes/rust.md".into());
        target.memory_type = MemoryType::Document;
        target.namespace = "project-a".into();
        target.created_at_ms = 100;
        target.updated_at_ms = 100;

        let mut other = memory("other", vec![1.0, 0.0]);
        other.source = Some("notes/python.md".into());
        other.namespace = "project-b".into();
        other.created_at_ms = 200;
        other.updated_at_ms = 200;

        store.insert_many(&[target, other]).unwrap();
        let hits = search_with_options(
            &store,
            &[1.0, 0.0],
            SearchOptions {
                top_k: 5,
                namespace: Some("project-a".into()),
                source: Some("notes/rust.md".into()),
                memory_type: Some(MemoryType::Document),
                min_created_at_ms: Some(100),
                max_created_at_ms: Some(100),
                ..SearchOptions::default()
            },
        )
        .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.id.as_str(), "target");
    }

    #[test]
    fn search_validates_query_configuration() {
        let store = MemoryStore::in_memory().unwrap();
        store.insert(&memory("one", vec![1.0, 0.0])).unwrap();

        assert!(matches!(
            search_with_options(
                &store,
                &[1.0],
                SearchOptions {
                    top_k: 1,
                    ..SearchOptions::default()
                }
            ),
            Err(SearchError::QueryDimensionMismatch { .. })
        ));

        assert!(matches!(
            search_with_options(
                &store,
                &[1.0, 0.0],
                SearchOptions {
                    top_k: 1,
                    query_model: Some("wrong-model".into()),
                    ..SearchOptions::default()
                }
            ),
            Err(SearchError::QueryModelMismatch { .. })
        ));
    }

    #[test]
    fn hnsw_returns_the_nearest_active_memory() {
        let records = vec![
            memory("near", vec![1.0, 0.0]),
            memory("far", vec![0.0, 1.0]),
        ];
        let index = HnswIndex::build(&records).unwrap();
        let hits = index.search(&[1.0, 0.0], 2, 32).unwrap();

        assert_eq!(index.len(), 2);
        assert_eq!(hits[0].memory.id.as_str(), "near");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn exact_and_hnsw_share_a_comparable_index_interface() {
        let store = MemoryStore::in_memory().unwrap();
        store.insert(&memory("near", vec![1.0, 0.0])).unwrap();
        store.insert(&memory("far", vec![0.0, 1.0])).unwrap();

        let exact = ExactIndex::new(&store).search(&[1.0, 0.0], 2).unwrap();
        let hnsw_index = HnswIndex::build_from_store(&store).unwrap();
        let hnsw = VectorIndex::search(&hnsw_index, &[1.0, 0.0], 2).unwrap();

        assert_eq!(exact[0].memory.id, hnsw[0].memory.id);
    }
}
