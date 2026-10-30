//! SQLite persistence for validated memory records.
//!
//! The database is deliberately boring at this stage: one memory record maps
//! to one SQLite row. Vector indexing and retrieval will be added later.

use std::collections::BTreeMap;
use std::path::Path;

use memory_core::{
    EmbeddingProvenance, MemoryId, MemoryRecord, MemoryStatus, MemoryType, MemoryValidationError,
};
use rusqlite::{Connection, OptionalExtension, params};
use thiserror::Error;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT,
    memory_type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_normalized INTEGER NOT NULL
)
"#;

/// A local persistent memory store.
pub struct MemoryStore {
    connection: Connection,
}

impl MemoryStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let connection = Connection::open(path)?;
        Self::from_connection(connection)
    }

    pub fn in_memory() -> Result<Self, StoreError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(connection: Connection) -> Result<Self, StoreError> {
        connection.execute_batch(SCHEMA)?;
        Ok(Self { connection })
    }

    pub fn insert(&self, record: &MemoryRecord) -> Result<(), StoreError> {
        record.validate()?;

        let metadata_json = serde_json::to_string(&record.metadata)?;
        self.connection.execute(
            r#"
            INSERT INTO memories (
                id, content, embedding, source, memory_type, namespace,
                created_at_ms, updated_at_ms, status, metadata_json,
                embedding_model, embedding_dimensions, embedding_normalized
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
            params![
                record.id.as_str(),
                record.content,
                pack_embedding(&record.embedding),
                record.source,
                memory_type_to_str(record.memory_type),
                record.namespace,
                record.created_at_ms,
                record.updated_at_ms,
                status_to_str(record.status),
                metadata_json,
                record.embedding_provenance.model,
                record.embedding_provenance.dimensions,
                record.embedding_provenance.normalized,
            ],
        )?;

        Ok(())
    }

    pub fn get(&self, id: &MemoryId) -> Result<Option<MemoryRecord>, StoreError> {
        let record = self
            .connection
            .query_row(
                "SELECT id, content, embedding, source, memory_type, namespace, created_at_ms, updated_at_ms, status, metadata_json, embedding_model, embedding_dimensions, embedding_normalized FROM memories WHERE id = ?1",
                params![id.as_str()],
                row_to_memory,
            )
            .optional()?;

        Ok(record)
    }

    pub fn list(&self, namespace: Option<&str>) -> Result<Vec<MemoryRecord>, StoreError> {
        let mut records = Vec::new();
        let mut statement = match namespace {
            Some(_) => self.connection.prepare(
                "SELECT id, content, embedding, source, memory_type, namespace, created_at_ms, updated_at_ms, status, metadata_json, embedding_model, embedding_dimensions, embedding_normalized FROM memories WHERE namespace = ?1 ORDER BY created_at_ms, id",
            )?,
            None => self.connection.prepare(
                "SELECT id, content, embedding, source, memory_type, namespace, created_at_ms, updated_at_ms, status, metadata_json, embedding_model, embedding_dimensions, embedding_normalized FROM memories ORDER BY created_at_ms, id",
            )?,
        };

        let rows = match namespace {
            Some(namespace) => statement.query_map(params![namespace], row_to_memory)?,
            None => statement.query_map([], row_to_memory)?,
        };

        for row in rows {
            records.push(row?);
        }
        Ok(records)
    }

    pub fn update(&self, record: &MemoryRecord) -> Result<(), StoreError> {
        record.validate()?;
        let metadata_json = serde_json::to_string(&record.metadata)?;
        let changed = self.connection.execute(
            r#"
            UPDATE memories SET
                content = ?1,
                embedding = ?2,
                source = ?3,
                memory_type = ?4,
                namespace = ?5,
                updated_at_ms = ?6,
                status = ?7,
                metadata_json = ?8,
                embedding_model = ?9,
                embedding_dimensions = ?10,
                embedding_normalized = ?11
            WHERE id = ?12
            "#,
            params![
                record.content,
                pack_embedding(&record.embedding),
                record.source,
                memory_type_to_str(record.memory_type),
                record.namespace,
                record.updated_at_ms,
                status_to_str(record.status),
                metadata_json,
                record.embedding_provenance.model,
                record.embedding_provenance.dimensions,
                record.embedding_provenance.normalized,
                record.id.as_str(),
            ],
        )?;

        if changed == 0 {
            return Err(StoreError::NotFound(record.id.as_str().to_string()));
        }
        Ok(())
    }

    pub fn archive(&self, id: &MemoryId) -> Result<(), StoreError> {
        self.set_status(id, MemoryStatus::Archived)
    }

    pub fn delete(&self, id: &MemoryId) -> Result<(), StoreError> {
        self.set_status(id, MemoryStatus::Deleted)
    }

    fn set_status(&self, id: &MemoryId, status: MemoryStatus) -> Result<(), StoreError> {
        let changed = self.connection.execute(
            "UPDATE memories SET status = ?1 WHERE id = ?2",
            params![status_to_str(status), id.as_str()],
        )?;
        if changed == 0 {
            return Err(StoreError::NotFound(id.as_str().to_string()));
        }
        Ok(())
    }
}

fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let id: String = row.get(0)?;
    let embedding_bytes: Vec<u8> = row.get(2)?;
    let metadata_json: String = row.get(9)?;
    let metadata: BTreeMap<String, String> =
        serde_json::from_str(&metadata_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                9,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;

    let record = MemoryRecord {
        id: MemoryId::new(id).map_err(to_sql_error)?,
        content: row.get(1)?,
        embedding: unpack_embedding(&embedding_bytes).map_err(to_sql_error)?,
        source: row.get(3)?,
        memory_type: memory_type_from_str(&row.get::<_, String>(4)?).map_err(to_sql_error)?,
        namespace: row.get(5)?,
        created_at_ms: row.get(6)?,
        updated_at_ms: row.get(7)?,
        status: status_from_str(&row.get::<_, String>(8)?).map_err(to_sql_error)?,
        metadata,
        embedding_provenance: EmbeddingProvenance {
            model: row.get(10)?,
            dimensions: row.get(11)?,
            normalized: row.get(12)?,
        },
    };

    record.validate().map_err(to_sql_error)?;
    Ok(record)
}

fn pack_embedding(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn unpack_embedding(bytes: &[u8]) -> Result<Vec<f32>, StoreError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(StoreError::InvalidEmbeddingBytes);
    }

    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn memory_type_to_str(value: MemoryType) -> &'static str {
    match value {
        MemoryType::Document => "document",
        MemoryType::Fact => "fact",
        MemoryType::Decision => "decision",
        MemoryType::Event => "event",
        MemoryType::Note => "note",
        MemoryType::Procedure => "procedure",
    }
}

fn memory_type_from_str(value: &str) -> Result<MemoryType, StoreError> {
    match value {
        "document" => Ok(MemoryType::Document),
        "fact" => Ok(MemoryType::Fact),
        "decision" => Ok(MemoryType::Decision),
        "event" => Ok(MemoryType::Event),
        "note" => Ok(MemoryType::Note),
        "procedure" => Ok(MemoryType::Procedure),
        other => Err(StoreError::UnknownMemoryType(other.to_string())),
    }
}

fn status_to_str(value: MemoryStatus) -> &'static str {
    match value {
        MemoryStatus::Active => "active",
        MemoryStatus::Archived => "archived",
        MemoryStatus::Deleted => "deleted",
    }
}

fn status_from_str(value: &str) -> Result<MemoryStatus, StoreError> {
    match value {
        "active" => Ok(MemoryStatus::Active),
        "archived" => Ok(MemoryStatus::Archived),
        "deleted" => Ok(MemoryStatus::Deleted),
        other => Err(StoreError::UnknownStatus(other.to_string())),
    }
}

fn to_sql_error(error: impl Into<StoreError>) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error.into()))
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("memory validation error: {0}")]
    Validation(#[from] MemoryValidationError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid embedding bytes")]
    InvalidEmbeddingBytes,
    #[error("unknown memory type: {0}")]
    UnknownMemoryType(String),
    #[error("unknown memory status: {0}")]
    UnknownStatus(String),
    #[error("memory not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn record() -> MemoryRecord {
        let id = MemoryId::new("memory-1").unwrap();
        MemoryRecord {
            id,
            content: "SQLite keeps this memory after restart.".into(),
            embedding: vec![0.1, 0.2, 0.3],
            source: Some("docs/storage.md".into()),
            memory_type: MemoryType::Decision,
            namespace: "default".into(),
            created_at_ms: 100,
            updated_at_ms: 100,
            status: MemoryStatus::Active,
            metadata: BTreeMap::from([(String::from("topic"), String::from("persistence"))]),
            embedding_provenance: EmbeddingProvenance {
                model: "test-embedder".into(),
                dimensions: 3,
                normalized: false,
            },
        }
    }

    #[test]
    fn insert_and_read_round_trip() {
        let store = MemoryStore::in_memory().unwrap();
        let original = record();
        store.insert(&original).unwrap();

        let restored = store.get(&original.id).unwrap().unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn missing_memory_returns_none() {
        let store = MemoryStore::in_memory().unwrap();
        let id = MemoryId::new("missing").unwrap();

        assert_eq!(store.get(&id).unwrap(), None);
    }

    #[test]
    fn invalid_record_is_not_inserted() {
        let store = MemoryStore::in_memory().unwrap();
        let mut invalid = record();
        invalid.embedding.pop();

        assert!(matches!(
            store.insert(&invalid),
            Err(StoreError::Validation(
                MemoryValidationError::EmbeddingDimensionMismatch { .. }
            ))
        ));
    }

    #[test]
    fn file_store_survives_close_and_reopen() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("local-ai-memory-{unique}.db"));
        let original = record();

        {
            let store = MemoryStore::open(&path).unwrap();
            store.insert(&original).unwrap();
        }

        {
            let reopened = MemoryStore::open(&path).unwrap();
            let restored = reopened.get(&original.id).unwrap().unwrap();
            assert_eq!(restored, original);
        }

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn update_archive_delete_and_list_work() {
        let store = MemoryStore::in_memory().unwrap();
        let original = record();
        store.insert(&original).unwrap();

        let mut updated = original.clone();
        updated.content = "Updated content.".into();
        updated.updated_at_ms = 200;
        store.update(&updated).unwrap();
        assert_eq!(
            store.get(&original.id).unwrap().unwrap().content,
            "Updated content."
        );

        store.archive(&original.id).unwrap();
        assert_eq!(
            store.get(&original.id).unwrap().unwrap().status,
            MemoryStatus::Archived
        );

        store.delete(&original.id).unwrap();
        assert_eq!(
            store.get(&original.id).unwrap().unwrap().status,
            MemoryStatus::Deleted
        );
        assert_eq!(store.list(Some("default")).unwrap().len(), 1);
    }
}
