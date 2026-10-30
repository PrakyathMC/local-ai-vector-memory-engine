use std::collections::BTreeMap;

use memory_core::{EmbeddingProvenance, MemoryId, MemoryRecord, MemoryStatus, MemoryType};
use memory_store::MemoryStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/demo-memory.db".to_string());

    let store = MemoryStore::open(&path)?;
    let record = MemoryRecord {
        id: MemoryId::new("demo-memory-1")?,
        content: "SQLite keeps this memory after the process exits.".into(),
        embedding: vec![0.1, 0.2, 0.3],
        source: Some("demo".into()),
        memory_type: MemoryType::Note,
        namespace: "default".into(),
        created_at_ms: 1,
        updated_at_ms: 1,
        status: MemoryStatus::Active,
        metadata: BTreeMap::from([(String::from("purpose"), String::from("storage-demo"))]),
        embedding_provenance: EmbeddingProvenance {
            model: "demo-embedder".into(),
            dimensions: 3,
            normalized: false,
        },
    };

    store.insert(&record)?;
    println!("Created database: {path}");
    println!("Stored memory ID: {}", record.id.as_str());
    Ok(())
}
