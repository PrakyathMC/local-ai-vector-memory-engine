//! Low-level diagnostics for a local memory database.

use std::path::PathBuf;

use memory_core::MemoryStatus;
use memory_store::{MemoryStore, StoreError};
use serde_json::json;

#[derive(Debug)]
enum Command {
    Inspect,
    List { namespace: Option<String> },
}

fn parse_args() -> Result<(PathBuf, Command), String> {
    let mut args = std::env::args().skip(1);
    let mut db = PathBuf::from("data/ingestion-demo.db");
    let mut command = None;
    let mut namespace = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--db" => {
                db = PathBuf::from(args.next().ok_or("--db requires a path")?);
            }
            "inspect" => command = Some(Command::Inspect),
            "list" => command = Some(Command::List { namespace: None }),
            "--namespace" => namespace = Some(args.next().ok_or("--namespace requires a value")?),
            "--help" | "-h" => {
                println!("memory-engine [--db PATH] inspect");
                println!("memory-engine [--db PATH] list [--namespace NAME]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let command = match command {
        Some(Command::List { .. }) => Command::List { namespace },
        Some(command) => command,
        None => return Err("expected inspect or list command".into()),
    };
    Ok((db, command))
}

fn run(db: PathBuf, command: Command) -> Result<(), StoreError> {
    let store = MemoryStore::open(&db)?;

    match command {
        Command::Inspect => {
            let memories = store.list(None)?;
            let active = memories
                .iter()
                .filter(|memory| memory.status == MemoryStatus::Active)
                .count();
            let archived = memories
                .iter()
                .filter(|memory| memory.status == MemoryStatus::Archived)
                .count();
            let deleted = memories
                .iter()
                .filter(|memory| memory.status == MemoryStatus::Deleted)
                .count();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "database": db,
                    "schema_version": store.schema_version()?,
                    "distance_metric": store.distance_metric()?,
                    "embedding_configuration": store.embedding_configuration()?,
                    "memories": memories.len(),
                    "active": active,
                    "archived": archived,
                    "deleted": deleted,
                }))
                .expect("diagnostic JSON is serializable")
            );
        }
        Command::List { namespace } => {
            for memory in store.list(namespace.as_deref())? {
                println!(
                    "{}\t{:?}\t{}\t{}\t{}",
                    memory.id.as_str(),
                    memory.status,
                    memory.namespace,
                    memory.source.unwrap_or_else(|| "-".into()),
                    memory.content.replace(['\r', '\n'], " ")
                );
            }
        }
    }
    Ok(())
}

fn main() {
    let (db, command) = match parse_args() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("argument error: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(db, command) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
