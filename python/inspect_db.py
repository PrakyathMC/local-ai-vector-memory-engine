"""Print the schema and human-readable rows from a local memory database."""

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", nargs="?", default="data/demo-memory.db")
    args = parser.parse_args()

    with sqlite3.connect(args.database) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        print("Tables:", [name for (name,) in tables])

        schema = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'memories'"
        ).fetchone()
        print("\nSchema:\n", schema[0] if schema else "<missing>")

        rows = connection.execute(
            "SELECT id, content, namespace, memory_type, status, "
            "embedding_dimensions, embedding_model FROM memories"
        ).fetchall()
        print("\nMemories:")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
