"""Download and verify the project's local embedding model."""

from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).parents[1] / "models"


def main() -> None:
    print(f"Loading {MODEL_NAME}")
    print(f"Local model directory: {MODEL_DIR}")

    model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_DIR))
    vector = model.encode("The retrieval core stores useful project memory.")

    print(f"Embedding dimensions: {len(vector)}")
    print(f"Model directory exists: {MODEL_DIR.exists()}")

    if len(vector) != 384:
        raise RuntimeError(f"Expected 384 dimensions, received {len(vector)}")

    print("Embedding model verified successfully.")


if __name__ == "__main__":
    main()
