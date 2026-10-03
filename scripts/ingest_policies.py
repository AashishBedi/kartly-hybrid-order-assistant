from pathlib import Path

from app.config import settings
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.ingest import doc_id_from_filename, ingest_document
from app.rag.store import PolicyStore


POLICIES_DIR = Path(__file__).parents[1] / "policies"


def main() -> None:
    store = PolicyStore(settings.CHROMA_PATH)
    embedder = SentenceTransformerEmbedder(settings.EMBEDDING_MODEL)

    for policy_path in sorted(POLICIES_DIR.glob("*.md")):
        doc_id = doc_id_from_filename(policy_path.name)
        text = policy_path.read_text(encoding="utf-8")
        result = ingest_document(
            doc_id=doc_id,
            text=text,
            store=store,
            embedder=embedder,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        print(
            f"doc_id={doc_id}, status={result['status']}, "
            f"chunks={result['chunks']}"
        )

    print(f"total chunks={store.count()}")


if __name__ == "__main__":
    main()
