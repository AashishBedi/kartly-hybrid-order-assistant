from pathlib import Path

import chromadb

from app.config import settings


class PolicyStore:
    def __init__(self, path: str | Path = settings.CHROMA_PATH) -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name="policy_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        version: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        ids = [f"{doc_id}:{version}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "version": version,
                "chunk_index": index,
                "title": chunk.splitlines()[0].strip(),
            }
            for index, chunk in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_other_versions(self, doc_id: str, keep_version: str) -> None:
        result = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        ids_to_delete = [
            chunk_id
            for chunk_id, metadata in zip(result["ids"], result["metadatas"])
            if metadata["version"] != keep_version
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

    def get_doc_version(self, doc_id: str) -> str | None:
        result = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
            include=["metadatas"],
        )
        if not result["metadatas"]:
            return None
        return str(result["metadatas"][0]["version"])

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        result_count = min(top_k, self.count())
        if result_count <= 0:
            return []

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": chunk_id,
                "doc_id": metadata["doc_id"],
                "text": document,
                "distance": distance,
            }
            for chunk_id, metadata, document, distance in zip(
                result["ids"][0],
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def count_for_doc(self, doc_id: str) -> int:
        result = self.collection.get(where={"doc_id": doc_id}, include=[])
        return len(result["ids"])

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return the stored text for specific chunk IDs."""
        if not chunk_ids:
            return {}
        result = self.collection.get(ids=chunk_ids, include=["documents"])
        return {
            str(chunk_id): str(document)
            for chunk_id, document in zip(result["ids"], result["documents"])
        }

    def list_doc_ids(self) -> list[str]:
        result = self.collection.get(include=["metadatas"])
        return sorted({str(metadata["doc_id"]) for metadata in result["metadatas"]})
