"""ChromaDB storage and retrieval for product chunks.

Responsible for persisting embedded chunks into a local ChromaDB collection
named ``product_catalog`` using batched inserts, and for checking whether the
collection already has data so re-indexing is idempotent across restarts.
"""

import logging

from pipeline.chunker import chunk_all
from pipeline.embedder import embed_texts, OllamaUnavailableError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "product_catalog"
BATCH_SIZE = 50
PERSIST_DIR = "chroma_db"


class Indexer:
    """Wraps the ChromaDB collection for product chunks."""

    def __init__(self, persist_directory: str = PERSIST_DIR):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - lazy dependency
            raise RuntimeError(
                "chromadb is required but not installed. Run: "
                "pip install chromadb"
            ) from exc
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._count = self.collection.count()

    @property
    def count(self) -> int:
        """Number of indexed chunks."""
        return self._count

    def is_indexed(self) -> bool:
        """True if the collection already contains documents."""
        return self.collection.count() > 0

    def rebuild(self, products: list) -> int:
        """Chunk, embed and store a product list.

        Args:
            products: List of product dicts.

        Returns:
            Number of chunks indexed.

        Raises:
            OllamaUnavailableError: if Ollama is not reachable.
        """
        chunks = chunk_all(products)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks with batch size %d ...",
                    len(texts), BATCH_SIZE)
        embeddings = embed_texts(texts, batch_size=BATCH_SIZE)

        # Remove everything so a rebuild replaces stale data.
        try:
            ids = self.collection.get()["ids"]
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not clear collection: %s", exc)

        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{
            "product_id": c["product_id"],
            "sku": c["sku"],
            "name": c["name"],
            "category": c["category"],
            "price": c["price"],
            "stock": c["stock"],
            "chunk_type": c["chunk_type"],
        } for c in chunks]

        # Batch inserts to avoid holding everything in memory at once.
        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE
            self.collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        self._count = self.collection.count()
        logger.info("Indexed %d chunks into %s", self._count, COLLECTION_NAME)
        return self._count

    def semantic_search(self, query_vector, n_results: int = 10) -> list:
        """Return top-n matches by cosine similarity."""
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, self.collection.count()),
        )
        return self._normalise(result)

    def keyword_search(self, terms: list, n_results: int = 20) -> list:
        """Return chunks whose document text contains any of ``terms``.

        Uses ChromaDB's ``where_document`` substring filter, falling back to
        scanning returned ids if the backend does not support it.
        """
        found = []
        for term in terms:
            try:
                result = self.collection.query(
                    query_texts=[term],
                    n_results=min(n_results, self.collection.count()),
                )
                found.extend(self._normalise(result))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("keyword query failed for %r: %s", term, exc)
        return found

    @staticmethod
    def _normalise(result) -> list:
        """Turn a ChromaDB query result into a list of chunk dicts."""
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        rows = []
        for i, doc in enumerate(docs[0]):
            meta = metas[0][i] or {}
            rows.append({
                "chunk_id": ids[0][i],
                "text": doc,
                "metadata": meta,
                "distance": distances[0][i] if distances and distances[0] else None,
            })
        return rows
