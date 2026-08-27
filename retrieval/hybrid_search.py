"""Hybrid retrieval combining semantic + keyword search.

Semantic results are weighted at 0.7, keyword results at 0.3, deduplicated by
chunk id, and the merged top results are returned for LLM context.
"""

import logging
import re

from pipeline.indexer import Indexer
from pipeline.embedder import embed_query, OllamaUnavailableError

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
TOP_K = 5

# Patterns for direct SQL/in-memory style filters (SKU lookup, price ranges).
_SKU_RE = re.compile(r"(?i)\bsku[: ]?([a-z0-9\-]+)")
_PRICE_UNDER_RE = re.compile(r"(?i)\bunder\s+\$?(\d+(?:\.\d+)?)")
_PRICE_OVER_RE = re.compile(r"(?i)\b(?:over|above|more than)\s+\$?(\d+(?:\.\d+)?)")


class HybridSearcher:
    """Coordinates semantic and keyword retrieval against the index."""

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # In-memory product table for direct SKU/price filtering.
        self.products = []

    def set_products(self, products: list):
        """Keep a reference to products for direct filtering."""
        self.products = products

    def search(self, query: str) -> list:
        """Run hybrid search and return the top merged chunks.

        Args:
            query: The raw user query.

        Returns:
            List of top chunks, each::

                {
                    "chunk_id": str,
                    "text": str,
                    "metadata": dict,
                    "score": float,
                }
        """
        merged = {}
        terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]

        # --- Semantic search (weight 0.7) ---
        try:
            qvec = embed_query(query)
            for row in self.indexer.semantic_search(qvec, n_results=10):
                score = SEMANTIC_WEIGHT
                _merge(merged, row, score)
        except OllamaUnavailableError as exc:
            logger.warning("Semantic search skipped: %s", exc)

        # --- Keyword search (weight 0.3) ---
        if terms:
            for row in self.indexer.keyword_search(terms[:3], n_results=20):
                _merge(merged, row, KEYWORD_WEIGHT)

        # --- Direct in-memory filters for SKU / price ---
        matched = self._direct_filter(query)
        for row in matched:
            # Boosting direct identity/filter hits slightly.
            score = KEYWORD_WEIGHT + 0.2
            key = row["chunk_id"]
            if key in merged:
                merged[key]["score"] = max(merged[key]["score"], score)
            else:
                merged[key] = {"chunk_id": key, "text": row["text"],
                               "metadata": row["metadata"], "score": score}

        results = sorted(merged.values(), key=lambda r: r["score"],
                         reverse=True)
        return results[:TOP_K]

    def _direct_filter(self, query: str) -> list:
        """Match SKU and price-range queries against the in-memory table."""
        hits = []
        sku_m = _SKU_RE.search(query)
        if sku_m and self.products:
            target = sku_m.group(1).upper()
            for p in self.products:
                if str(p.get("sku", "")).upper() == target:
                    hits.append(self._row_for(p, "identity"))
                    hits.append(self._row_for(p, "pricing"))

        under = _PRICE_UNDER_RE.search(query) or _PRICE_OVER_RE.search(query)
        if under and self.products:
            limit = float(under.group(1))
            for p in self.products:
                price = p.get("price")
                if price is not None and float(price) <= limit:
                    hits.append(self._row_for(p, "pricing"))
        return hits

    def _row_for(self, product, chunk_type: str) -> dict:
        """Build a search-result-like dict from an in-memory product."""
        from pipeline.chunker import build_chunks
        product_id = product.get("id") or product.get("sku") or ""
        for chunk in build_chunks(product):
            if chunk["chunk_type"] == chunk_type:
                return {
                    "chunk_id": f"{product_id}_{chunk_type}",
                    "text": chunk["text"],
                    "metadata": {
                        k: chunk[k] for k in (
                            "product_id", "sku", "name", "category",
                            "price", "stock", "chunk_type",
                        )
                    },
                }
        return {"chunk_id": str(product.get("sku")), "text": str(product),
                "metadata": {"name": product.get("name")}}


def _merge(target: dict, row: dict, score: float):
    """Merge a row into the map, updating score on duplicates."""
    key = row["chunk_id"]
    if key in target:
        target[key]["score"] += score
        target[key]["text"] = row["text"]
    else:
        target[key] = {
            "chunk_id": key,
            "text": row["text"],
            "metadata": row["metadata"],
            "score": score,
        }
