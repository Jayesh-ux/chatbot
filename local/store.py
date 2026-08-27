"""In-memory vector store + hybrid retrieval for the local demo.

Replicates the production hybrid search (semantic 0.7 + keyword 0.3, merged and
deduped by chunk_id, plus direct SKU / price-range filters) but backed by a
plain Python list instead of ChromaDB.
"""

import re

from local.vectorizer import vectorize, cosine

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
TOP_K = 5

_SKU_RE = re.compile(r"(?i)\bsku[: ]?([a-z0-9\-]+)")
_PRICE_UNDER_RE = re.compile(r"(?i)\bunder\s+\$?(\d+(?:\.\d+)?)")
_PRICE_OVER_RE = re.compile(r"(?i)\b(?:over|above|more than)\s+\$?(\d+(?:\.\d+)?)")


class LocalStore:
    """Indexes product chunks and answers hybrid-retrieval queries."""

    def __init__(self, products: list, chunks: list):
        self.products = products
        # Normalize chunker output into {chunk_id, text, metadata} shape.
        self.chunks = []
        for c in chunks:
            self.chunks.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": {k: c.get(k) for k in (
                    "product_id", "sku", "name", "category", "price",
                    "stock", "chunk_type")},
            })
        # Precompute vectors once at startup.
        self._vecs = [vectorize(c["text"]) for c in self.chunks]

    # -- searching ----------------------------------------------------------
    def semantic_search(self, query: str, n: int = 10) -> list:
        """Rank chunks by cosine similarity, keeping only real overlaps."""
        qv = vectorize(query)
        scored = [(cosine(qv, self._vecs[i]), self.chunks[i])
                  for i in range(len(self.chunks))]
        scored.sort(key=lambda x: x[0], reverse=True)
        # Drop chunks with no real term overlap (cosine ~ 0) so irrelevant
        # queries don't surface arbitrary products.
        return [self._hits(chunk, s)
                for s, chunk in scored[:n] if s > 0.0]

    def keyword_search(self, query: str, n: int = 20) -> list:
        """Rank chunks by raw token overlap (no embedding)."""
        qterms = set(vectorize(query).keys())
        scored = []
        for i, chunk in enumerate(self.chunks):
            cterms = set(self._vecs[i].keys())
            overlap = len(qterms & cterms)
            if overlap:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._hits(chunk, s) for s, chunk in scored[:n]]

    def direct_filter(self, query: str) -> list:
        """SKU lookup and price-range queries against the product table."""
        hits = []
        sku_m = _SKU_RE.search(query)
        if sku_m:
            target = sku_m.group(1).upper()
            for p in self.products:
                if str(p.get("sku", "")).upper() == target:
                    hits.append(self._row_for(p, "identity"))
                    hits.append(self._row_for(p, "pricing"))
                    hits.append(self._row_for(p, "summary"))

        under = _PRICE_UNDER_RE.search(query) or _PRICE_OVER_RE.search(query)
        if under:
            limit = float(under.group(1))
            for p in self.products:
                price = p.get("price")
                if price is not None and float(price) <= limit:
                    hits.append(self._row_for(p, "pricing"))
        return hits

    # -- hybrid merge -------------------------------------------------------
    def search(self, query: str) -> list:
        """Merge semantic + keyword + direct-filter results, dedup by chunk."""
        merged = {}
        for row in self.semantic_search(query):
            _add(merged, row, SEMANTIC_WEIGHT)
        for row in self.keyword_search(query):
            _add(merged, row, KEYWORD_WEIGHT)
        for row in self.direct_filter(query):
            _add(merged, row, KEYWORD_WEIGHT + 0.2)

        results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
        # Relevance floor: require a genuine signal (a real keyword/match;
        # direct SKU/price boosts land well above this threshold).
        if not results or results[0]["score"] < 0.29:
            return []
        return results[:TOP_K]

    # -- helpers ------------------------------------------------------------
    def _hits(self, chunk: dict, base_score: float) -> dict:
        base_score = base_score if base_score > 0 else 0.0
        return {"chunk_id": chunk["chunk_id"], "text": chunk["text"],
                "metadata": chunk["metadata"], "score": base_score}

    def _row_for(self, product: dict, chunk_type: str) -> dict:
        from pipeline.chunker import build_chunks
        pid = product.get("id") or product.get("sku") or ""
        for chunk in build_chunks(product):
            if chunk["chunk_type"] == chunk_type:
                return {
                    "chunk_id": f"{pid}_{chunk_type}",
                    "text": chunk["text"],
                    "metadata": {k: chunk[k] for k in (
                        "product_id", "sku", "name", "category", "price",
                        "stock", "chunk_type")},
                    "score": 0.0,
                }
        return {"chunk_id": str(pid), "text": str(product),
                "metadata": {"name": product.get("name")}, "score": 0.0}


def _add(target: dict, row: dict, score: float):
    key = row["chunk_id"]
    if key in target:
        target[key]["score"] += score
        target[key]["text"] = row["text"]
    else:
        target[key] = {"chunk_id": key, "text": row["text"],
                       "metadata": row["metadata"], "score": score}
