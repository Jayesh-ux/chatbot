"""Field-level chunking strategy.

Instead of embedding an entire product row as one dense blob, each product is
split into several focused chunks so that price queries, description queries,
and SKU lookups each retrieve the most relevant chunk — rather than burying all
info in one dense vector that dilutes semantic relevance.
"""

import logging

logger = logging.getLogger(__name__)


def _money(value):
    """Format a price value as a readable string."""
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def build_chunks(product) -> list:
    """Build the list of chunk dicts for a single product.

    Each chunk dict carries::

        {
            "text": str,
            "product_id": ...,
            "sku": ...,
            "name": ...,
            "category": ...,
            "price": ...,
            "stock": ...,
            "chunk_type": "identity" | "pricing" | "description" | "summary",
        }

    Args:
        product: A product dict from the data loader.

    Returns:
        List of chunk dicts (may be of length 4, or fewer if fields are empty).
    """
    name = product.get("name") or "Unnamed"
    sku = product.get("sku") or "N/A"
    category = product.get("category") or "Uncategorized"
    price = _money(product.get("price"))
    stock = product.get("stock")
    stock_txt = str(stock) if stock is not None else "unknown"
    description = (product.get("description") or "").strip()

    # Justification for field-level chunking: price queries, description
    # queries, and SKU lookups each retrieve the most relevant focused chunk —
    # rather than burying all info in one dense vector that dilutes semantic
    # relevance across unrelated fields.
    chunks = []

    # 1. Identity chunk
    chunks.append({
        "text": (f"Product: {name}. SKU: {sku}. Category: {category}."),
        "chunk_type": "identity",
    })

    # 2. Pricing chunk
    chunks.append({
        "text": (f"Product: {name} (SKU: {sku}) costs {price}. "
                 f"Stock available: {stock_txt} units."),
        "chunk_type": "pricing",
    })

    # 3. Description chunk (skip if empty)
    if description:
        chunks.append({
            "text": f"Product: {name} — {description}",
            "chunk_type": "description",
        })

    # 4. Combined summary chunk
    summary = (
        f"Product Summary: {name} (SKU {sku}) is in the {category} category. "
        f"It costs {price} and currently has {stock_txt} units in stock."
    )
    if description:
        summary += f" Description: {description}"
    summary += "."
    chunks.append({
        "text": summary,
        "chunk_type": "summary",
    })

    # Attach shared metadata to every chunk.
    metadata = {
        "product_id": str(product.get("id") or product.get("sku") or ""),
        "sku": str(sku),
        "name": str(name),
        "category": str(category),
        "price": product.get("price"),
        "stock": product.get("stock"),
    }
    for chunk in chunks:
        chunk.update(metadata)

    return chunks


def chunk_all(products: list) -> list:
    """Chunk an entire list of products.

    Args:
        products: List of product dicts.

    Returns:
        Flat list of chunk dicts, each with a unique ``chunk_id`` added.
    """
    chunks = []
    for product in products:
        for chunk in build_chunks(product):
            # Stable id: <product_id>_<chunk_type>
            chunk["chunk_id"] = (
                f"{chunk['product_id']}_{chunk['chunk_type']}"
            )
            chunks.append(chunk)
    logger.info("Built %d chunks from %d products", len(chunks),
                len(products))
    return chunks
