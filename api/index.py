"""Vercel serverless entry point (FastAPI) for the Product Catalog Chatbot.

Runs the stdlib-only in-memory engine (same loader / chunker / hybrid-search /
answerer logic as the local app) on Vercel. ChromaDB & Ollama are not used here
because serverless runtimes have a read-only filesystem and no local model
server.

Endpoints:
    GET  /            -> chat UI (from static/index.html)
    POST /api/chat    -> {"query": ...} -> {"response","sources"}
    GET  /health      -> {"status","indexed_chunks"}
    GET  /api/health  -> same as /health
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "local")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from local.store import LocalStore              # noqa: E402
from local.answerer import answer, NO_RESULTS   # noqa: E402
from local.gemini_responder import (           # noqa: E402
    generate_response as llm_respond,
    LLMUnavailableError,
    NO_RESULTS_MESSAGE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-chatbot")

app = FastAPI(title="Product Catalog Chatbot")

_store = None
_chunk_count = 0

SQL_CANDIDATES = [c for c in (
    ROOT / "data" / "data.sql",
    ROOT / "data" / "sample_data.sql",
) if c.is_file()]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)


def _load_index():
    """Build and cache the index on first use."""
    global _store, _chunk_count
    if _store is not None:
        return _store
    if not SQL_CANDIDATES:
        raise RuntimeError("No SQL data bundled in this deployment")
    from data.loader import load_products
    from pipeline.chunker import chunk_all
    logger.info("Loading products from %s", SQL_CANDIDATES[0])
    products = load_products(str(SQL_CANDIDATES[0]))
    chunks = chunk_all(products)
    _store = LocalStore(products, chunks)
    _chunk_count = len(chunks)
    logger.info("Ready: %d products, %d chunks", len(products), _chunk_count)
    return _store


def _sources(rows):
    seen, out = set(), []
    for r in rows:
        n = r["metadata"].get("name")
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


@app.get("/")
@app.get("/index.html")
def index():
    return FileResponse(str(ROOT / "static" / "index.html"),
                        media_type="text/html")


@app.get("/health")
@app.get("/api/health")
def health():
    try:
        _load_index()
        return {"status": "ok", "indexed_chunks": _chunk_count}
    except Exception as exc:  # pragma: no cover
        return JSONResponse(status_code=500,
                            content={"status": "error", "detail": str(exc)})


@app.post("/api/chat")
@app.post("/chat")
@app.post("/chatbot/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")
    try:
        store = _load_index()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500,
                            detail="Index unavailable: %s" % exc)
    rows = store.search(req.query)
    if not rows:
        return {"response": NO_RESULTS, "sources": []}
    # Prefer a real LLM answer grounded on the retrieved product context.
    try:
        response = llm_respond(req.query, rows)
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable, using retrieval answer: %s", exc)
        response = answer(req.query, rows)
    return {"response": response, "sources": _sources(rows)}
