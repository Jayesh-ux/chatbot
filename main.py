"""FastAPI entry point for the Product Catalog Chatbot.

Pipeline (load -> chunk -> embed -> store) runs once automatically at startup.
The app is served behind a ``/chatbot/`` prefix (via reverse proxy) so it is
reachable at http://penguin.linux.test/chatbot/.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data.loader import load_products
from pipeline.embedder import OllamaUnavailableError
from pipeline.indexer import Indexer
from retrieval.hybrid_search import HybridSearcher
from llm.responder import generate_response, NO_RESULTS_MESSAGE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot")

BASE_DIR = Path(__file__).resolve().parent
# Override with: CHATBOT_SQL=/path/to/data.sql
SQL_PATH = Path(os.environ.get(
    "CHATBOT_SQL",
    "/storage/emulated/0/Download/job apply/data.sql",
))
STATIC_DIR = BASE_DIR / "static"
PERSIST_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# State singletons
# ---------------------------------------------------------------------------
indexer = None
searcher = None
products = []
_INDEX_LOCK = threading.Lock()


def run_pipeline() -> None:
    """Run load -> chunk -> embed -> store once on startup."""
    global indexer, searcher, products
    with _INDEX_LOCK:
        try:
            products = load_products(str(SQL_PATH))
        except FileNotFoundError:
            logger.error("SQL file missing: %s. Falling back to sample data.",
                         SQL_PATH)
            sample = BASE_DIR / "data" / "sample_data.sql"
            if sample.exists():
                products = load_products(str(sample))
            else:
                logger.error("No sample data available either. Exiting.")
                raise SystemExit(1)
        except ValueError as exc:
            logger.error("Malformed SQL file: %s", exc)
            raise SystemExit(1)

        indexer = Indexer(str(PERSIST_DIR))
        if indexer.is_indexed():
            logger.info("Collection already indexed (%d chunks) — skipping "
                        "re-index.", indexer.count)
        else:
            indexer.rebuild(products)

        searcher = HybridSearcher(indexer)
        searcher.set_products(products)
        logger.info("Pipeline complete. %d chunks indexed.", indexer.count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running data pipeline at startup ...")
    run_pipeline()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Product Catalog Chatbot", lifespan=lifespan)
app.mount("/chatbot/static", StaticFiles(directory=str(STATIC_DIR)),
          name="static")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ReindexRequest(BaseModel):
    pass


@app.get("/health")
def health():
    """Health check reporting indexed chunk count."""
    return {"status": "ok", "indexed_chunks": indexer.count if indexer else 0}


@app.get("/chatbot/")
def index_page():
    """Serve the single-page chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/")
def root():
    """Redirect root to the prefixed UI path."""
    return JSONResponse({"message": "Go to /chatbot/"}, status_code=200)


@app.post("/chatbot/chat")
@app.post("/chat")
def chat(req: ChatRequest):
    """Accept a user query and return the generated answer plus sources."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")

    if searcher is None:
        raise HTTPException(status_code=503, detail="Index not ready")

    # If no semantic engine (Ollama down), still allow direct keyword matches.
    try:
        chunks = searcher.search(req.query)
    except OllamaUnavailableError as exc:
        logger.warning("Search error (maybe Ollama down): %s", exc)
        return JSONResponse(status_code=503, content={
            "error": "Ollama is not available",
            "detail": str(exc),
        })

    if not chunks:
        return {"response": NO_RESULTS_MESSAGE, "sources": []}

    try:
        answer = generate_response(req.query, chunks)
    except OllamaUnavailableError as exc:
        logger.error("LLM unavailable: %s", exc)
        return JSONResponse(status_code=503, content={
            "error": "Ollama is not available",
            "detail": str(exc),
        })

    sources = _unique_sources(chunks)
    return {"response": answer, "sources": sources}


@app.post("/chatbot/reindex")
@app.post("/reindex")
def reindex():
    """Force a fresh data ingestion (for demo purposes)."""
    global products, searcher, indexer
    with _INDEX_LOCK:
        logger.info("Reindexing ...")
        products = load_products(str(SQL_PATH))
        indexer.rebuild(products)
        searcher = HybridSearcher(indexer)
        searcher.set_products(products)
    return {"status": "reindexed", "indexed_chunks": indexer.count}


def _unique_sources(chunks: list) -> list:
    """Return unique product names from the merged chunks, in order."""
    seen, out = set(), []
    for c in chunks:
        name = c["metadata"].get("name")
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
