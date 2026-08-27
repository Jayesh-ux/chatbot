"""Vercel serverless entry point for the Product Catalog Chatbot.

Deploys the stdlib-only local engine (same loader/chunker/hybrid-search logic)
as a stateless-but-warm WSGI application. The index is built once on first
request and cached at module level so warm instances reuse it.
"""

import json
import logging
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "local")):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vercel-chatbot")

from local.store import LocalStore              # noqa: E402
from local.answerer import answer, NO_RESULTS   # noqa: E402

_CACHE_LOCK = threading.Lock()
_store = None
_chunk_count = 0

# SQL candidates inside the deployment bundle.
SQL_CANDIDATES = [
    ROOT / "data" / "data.sql",
    ROOT / "data" / "sample_data.sql",
]
SQL_CANDIDATES = [c for c in SQL_CANDIDATES if c.is_file()]


def _load_index():
    """Build and cache the index (thread-safe)."""
    global _store, _chunk_count
    if _store is not None:
        return _store
    with _CACHE_LOCK:
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


# --- tiny WSGI helpers ----------------------------------------------------
def _json_response(start_response, obj, status="200 OK"):
    body = json.dumps(obj).encode("utf-8")
    start_response(status, [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ])
    return [body]


_HTML = None


def _html_page():
    global _HTML
    if _HTML is None:
        p = ROOT / "static" / "index.html"
        _HTML = p.read_bytes() if p.is_file() else b"<h1>Chatbot</h1>"
    return _HTML


def application(environ, start_response):
    """WSGI app: GET / -> UI, POST /api/chat -> chat, GET /api/health."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = urlparse(environ.get("PATH_INFO", "/")).path

    if method == "GET":
        if path in ("/", "/chatbot", "/chatbot/", "/index.html"):
            body = _html_page()
            start_response("200 OK", [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ])
            return [body]
        if path in ("/health", "/api/health"):
            try:
                _load_index()
            except Exception as exc:  # pragma: no cover
                return _json_response(start_response,
                                      {"status": "error", "detail": str(exc)})
            return _json_response(start_response,
                                  {"status": "ok", "indexed_chunks": _chunk_count})
        return _json_response(start_response, {"error": "not found"}, "404 Not Found")

    if method == "POST" and path in ("/api/chat", "/chatbot/chat", "/chat"):
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            req = json.loads(environ["wsgi.input"].read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return _json_response(start_response, {"error": "invalid JSON"},
                                  "400 Bad Request")
        query = (req.get("query") or "").strip()
        if not query:
            return _json_response(start_response,
                                  {"error": "query must not be empty"},
                                  "422 Unprocessable Entity")
        try:
            store = _load_index()
        except Exception as exc:  # pragma: no cover
            logger.error("index build failed: %s", exc)
            return _json_response(start_response,
                                  {"error": "index unavailable: %s" % exc}, "500")
        rows = store.search(query)
        if not rows:
            return _json_response(start_response,
                                  {"response": NO_RESULTS, "sources": []})
        return _json_response(start_response,
                              {"response": answer(query, rows),
                               "sources": _sources(rows)})

    return _json_response(start_response, {"error": "not found"}, "404 Not Found")


app = application
