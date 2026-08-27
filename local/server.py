"""Local stdlib HTTP server for the Product Catalog Chatbot demo.

Runs without FastAPI / ChromaDB / Ollama so it works on this Android/PRoot box.
Serves a small chat UI and JSON endpoints mirroring the production app:
    GET  /chatbot/            -> chat UI
    POST /chatbot/chat        -> {"query": ...} -> {"response","sources"}
    GET  /health              -> {"status","indexed_chunks"}
    POST /chatbot/reindex     -> force re-ingest
"""

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from local.store import LocalStore
from local.answerer import answer, NO_RESULTS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local-chatbot")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PROJECT_DIR = BASE_DIR.parent
SQL_CANDIDATES = [
    Path("/storage/emulated/0/Download/job apply/data.sql"),
    PROJECT_DIR / "data" / "data.sql",
    PROJECT_DIR / "data" / "sample_data.sql",
]


def _find_sql() -> Path:
    for cand in SQL_CANDIDATES:
        if cand.is_file():
            return cand
    return SQL_CANDIDATES[-1]


def build_index():
    """Load -> chunk -> store once. Returns (LocalStore, products, chunks)."""
    from data.loader import load_products
    from pipeline.chunker import chunk_all

    sql_path = _find_sql()
    logger.info("Loading products from %s", sql_path)
    products = load_products(str(sql_path))
    logger.info("Chunking %d products", len(products))
    chunks = chunk_all(products)
    store = LocalStore(products, chunks)
    logger.info("Ready: %d chunks indexed", len(chunks))
    return store, products, chunks


def _sources(rows):
    seen, out = set(), []
    for r in rows:
        n = r["metadata"].get("name")
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalCatalogChatbot/1.0"

    def log_message(self, fmt, *args):  # keep logs tidy
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, relpath, ctype):
        path = STATIC_DIR / relpath
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/chatbot", "/chatbot/"):
            self._send_file("index.html", "text/html; charset=utf-8")
        elif path == "/chatbot/static/index.html":
            self._send_file("index.html", "text/html; charset=utf-8")
        elif path in ("/health", "/chatbot/health"):
            self._send_json({"status": "ok",
                             "indexed_chunks": self.server.chunk_count})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/chatbot/chat", "/chat"):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        query = (req.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query must not be empty"}, 422)
            return

        rows = self.server.store.search(query)
        if not rows:
            self._send_json({"response": NO_RESULTS, "sources": []})
            return
        self._send_json({"response": answer(query, rows),
                         "sources": _sources(rows)})


def run(host="127.0.0.1", port=8000):
    store, products, chunks = build_index()

    server = ThreadingHTTPServer((host, port), Handler)
    server.store = store
    server.chunk_count = len(chunks)
    logger.info("Local chatbot server: http://%s:%d/chatbot/", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    run(host, port)
