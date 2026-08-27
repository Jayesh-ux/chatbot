# Product Catalog Chatbot

A retrieval-augmented (RAG) product catalog chatbot. It loads a SQL product dump,
chunks each product into focused field-level chunks, embeds them with Ollama
(`nomic-embed-text`), stores them in a local ChromaDB collection, and answers
questions with an LLM (`llama3`/`mistral`) grounded on hybrid semantic + keyword
retrieval.

## Structure

```
chatbot/
├── main.py                  FastAPI entry point (endpoints, startup pipeline)
├── data/
│   ├── loader.py            SQL import + flat/Magento-EAV parsing
│   └── sample_data.sql      fallback dataset when no real dump is found
├── pipeline/
│   ├── chunker.py           Field-level chunking (identity/pricing/desc/summary)
│   ├── embedder.py          Ollama nomic-embed-text embeddings
│   └── indexer.py           ChromaDB storage + retrieval (idempotent startup)
├── retrieval/
│   └── hybrid_search.py     Semantic (0.7) + keyword (0.3), merge/dedup
├── llm/
│   └── responder.py         Ollama chat call + grounded prompt builder
├── static/
│   └── index.html           Single-page chat UI (no external deps)
├── chroma_db/               persisted ChromaDB data (created at runtime)
└── requirements.txt
```

## The real dataset

The loader auto-detects and supports both a simple flat `products` table and a
**Magento 2 / MariaDB EAV dump** (the `data.sql` from `~/Download/job apply/`).
For EAV it reconstructs products from `catalog_product_entity` plus the
`catalog_product_entity_{varchar,text,decimal}` value tables using Magento's
default attribute ids (name=73, description=75, short_description=76, price=77,
special_price=78, upc=190, url_key=126, vendor=448), merged on `entity_id`.

That dump contains ~8,000 products. Note: it has **no category table** and **no
inventory rows**, so category defaults to `Uncategorized` and stock is absent from
those records.

**Override the data path:**
```
CHATBOT_SQL=/path/to/data.sql uvicorn main:app ...
```

## Requirements (run on the target machine, e.g. Chrome OS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (native on ChromeOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Install & start Ollama with the models:
   ```bash
   ollama pull nomic-embed-text      # embeddings
   ollama pull llama3                # or mistral
   ollama serve
   ```
2. Start the app (bind all interfaces so it's reachable by hostname):
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 80
   ```
   or `python main.py` (defaults to `0.0.0.0:80`).

3. The data pipeline (load → chunk → embed → store) runs **once on startup**.
   If the ChromaDB collection already has documents it skips re-embedding
   (idempotent). Rebuild on demand via `POST /reindex`.

## Run with Docker (Chrome OS / company PC)

```bash
# 1. Put your real dump in the project folder as data.sql
#    (e.g. copy /Documents/ChatBot/data.sql or your job-apply one here)
cp /path/to/data.sql ./data.sql

# 2. Build & start both containers (app + ollama)
docker compose up --build -d

# 3. One-time: pull the models into the running Ollama container
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3

# Open the UI
open http://localhost/chatbot/
```

- The app container talks to Ollama at `http://host.docker.internal:11434`.
- ChromaDB data persists in `./chroma_db`, Ollama models in `./ollama_data`.
- To re-run the pipeline: `curl -X POST http://localhost/reindex`.

## Access

Behind a reverse proxy that maps `/chatbot/` and sets `X-Forwarded-Prefix` to
`/chatbot/` (and proxies both HTTP and WebSocket/`Connection: upgrade` headers),
the UI is served at:

```
http://penguin.linux.test/chatbot/
```

## Endpoints

| Method | Path                    | Description                                   |
|--------|-------------------------|-----------------------------------------------|
| GET    | `/chatbot/`             | Serves the chat UI (index.html)               |
| POST   | `/chatbot/chat`         | `{"query":"..."}` → `{"response","sources"}`  |
| GET    | `/health`               | `{"status","indexed_chunks"}`                 |
| POST   | `/chatbot/reindex`      | Force a fresh data ingestion                  |

## Edge-case handling

- No matching context → polite "couldn't find" message (no LLM call).
- Ollama down → `HTTP 503` with a clear error.
- Missing/malformed SQL file → logs error and exits (or falls back to sample).
- Empty query → `422` validation error.
- Concurrent re-index is guarded by a lock.
