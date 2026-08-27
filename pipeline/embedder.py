"""Ollama embedding generation.

Generates text embeddings using the ``nomic-embed-text`` model via Ollama's
HTTP API at ``http://localhost:11434/api/embeddings``.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


class OllamaUnavailableError(Exception):
    """Raised when the Ollama server cannot be reached or the model is missing."""


def embed_texts(texts: list, batch_size: int = 50) -> list:
    """Embed a list of texts, returning a list of vectors.

    Args:
        texts: List of strings to embed.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (each a list of floats), same order as input.

    Raises:
        OllamaUnavailableError: if Ollama is not reachable or errors out.
    """
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(_embed_batch(batch))
    return vectors


def embed_query(text: str):
    """Embed a single query string."""
    return _embed_batch([text])[0]


def _embed_batch(texts: list):
    url = f"{OLLAMA_HOST}/api/embeddings"
    out = []
    for text in texts:
        try:
            resp = requests.post(url, json={"model": EMBED_MODEL, "prompt": text},
                                 timeout=60)
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OllamaUnavailableError(
                f"Ollama embeddings error (HTTP {resp.status_code}): "
                f"{resp.text[:300]}"
            )
        data = resp.json()
        if "embedding" not in data:
            raise OllamaUnavailableError(
                f"Ollama returned no embedding for {EMBED_MODEL}"
            )
        out.append(data["embedding"])
    return out
