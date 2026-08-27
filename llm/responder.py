"""LLM response generation via Ollama.

Builds a grounded product-catalog prompt and calls Ollama's chat API using the
configured LLM (``llama3`` or ``mistral``). Returns a polite fallback when no
context matches without invoking the LLM.
"""

import logging
import os

import requests

from pipeline.embedder import OllamaUnavailableError

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")

SYSTEM_PROMPT = (
    "You are a helpful product catalog assistant. Answer the user's question "
    "using ONLY the product information provided below. If the answer is not "
    "in the context, say \"I don't have information about that in our "
    "catalog.\" Do not make up product details."
)

NO_RESULTS_MESSAGE = (
    "I couldn't find any products matching that in our catalog. "
    "Try asking by product name, category, SKU, or price range."
)


def _build_prompt(query: str, chunks: list) -> str:
    """Compose the user prompt that injects the retrieved context."""
    context = "\n\n".join(
        f"- {c['text']} (source: {c['metadata'].get('name', 'unknown')})"
        for c in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer based only on the context above."
    )


def generate_response(query: str, chunks: list) -> str:
    """Generate an answer for a query using the retrieved chunks.

    Args:
        query: The user's question.
        chunks: Top merged chunks used as context.

    Returns:
        The assistant's answer string.

    Raises:
        OllamaUnavailableError: if Ollama cannot be reached.
    """
    if not chunks:
        logger.info("No context for query; returning fallback message.")
        return NO_RESULTS_MESSAGE

    prompt = _build_prompt(query, chunks)
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Cannot reach LLM at {OLLAMA_HOST}: {exc}"
        ) from exc

    if resp.status_code != 200:
        logger.error("Ollama chat error (HTTP %s): %s",
                     resp.status_code, resp.text[:300])
        raise OllamaUnavailableError(
            f"Ollama chat error (HTTP {resp.status_code})"
        )

    data = resp.json()
    message = data.get("message", {}).get("content", "")
    return message.strip() or "I couldn't generate an answer."
