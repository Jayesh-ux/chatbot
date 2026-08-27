"""Hosted LLM response generation via Google Gemini for the serverless app.

Mirrors the logic of the production ``llm/responder.py`` (which calls Ollama):
the same grounded system prompt, the same context injection from retrieved
chunks, and the same polite fallback without calling the LLM when nothing is
retrieved. Only the transport differs — this calls Google's public Gemini API
so it runs on Vercel, where a local Ollama server cannot.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = os.environ.get(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
)

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


class LLMUnavailableError(Exception):
    """Raised when the Gemini API cannot be reached or returns an error."""


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
    """Generate a grounded answer using the retrieved chunks.

    Args:
        query: The user's question.
        chunks: Top merged chunks used as context.

    Returns:
        The assistant's answer string.

    Raises:
        LLMUnavailableError: if the Gemini API cannot be reached or the model
            returns no usable text.
    """
    if not chunks:
        logger.info("No context for query; returning fallback message.")
        return NO_RESULTS_MESSAGE

    if not GEMINI_API_KEY:
        raise LLMUnavailableError("GEMINI_API_KEY is not set")

    prompt = _build_prompt(query, chunks)
    url = (f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
           f"?key={GEMINI_API_KEY}")
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {"temperature": 0.2},
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise LLMUnavailableError(f"Cannot reach Gemini API: {exc}") from exc

    if resp.status_code != 200:
        logger.error("Gemini HTTP %s: %s", resp.status_code, resp.text[:300])
        raise LLMUnavailableError(f"Gemini API error (HTTP {resp.status_code})")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        logger.error("Unexpected Gemini response shape: %s",
                     str(data)[:300])
        raise LLMUnavailableError("Gemini returned no usable text")
