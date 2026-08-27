"""Pure-stdlib text vectorization + cosine similarity.

No numpy / chromadb / ollama available on this Android/PRoot box, so we use a
lightweight bag-of-words TF-style vector with a small tokenizer. This gives a
real, numeric semantic-ish retrieval signal so the local demo actually ranks by
relevance instead of just keyword overlap.
"""

import math
import re
import string

_STOPWORDS = set("""
a an and are as at be but by for from has have he her his i if in is it its
of on or our she so that the their them then there they this to was we were
what when where which who will with you your
""".split())

_PUNCT = string.punctuation


def tokenize(text: str) -> list:
    """Lowercase, split on non-alphanumerics, drop stopwords and empties."""
    text = text.translate(str.maketrans("", "", _PUNCT)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t and t not in _STOPWORDS]


def vectorize(text: str) -> dict:
    """Return a sparse bag-of-words dict {term: tf}."""
    vec = {}
    for tok in tokenize(text):
        vec[tok] = vec.get(tok, 0) + 1
    return vec


def _norm(vec: dict) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse vectors in [0, 1]."""
    if not a or not b:
        return 0.0
    denom = _norm(a) * _norm(b)
    if denom == 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for term, count in a.items():
        if term in b:
            dot += count * b[term]
    return dot / denom
