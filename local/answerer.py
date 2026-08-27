"""Local rule-based answerer.

No Ollama/LLM runs on this box, so answers are assembled deterministically
from the retrieved product chunks plus direct filters. The questions the user
can ask (and that we can answer truthfully) are things like product names,
SKUs, and price ranges — the same queries the production LLM is grounded on.
"""

import re
from local.vectorizer import tokenize

NO_RESULTS = ("I couldn't find any products matching that in our catalog. "
              "Try asking by product name, SKU, or price range such as "
              "\"products under $50\".")

_PRICE_UNDER_RE = re.compile(r"(?i)\bunder\s+\$?(\d+(?:\.\d+)?)")
_PRICE_OVER_RE = re.compile(r"(?i)\b(?:over|above|more than)\s+\$?(\d+(?:\.\d+)?)")
_SKU_RE = re.compile(r"(?i)\bsku[: ]?([a-z0-9\-]+)")


def _money(v):
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _price_limit(query) -> float | None:
    for r in (_PRICE_UNDER_RE, _PRICE_OVER_RE):
        m = r.search(query)
        if m:
            return float(m.group(1))
    return None


def _format_pricing(row):
    md = row["metadata"]
    return (f"{md.get('name')} (SKU {md.get('sku')}) — "
            f"{_money(md.get('price'))}")


def answer(query: str, rows: list) -> str:
    """Build a natural answer from the retrieved rows (no LLM needed)."""
    if not rows:
        return NO_RESULTS

    # Price range question: list matches with their prices.
    limit = _price_limit(query)
    if limit is not None and _PRICE_UNDER_RE.search(query):
        priced = _dedupe_rows([r for r in rows
                               if r["metadata"].get("price") is not None])
        priced = sorted(priced, key=lambda r: float(r["metadata"]["price"]))[:5]
        line = "\n".join(f"  • {_format_pricing(r)}" for r in priced)
        return (f"Here are products at or under {_money(limit)}:\n{line}" if line
                else NO_RESULTS)

    # SKU lookup: show the single requested product in detail.
    sku_m = _SKU_RE.search(query)
    if sku_m:
        target = sku_m.group(1).upper()
        for r in rows:
            if str(r["metadata"].get("sku", "")).upper() == target:
                md = r["metadata"]
                desc = (md.get("description") or "").strip()
                out = (f"{md.get('name')} (SKU {md.get('sku')}) — "
                       f"{_money(md.get('price'))}")
                if desc:
                    out += f"\n\n{desc}"
                return out
        return NO_RESULTS

    # Generic: show top matches by name + price.
    names = _dedupe([r["metadata"].get("name") for r in rows
                     if r["metadata"].get("name")])
    if not names:
        return NO_RESULTS
    return ("Here are products I found in the catalog:\n" +
            "\n".join(f"  • {n}" for n in names[:5]))


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _dedupe_rows(rows):
    """Dedupe search-result rows by product (sku) identity."""
    seen, out = set(), []
    for r in rows:
        key = r["metadata"].get("sku")
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out
