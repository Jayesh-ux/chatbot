"""SQL import and product-data parsing.

Turns a raw SQL dump (MySQL / MariaDB / SQLite flavored, including Magento 2
EAV schemas) into a normalized list of product dictionaries.

Two strategies are supported and auto-detected:

1. **Flat schema** — a single ``products`` style table with columns such as
   id, name, sku, description, price, category, stock.

2. **Magento 2 EAV schema** — the product information is spread over
   ``catalog_product_entity`` plus per-attribute value tables
   (``catalog_product_entity_varchar/text/int/decimal/datetime``) keyed by
   ``entity_id`` and Magento's numeric ``attribute_id`` codes.

The EAV strategy reconstructs flat records using these well-known Magento
attribute ids::

    name=73, description=75, short_description=76, price=77,
    special_price=78, url_key=126, upc=190, image=87, vendor=448
"""

import logging
import os
import re
import html
import functools

import sqlparse

logger = logging.getLogger(__name__)

try:
    from html import unescape as _html_unescape
except ImportError:  # pragma: no cover
    _html_unescape = lambda s: s

# Magento 2 default EAV attribute ids we care about.
_EAV_NAME = 73
_EAV_DESCRIPTION = 75
_EAV_SHORT_DESCRIPTION = 76
_EAV_PRICE = 77
_EAV_SPECIAL_PRICE = 78
_EAV_URL_KEY = 126
_EAV_UPC = 190
_EAV_IMAGE = 87
_EAV_VENDOR = 448

EAV_TABLE_NAMES = (
    "catalog_product_entity",
    "catalog_product_entity_varchar",
    "catalog_product_entity_text",
    "catalog_product_entity_int",
    "catalog_product_entity_decimal",
    "catalog_product_entity_datetime",
)

# Canonical field names we understand. Anything else is ignored.
KNOWN_FIELDS = {"id", "sku", "name", "title", "product_name",
                "description", "details", "price", "cost",
                "category", "stock", "quantity", "inventory"}

_PRICE_RE = re.compile(r"^\d+(\.\d+)?$")


def _normalise_column(col: str) -> str:
    """Map a raw column name to a canonical one, or None if unknown."""
    col = col.strip().strip("`\"'").lower().replace(" ", "_")
    if col in {"name", "title", "product_name", "product"}:
        return "name"
    if col in {"description", "details", "desc", "product_desc"}:
        return "description"
    if col in {"price", "cost", "unit_price", "retail_price"}:
        return "price"
    if col in {"stock", "quantity", "inventory", "stock_qty"}:
        return "stock"
    if col == "sku":
        return "sku"
    if col == "category":
        return "category"
    if col == "id":
        return "id"
    return None


def _parse_insert_statement(statement) -> list:
    """Extract rows from a single INSERT statement as a list of dicts."""
    tokens = re.split(r"\s+VALUES\s+", statement, flags=re.IGNORECASE)
    if len(tokens) < 2:
        return []

    # Column list: INSERT INTO tbl (col1, col2, ...) VALUES ...
    header = tokens[0]
    m = re.search(r"\((.*?)\)\s*$", header, flags=re.IGNORECASE)
    raw_cols = [c for c in (m.group(1).split(",") if m else []) if c.strip()]
    columns = [(_normalise_column(c), c) for c in raw_cols]

    # VALUES (...) might contain commas inside string literals, so we split
    # on the top-level parentheses rather than on commas.
    rows_blob = tokens[1].rstrip(";").strip()
    tuples = _split_top_level_tuples(rows_blob)

    products = []
    for tup in tuples:
        values = _split_top_level_values(tup)
        if not columns:
            # No explicit column list -> guess positional layout.
            columns = _guess_columns(len(values))
        if len(values) < len(columns):
            continue

        product = {}
        raw_ok = False
        for (canonical, _raw), value in zip(columns, values):
            if canonical is None:
                continue
            value = _clean_value(value)
            if canonical == "price":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    raw_ok = True
            elif canonical == "stock":
                try:
                    value = int(float(value))
                except (TypeError, ValueError):
                    value = None
            product[canonical] = value

        product = {k: v for k, v in product.items() if v is not None}
        if product.get("name"):
            product.setdefault("description", "")
            product.setdefault("price", None)
            product.setdefault("stock", None)
            product.setdefault("sku", product.get("id"))
            product.setdefault("id", product.get("sku"))
            product.setdefault("category", "Uncategorized")
            raw_ok = True

        if raw_ok and product.get("name"):
            products.append(product)

    return products


def _split_top_level_tuples(blob: str) -> list:
    """Split a blob of '(...),(...),...' rows at top-level commas."""
    tuples, depth = [], 0
    current = []
    i, n = 0, len(blob)
    while i < n:
        ch = blob[i]
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tuples.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        tuples.append("".join(current))
    return [t.strip().strip("()") for t in tuples if t.strip().strip("()")]


def _split_top_level_values(tup: str) -> list:
    """Split a single row's values honoring quoted strings."""
    values, current, in_str, esc = [], [], False, False
    quote = None
    for ch in tup:
        if esc:
            current.append(ch)
            esc = False
            continue
        if ch == "\\":
            esc = True
            current.append(ch)
            continue
        if in_str:
            current.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            current.append(ch)
        elif ch == ",":
            values.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        values.append("".join(current))
    return values


def _clean_value(value: str):
    """Strip quotes/NULL handling from a raw SQL value string."""
    value = value.strip()
    if value.upper() in ("NULL", "NONE", "DEFAULT"):
        return None
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        value = value[1:-1]
        value = value.replace("''", "'").replace('\\"', '"')
    return value


def _guess_columns(count: int) -> list:
    """Best-effort positional column mapping when no column list is present."""
    order = ["id", "name", "sku", "description", "price", "category", "stock"]
    out = []
    for i in range(count):
        out.append((order[i] if i < len(order) else None, f"col{i + 1}"))
    return out


def load_products(sql_path: str) -> list:
    """Load product records from a SQL dump file.

    Args:
        sql_path: Absolute/relative path to the ``.sql`` file.

    Returns:
        List of product dicts with canonical keys
        (id, sku, name, description, price, category, stock).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file exists but contains no usable product rows.
    """
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    with open(sql_path, "r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read()

    # Try real SQL parsing first; fall back to regex if sqlparse chokes.
    statements = []
    try:
        statements = sqlparse.parse(raw)
    except Exception:  # pragma: no cover - defensive
        statements = _regex_split_statements(raw)

    if _is_eav_dump(statements):
        logger.info("Detected Magento 2 EAV dump; using EAV loader.")
        return _load_eav(statements)

    products = []
    for stmt in statements:
        text = str(stmt).strip()
        if not text.lower().startswith("insert"):
            continue
        products.extend(_parse_insert_statement(text))

    if not products:
        raise ValueError(
            f"SQL file '{sql_path}' contained no parseable product rows "
            "(flat schema or supported EAV schema)."
        )

    logger.info("Loaded %d products from %s", len(products), sql_path)
    return products


def _is_eav_dump(statements) -> bool:
    """True if the dump contains Magento EAV value tables."""
    haystack = " ".join(str(s) for s in statements).lower()
    return ("catalog_product_entity_varchar" in haystack
            or "catalog_product_entity_decimal" in haystack)


# ---------------------------------------------------------------------------
# Magento 2 EAV loader
# ---------------------------------------------------------------------------

def _load_eav(statements) -> list:
    """Reconstruct flat product records from a Magento EAV dump.

    Strategy:
        * ``catalog_product_entity`` provides entity_id + sku + timestamps.
        * EAV value tables populate name/description/price/etc. per entity_id
          filtered to the relevant attribute ids and the default store (0).
        * Keeps the first value per (entity_id, attribute) so per-store
          duplicates don't explode the dataset.

    Returns:
        List of flat product dicts.
    """
    entity_skus = {}
    values = {}  # (entity_id, table, attribute_id) -> value

    # Map from table name -> category of columns we care about.
    eav_extract = {
        "catalog_product_entity": _extract_entity_rows,
        "catalog_product_entity_varchar": _extract_eav_rows,
        "catalog_product_entity_text": _extract_eav_rows,
        "catalog_product_entity_decimal": _extract_eav_rows,
        "catalog_product_entity_int": _extract_eav_rows,
        "catalog_product_entity_datetime": _extract_eav_rows,
    }

    for stmt in statements:
        text = str(stmt).strip()
        m = re.match(r"(?i)^INSERT INTO `([a-z0-9_]+)`\s+VALUES\s+(.*)$", text,
                     re.S)
        if not m:
            continue
        table = m.group(1).lower()
        body = m.group(2).rstrip(";").strip()
        if table not in eav_extract:
            continue
        eav_extract[table](body, entity_skus, values)

    products = _assemble_eav_products(entity_skus, values)
    logger.info("Loaded %d products (EAV rebuild) from dump.",
                len(products))
    return products


def _extract_entity_rows(body, entity_skus, values):
    """Parse catalog_product_entity rows for entity_id -> sku and timestamps."""
    # Row shape: (entity_id, attribute_set_id, type_id, approval, sku,
    #             has_options, required_options, created_at, updated_at)
    for mm in re.finditer(
            r"\((\d+),\s*\d+,\s*'([^']*)',\s*\d+,\s*'([^']*)'", body):
        eid = int(mm.group(1))
        entity_skus[eid] = {"sku": mm.group(3), "type": mm.group(2)}


def _extract_eav_rows(body, entity_skus, values):
    """Parse an EAV value table into (entity_id, attr_id) -> value dict."""
    # Row shape: (value_id, attribute_id, store_id, entity_id, value)
    for mm in re.finditer(
            r"\((?:\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(?:'(.*?)'|([0-9.\-]+))\)",
            body, re.S):
        attr_id = int(mm.group(1))
        store_id = int(mm.group(2))
        entity_id = int(mm.group(3))
        value = mm.group(4) if mm.group(4) is not None else mm.group(5)
        if store_id != 0:
            continue  # use only default-store values
        key = (entity_id, attr_id)
        if key not in values and value is not None and str(value).strip():
            values[key] = value


def _assemble_eav_products(entity_skus, values) -> list:
    """Join entity + EAV values into flat product records."""
    products = []
    for eid, ent in entity_skus.items():
        sku = ent.get("sku", "")
        price = _to_float(values.get((eid, _EAV_PRICE)))
        special = _to_float(values.get((eid, _EAV_SPECIAL_PRICE)))
        # Effective price: special price wins when present and lower.
        if special is not None:
            price = special if price is None else min(price, special)

        name = _clean_text(values.get((eid, _EAV_NAME), ""))
        if not name:
            name = _clean_text(values.get((eid, 84), ""))
        description = _strip_html(
            values.get((eid, _EAV_DESCRIPTION), "")
            or values.get((eid, _EAV_SHORT_DESCRIPTION), "")
        )

        products.append({
            "id": eid,
            "sku": sku,
            "name": name,
            "description": description.strip(),
            "price": price,
            "category": "Uncategorized",  # no category tables in this dump
            "stock": None,                # no inventory rows in this dump
            "upc": str(values.get((eid, _EAV_UPC), "") or ""),
            "url_key": str(values.get((eid, _EAV_URL_KEY), "") or ""),
            "vendor": str(values.get((eid, _EAV_VENDOR), "") or ""),
        })
    # Drop records with no usable identity.
    return [p for p in products if p["sku"] or p["name"]]


def _to_float(value):
    """Best-effort conversion of a value to float (or None)."""
    try:
        return float(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _strip_html(text: str) -> str:
    """Remove HTML tags/entities and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html_unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text: str) -> str:
    """Clean escaped SQL double-quotes and collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\\\"", '"').replace("\\\\r\\\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _regex_split_statements(raw: str) -> list:
    """Fallback: naive split of a dump on full INSERT inserts."""
    parts = re.split(r"(?i)(?=INSERT INTO)", raw)
    return [p for p in parts if p.strip()]
