"""
hotel_core - framework-free domain logic for the hotel agent.

No agent-framework, cloud or LLM imports: this module can be reused unchanged behind
LangGraph, Microsoft Agent Framework, OpenAI Agents SDK, a REST API or a test suite.

Contents
  - dates and record building (price per night, % positive reviews)
  - field catalog: discovers every field path in the live data
  - generic query engine: {field, op, value} conditions, AND/OR, scoped, multi-key sort
  - SerpApi adapters: Google Hotels (paged) and Google Hotels Reviews
  - rendering: console table and CSV
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_MAX_PAGES = 5           # ~20 hotels per page; each page costs 1 SerpApi credit
HARD_MAX_PAGES = 25
MAX_REVIEW_HOTELS = 5           # each hotel's reviews cost 1 SerpApi credit
REVIEWS_PER_HOTEL = 15
DROP_KEYS = {"images", "thumbnail", "serpapi_property_details_link",
             "serpapi_google_hotels_reviews_link", "category_token", "serpapi_link", "source_icon", "logo"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def validate_dates(check_in: str, check_out: str, today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    try:
        ci, co = date.fromisoformat(check_in), date.fromisoformat(check_out)
    except ValueError as exc:
        raise ValueError("Dates must be in YYYY-MM-DD format.") from exc
    if ci < today:
        raise ValueError(f"Check-in {ci} is in the past (today is {today}).")
    if co <= ci:
        raise ValueError("Check-out must be after check-in.")
    return ci, co


def strip_noise(obj: Any) -> Any:
    """Drop images, tokens and SerpApi links; keep every data field."""
    if isinstance(obj, dict):
        return {k: strip_noise(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip_noise(v) for v in obj]
    return obj


def positive_review_pct(prop: dict) -> Optional[float]:
    counts = {r.get("stars"): r.get("count", 0) or 0 for r in prop.get("ratings") or []}
    total = sum(counts.values())
    return None if total == 0 else round(100 * (counts.get(5, 0) + counts.get(4, 0)) / total, 1)


def build_record(prop: dict, nights: int, currency: str) -> Optional[dict]:
    """Keep the full SerpApi record and add a few derived, comparable fields."""
    total = (prop.get("total_rate") or {}).get("extracted_lowest")
    per_night = (prop.get("rate_per_night") or {}).get("extracted_lowest")
    if total is None and per_night is None:
        return None  # no price for these dates
    per_night = per_night if per_night is not None else total / nights
    total = total if total is not None else per_night * nights
    rec = strip_noise(prop)
    rec.update({
        "_token": prop.get("property_token"),
        "price_per_night": round(float(per_night), 2),
        "total_price": round(float(total), 2),
        "currency": currency,
        "positive_review_pct": positive_review_pct(prop),
    })
    rec.pop("property_token", None)
    return rec


def resolve(record: Any, path: str) -> list[Any]:
    """All leaf values at a dot path, flattening lists: 'nearby_places.transportations.type'."""
    vals = [record]
    for part in path.split("."):
        nxt = []
        for v in vals:
            if isinstance(v, dict) and part in v:
                x = v[part]
                nxt.extend(x if isinstance(x, list) else [x])
        vals = nxt
    flat = []
    for v in vals:
        flat.extend(v if isinstance(v, list) else [v])
    return [v for v in flat if v is not None and not isinstance(v, dict)]


def field_catalog(hotels: list[dict], max_examples: int = 8) -> dict[str, dict]:
    """Discover every field path in the data with its type, coverage and example values."""
    present: Counter = Counter()
    values: dict[str, list] = defaultdict(list)

    def walk(obj: Any, path: str, seen: set) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not k.startswith("_"):
                    walk(v, f"{path}.{k}" if path else k, seen)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, path, seen)
        elif obj is not None and path:
            if path not in seen:
                seen.add(path)
                present[path] += 1
            values[path].append(obj)

    for h in hotels:
        walk(h, "", set())

    catalog = {}
    for path, vals in values.items():
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        entry: dict[str, Any] = {"in_hotels": present[path]}
        if nums and len(nums) == len(vals):
            entry.update(type="number", min=min(nums), max=max(nums))
        elif all(isinstance(v, bool) for v in vals):
            entry.update(type="bool")
        else:
            common = Counter(str(v) for v in vals).most_common()
            entry["type"] = "text"
            if path.endswith("amenities"):
                n = 60                                   # the vocabulary the LLM must match
            elif len(common) < len(vals) * 0.7:
                n = max_examples                         # categorical: type, hotel_class, ...
            else:
                n = 2                                    # free text: name, description, ...
            entry["examples"] = [c for c, _ in common[:n]]
        catalog[path] = entry
    return dict(sorted(catalog.items()))


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


class Condition(BaseModel):
    scope: Optional[str] = Field(description=(
        "Null normally. For list-of-object fields, the list path (e.g. 'nearby_places') when several "
        "conditions must hold for the SAME item; then 'field' is relative to it (e.g. 'name')."))
    field: str = Field(description="Field path from the catalog, e.g. 'amenities', 'extracted_hotel_class', 'nearby_places.name'.")
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "not_contains", "exists", "not_exists"]
    value: Optional[str] = Field(description="Value to compare (numbers as text, e.g. '4'). Null for exists/not_exists/in.")
    values: Optional[list[str]] = Field(description="Values for op 'in', e.g. ['4','5']. Null otherwise.")


class SortKey(BaseModel):
    field: str
    descending: bool


def check(record: dict, c: Condition) -> bool:
    vals = resolve(record, c.field)
    if c.op == "exists":
        return bool(vals)
    if c.op == "not_exists":
        return not vals
    target = (c.value or "").strip().lower()
    if c.op == "contains":
        return any(target in str(v).lower() for v in vals)
    if c.op == "not_contains":
        return not any(target in str(v).lower() for v in vals)
    if c.op in ("eq", "ne", "in"):
        wanted = [w.strip().lower() for w in (c.values or [])] if c.op == "in" else [target]
        def same(v: Any, w: str) -> bool:
            nv, nw = _num(v), _num(w)
            return nv == nw if nv is not None and nw is not None else str(v).lower() == w
        hit = any(same(v, w) for v in vals for w in wanted)
        return not hit if c.op == "ne" else hit
    x = _num(c.value)
    nums = [n for v in vals if (n := _num(v)) is not None]
    if x is None or not nums:
        return False
    return {"gt": any(n > x for n in nums), "gte": any(n >= x for n in nums),
            "lt": any(n < x for n in nums), "lte": any(n <= x for n in nums)}[c.op]


def full_path(c: Condition) -> str:
    return f"{c.scope}.{c.field}" if c.scope else c.field


def objects_at(record: dict, path: str) -> list[dict]:
    """The dict items at a list path, e.g. each entry of 'nearby_places'."""
    items: list[Any] = [record]
    for part in path.split("."):
        nxt = []
        for v in items:
            if isinstance(v, dict) and part in v:
                x = v[part]
                nxt.extend(x if isinstance(x, list) else [x])
        items = nxt
    return [i for i in items if isinstance(i, dict)]


def check_all(record: dict, conds: list[Condition]) -> bool:
    """AND of conditions; scoped ones must all hold on one and the same list item."""
    scoped: dict[str, list[Condition]] = defaultdict(list)
    for c in conds:
        if c.scope:
            scoped[c.scope].append(c)
        elif not check(record, c):
            return False
    return all(any(all(check(obj, c) for c in group) for obj in objects_at(record, scope))
               for scope, group in scoped.items())


def run_query(hotels: list[dict], all_of: list[Condition], any_of: list[Condition], sort: list[SortKey]) -> list[dict]:
    out = [h for h in hotels
           if check_all(h, all_of) and (not any_of or any(check_all(h, [c]) for c in any_of))]
    for key in reversed(sort or [SortKey(field="overall_rating", descending=True)]):
        def k(h: dict, key=key):
            nums = [n for v in resolve(h, key.field) if (n := _num(v)) is not None]
            if not nums:
                return (1, 0.0)                       # missing values always last
            n = max(nums) if key.descending else min(nums)
            return (0, -n if key.descending else n)
        out.sort(key=k)                                # stable: earlier keys win
    return out


def find_hotels(hotels: list[dict], names: list[str]) -> list[dict]:
    found = []
    for n in names:
        n = n.lower().strip()
        match = next((h for h in hotels if h.get("name", "").lower() == n), None) or \
                next((h for h in hotels if n in h.get("name", "").lower()), None)
        if match and match not in found:
            found.append(match)
    return found


# ---------------------------------------------------------------------------
# SerpApi I/O
# ---------------------------------------------------------------------------
def _api_key() -> str:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("SERPAPI_API_KEY is not set.")
    return key


async def fetch_all_hotels(location: str, check_in: str, check_out: str, adults: int,
                           currency: str, max_pages: int) -> tuple[list[dict], int, bool]:
    base = {"engine": "google_hotels", "q": location, "check_in_date": check_in,
            "check_out_date": check_out, "adults": adults, "currency": currency, "hl": "en",
            "gl": os.environ.get("SERPAPI_GL", "in"), "api_key": _api_key()}
    seen: set[str] = set()
    props: list[dict] = []
    token: Optional[str] = None
    pages = 0
    async with httpx.AsyncClient(timeout=40) as client:
        while pages < max_pages:
            resp = await client.get(SERPAPI_URL, params=dict(base, **({"next_page_token": token} if token else {})))
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                if pages == 0:
                    raise RuntimeError(data["error"])
                break
            pages += 1
            for p in data.get("properties", []):
                key = p.get("property_token") or p.get("name")
                if key and key not in seen:
                    seen.add(key)
                    props.append(p)
            token = (data.get("serpapi_pagination") or {}).get("next_page_token")
            if not token:
                return props, pages, False
    return props, pages, bool(token)


async def fetch_reviews(token: str) -> list[dict]:
    params = {"engine": "google_hotels_reviews", "property_token": token, "hl": "en", "api_key": _api_key()}
    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.get(SERPAPI_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    out = []
    for r in data.get("reviews", [])[:REVIEWS_PER_HOTEL]:
        item = {"rating": r.get("rating"), "date": r.get("date"),
                "text": (r.get("snippet") or "")[:500],
                "highlights": r.get("hotel_highlights") or [],
                "about": [a.get("name") for a in r.get("attributes") or [] if a.get("name")]}
        if item["text"] or item["highlights"]:
            out.append(item)
    return out




def known_fields(hotels: list[dict]) -> set[str]:
    return set(field_catalog(hotels)) | {"price_per_night", "total_price", "positive_review_pct"}


def preview(rows: list[dict], fields: list[str], n: int = 5) -> list[dict]:
    """Compact view of the top rows for the LLM: core fields plus the fields being used."""
    cols = list(dict.fromkeys(["name", "price_per_night", "overall_rating", "extracted_hotel_class", *fields]))
    out = []
    for h in rows[:n]:
        row = {}
        for c in cols:
            v = resolve(h, c)
            row[c] = v[:3] if len(v) > 1 else (v[0] if v else None)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Rendering (console table + CSV)
# ---------------------------------------------------------------------------
def _qpath(c: dict) -> str:
    return f"{c['scope']}.{c['field']}" if c.get("scope") else c["field"]


def _cell(h: dict, path: str, query: dict) -> str:
    vals = resolve(h, path)
    terms = [c["value"].lower() for c in query.get("all_of", []) + query.get("any_of", [])
             if _qpath(c) == path and c["op"] == "contains" and c.get("value")]
    if terms:  # show only the values that matched, e.g. 'Free parking'
        vals = [v for v in vals if any(t in str(v).lower() for t in terms)] or vals
    return ", ".join(dict.fromkeys(str(v) for v in vals))


def format_table(search: dict, rows: list[dict], query: dict, total: int, limit: int = 25) -> str:
    lines = [f"\n{search['location']} | {search['check_in']} -> {search['check_out']} "
             f"({search['nights']} nights) | {len(rows)} of {total} hotels"]
    if not rows:
        return "\n".join(lines)
    core = {"name", "price_per_night", "overall_rating", "extracted_hotel_class", "positive_review_pct"}
    extra = list(dict.fromkeys([_qpath(c) for c in query.get("all_of", []) + query.get("any_of", [])]
                               + [k["field"] for k in query.get("sort", [])]))
    extra = [e for e in extra if e not in core]
    head = lambda e: ".".join(e.split(".")[-2:])[:24]
    lines.append(f"{'#':>3}  {'Hotel':<36} {'Class':>5} {'Rating':>6} {'Pos%':>6} {'Per night':>13}  "
                 + "  ".join(f"{head(e):<24}" for e in extra))
    lines.append("-" * (75 + 26 * len(extra)))
    cur = search["currency"]
    for i, h in enumerate(rows[:limit], 1):
        pos = h.get("positive_review_pct")
        lines.append(f"{i:>3}  {h.get('name', '?')[:36]:<36} {h.get('extracted_hotel_class') or '-':>5} "
                     f"{h.get('overall_rating') or '-':>6} {(str(pos) + '%') if pos is not None else '-':>6} "
                     f"{cur} {h['price_per_night']:>9,.0f}  "
                     + "  ".join(f"{_cell(h, e, query)[:24]:<24}" for e in extra))
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more (full list in the CSV).")
    return "\n".join(lines)


def save_csv(search: dict, rows: list[dict], out_dir: str = ".") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", search["location"].lower()).strip("-")
    path = os.path.join(out_dir, f"hotels_{slug}_{search['check_in']}_{search['check_out']}.csv")
    cols = ["name", "type", "extracted_hotel_class", "overall_rating", "positive_review_pct", "reviews",
            "location_rating", "price_per_night", "total_price", "currency", "amenities", "link"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for h in rows:
            w.writerow(["; ".join(map(str, resolve(h, c))) for c in cols])
    return path