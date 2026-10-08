"""
Ranking, budget filtering, sorting and "top picks". Pure code, no LLM,
no SerpApi calls.

Because this is separate from the agent, the UI can change the sort
order or budget instantly WITHOUT re-running the agent (zero credits,
zero LLM calls):

    raw = run_agent("mug")                         # expensive, run once
    view = finalize(raw, max_price=500, sort_by="price_low_high")   # free

There is deliberately NO numeric "eco score". We can't measure that
honestly. Products are ordered by: the agent's relevance order, price
(weighted heavily if the user picked low_cost), rating as a small
tiebreaker (often missing), and, once the evidence step exists, the
number of claims backed by a supporting source. Each product carries a
plain-language "why_ranked" line instead of a mystery number.
"""

import re
import copy


def extract_price(value) -> float | None:
    """
    The one price parser used across the backend.

    599 -> 599.0, '₹599' -> 599.0, '₹1,299.50' -> 1299.5,
    '₹1,29,999' (Indian grouping) -> 129999.0, None/unparseable -> None.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def parse_number(value) -> float | None:
    """Rating/review counts: 4.5, '4.5', '1,234', '1.2K' (K/M suffix)."""
    if value in (None, "", "N/A") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?", str(value))
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    return number * (1_000 if suffix == "k" else 1_000_000 if suffix == "m" else 1)


def _score(item: dict, prefs: set, lo: float, hi: float, n: int) -> tuple[float, str]:
    parts = []

    # 1) The agent's own relevance order (earlier = better match)
    relevance = 1 - item["_idx"] / max(n, 1)
    score = 0.5 * relevance

    # 2) Price, normalised within this group
    price = item["price_value"]
    if price is not None:
        price_score = 1.0 if hi == lo else 1 - (price - lo) / (hi - lo)
        score += (0.7 if "low_cost" in prefs else 0.25) * price_score
        if hi != lo:
            if price_score >= 0.67:
                parts.append("among the cheaper options in its group")
            elif price_score <= 0.33:
                parts.append("on the pricier side for its group")

    # 3) Rating: small tiebreaker, often missing
    rating = parse_number(item.get("rating"))
    if rating:
        score += 0.15 * (min(rating, 5.0) / 5)
        reviews = item.get("reviews")
        parts.append(f"rated {rating}" + (f" ({reviews} reviews)" if reviews else ""))

    # 4) The user's explicit requirements come before environmental signals:
    #    a product whose stated requirements are all backed by retrieved text
    #    ranks above one where they are unverified.
    user_checks = [c for c in item.get("requirement_checks", []) if c.get("kind") == "user"]
    if user_checks:
        unverified = [c["requirement"] for c in user_checks if c.get("status") == "unverified"]
        if not unverified:
            score += 0.5
            parts.append("your stated requirements are backed by the listing or a source")
        else:
            parts.append(f"not verified: {', '.join(unverified)}")

    # 5) Evidence: only product-specific evidence found by research counts.
    #    Seller-stated words (eco, organic, recycled, natural...) are claims
    #    to investigate, never a ranking bonus on their own.
    supported = [c for c in item.get("claims", []) if c.get("status") == "supported_by_search"]
    if supported:
        score += 0.4 * min(len(supported), 3) / 3
        parts.append(f"{len(supported)} claim(s) with evidence found")

    if not parts:
        parts.append("agent's relevance order")

    text = "; ".join(parts)
    return round(score, 3), text[0].upper() + text[1:]


SORT_OPTIONS = ["recommended", "price_low_high", "price_high_low", "rating", "reviews"]


def _apply_sort(items: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "price_low_high":
        items.sort(key=lambda i: (i["price_value"] is None, i["price_value"] or 0))
    elif sort_by == "price_high_low":
        items.sort(key=lambda i: (i["price_value"] is None, -(i["price_value"] or 0)))
    elif sort_by == "rating":
        items.sort(key=lambda i: (parse_number(i.get("rating")) is None, -(parse_number(i.get("rating")) or 0)))
    elif sort_by == "reviews":
        items.sort(key=lambda i: (parse_number(i.get("reviews")) is None, -(parse_number(i.get("reviews")) or 0)))
    else:  # "recommended"
        items.sort(key=lambda i: i["score"], reverse=True)
    return items


def rank_products(
    products: list[dict],
    preferences: list[str] | None = None,
    max_price: float | None = None,
    sort_by: str = "recommended",
) -> list[dict]:
    prefs = set(preferences or [])
    n = len(products)

    items = []
    for idx, p in enumerate(products):
        price = extract_price(p.get("extracted_price"))
        if price is None:
            price = extract_price(p.get("price"))
        # Products with an unknown price are kept: we can't prove they're over budget.
        if max_price is not None and price is not None and price > max_price:
            continue
        items.append({**p, "price_value": price, "_idx": idx})

    prices = [i["price_value"] for i in items if i["price_value"] is not None]
    lo, hi = (min(prices), max(prices)) if prices else (0.0, 0.0)

    for i in items:
        i["score"], i["why_ranked"] = _score(i, prefs, lo, hi, n)
        # Lowest price AMONG THESE LISTINGS — not a same-item comparison
        # across merchants (each listing is a different seller/product;
        # we never claim two listings are the same item).
        i["is_lowest_price"] = bool(prices) and len(items) > 1 and i["price_value"] == lo

    _apply_sort(items, sort_by)

    for i in items:
        i.pop("_idx", None)
    return items


def build_top_picks(
    materials: list[dict], n: int = 4, sort_by: str = "recommended",
    request_sustainability: str = "none",
) -> list[dict]:
    """
    Best product from each material first (for variety), then fill by
    score. Priority order depends on what the user asked for:
      - eco_leaning: the matching material leads (they asked for
        something already better; give it to them first).
      - high_impact: the matching material is deprioritized — the
        OTHER (better) materials lead instead, since that's what the
        user should actually see first.
      - none: no material is prioritized over another.
    """
    matching = [m for m in materials if m.get("matches_request")]
    other = [m for m in materials if not m.get("matches_request")]

    if request_sustainability == "high_impact":
        lead, trail = other, matching
    else:
        lead, trail = matching, other

    def best_first(mats):
        best = [m["products"][0] for m in mats if m["products"]]
        best.sort(key=lambda p: p["score"], reverse=True)
        return best

    picks = best_first(lead) + best_first(trail)
    picks = picks[:n]

    if len(picks) < n:
        rest = [p for m in (lead + trail) for p in m["products"][1:]]
        rest.sort(key=lambda p: p["score"], reverse=True)
        picks += rest[: n - len(picks)]

    if sort_by == "recommended":
        lead_types = {m["type"] for m in lead}
        picks.sort(key=lambda p: 0 if p.get("material_type") in lead_types else 1)
        return picks

    return _apply_sort(picks, sort_by)


def finalize(
    result: dict,
    preferences: list[str] | None = None,
    max_price: float | None = None,
    sort_by: str = "recommended",
) -> dict:
    """Apply ranking, budget and sorting to a raw agent result. Free to call repeatedly."""
    out = copy.deepcopy(result)

    for m in out["materials"]:
        m["products"] = rank_products(m["products"], preferences, max_price, sort_by)
    out["materials"] = [m for m in out["materials"] if m["products"]]

    req = out.get("request_sustainability", "none")
    if req == "high_impact":
        out["materials"].sort(key=lambda m: 1 if m.get("matches_request") else 0)
    else:
        out["materials"].sort(key=lambda m: 0 if m.get("matches_request") else 1)

    out["top_picks"] = build_top_picks(out["materials"], sort_by=sort_by, request_sustainability=req)
    out["no_suitable_alternative"] = len(out["materials"]) == 0

    all_priced = [p for m in out["materials"] for p in m["products"] if p["price_value"] is not None]
    out["cheapest_found"] = min(all_priced, key=lambda p: p["price_value"]) if all_priced else None

    return out