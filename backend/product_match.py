"""
Pure-code product-type enforcement. No LLM call, no SerpApi call.

The model states a "product_type" (e.g. "chair") in its JSON output.
This module is the ENFORCEMENT layer: it checks that word actually
appears in each product's own title, and drops anything that doesn't
— regardless of what material/why_suggested text the model wrote.

This exists because a prompt instruction alone ("don't show a bench
when a chair was asked for") is a request, not a guarantee — the model
can and did ignore it once already. Same pattern as evidence.py and
the high-impact safety net in agent.py: never trust the model alone,
verify in code.
"""


def _normalize(text: str | None) -> str:
    return (text or "").lower()


def filter_by_product_type(materials: list[dict], product_type: str | None) -> tuple[list[dict], int]:
    """
    Drops a product whose title doesn't contain `product_type` as a
    substring (case-insensitive — this also forgives plurals, e.g.
    "chair" matches "Chairs") — but ONLY within a material group where
    at least one product DOES match.

    Why the "at least one match" gate matters: for physical-form
    categories (chair, mug) a mismatch is a real error — a bench
    turning up among chairs. But for reformulation categories (toilet
    cleaner, shampoo) a genuinely good alternative format often won't
    contain the category noun at all — "Citric Acid Cleaning Powder"
    is a legitimate toilet-cleaner alternative that will never say
    "toilet cleaner" in its title. If NOTHING in a group matches, that
    means the category noun just doesn't apply to how this material's
    products are naturally titled — nuking the whole group in that
    case would silently kill legitimate alternatives (this happened:
    citric acid and baking soda groups were wrongly dropped for a
    "toilet cleaner" search). So an all-zero group is left untouched
    and trusted to the model/evidence/critique layers instead.

    Returns (filtered_materials, dropped_count).
    """
    needle = _normalize(product_type).strip()
    if not needle:
        return materials, 0

    dropped = 0
    kept_materials = []

    for m in materials:
        products = m["products"]
        matches = [p for p in products if needle in _normalize(p.get("name"))]

        if matches:
            dropped += len(products) - len(matches)
            m["products"] = matches
        # else: zero matches in this group — leave products as-is, untouched.

        kept_materials.append(m)

    return kept_materials, dropped