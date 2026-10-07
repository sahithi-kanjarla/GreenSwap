"""
Pure-code product-type enforcement. No LLM call, no SerpApi call.

The model states a "product_type" (e.g. "chair") in its JSON output.
This module is the ENFORCEMENT layer: it checks that the type actually
appears in each product's own title, and drops anything that doesn't
— regardless of what material/why_suggested text the model wrote.

This exists because a prompt instruction alone ("don't show a bench
when a chair was asked for") is a request, not a guarantee — the model
can and did ignore it once already. Same pattern as evidence.py and
the high-impact safety net in agent.py: never trust the model alone,
verify in code.

Matching is word-based, not raw substring:
  - "cup" does NOT match "Cupboard" (whole words only);
  - plurals are forgiven ("knife" matches "Knives", "box" matches "Boxes");
  - multi-word types match when every word is present in any order
    ("food container" matches "Food Storage Container");
  - multi-word types also match their compound spelling
    ("lunch box" matches "Lunchbox").
"""

import re


def _stem(word: str) -> str:
    """Very small English plural stemmer — enough for product nouns."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves"):
        return word[:-3] + "f"
    if word.endswith("fe"):
        return word[:-2] + "f"
    if word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _words(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def title_matches(product_type: str | None, title: str | None) -> bool:
    """True when `title` names the same kind of product as `product_type`."""
    needle = _words(product_type)
    if not needle:
        return True

    title_words = _words(title)
    title_stems = {_stem(w) for w in title_words}

    if all(_stem(w) in title_stems for w in needle):
        return True

    # Compound spelling: "lunch box" -> "lunchbox", "water bottle" -> "waterbottle".
    if len(needle) > 1:
        compound = _stem("".join(needle))
        return any(_stem(w) == compound for w in title_words)

    return False


def filter_by_product_type(materials: list[dict], product_type: str | None) -> tuple[list[dict], int]:
    """
    Drops a product whose title doesn't name `product_type` — but ONLY
    within a material group where at least one product DOES match.

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
    if not _words(product_type):
        return materials, 0

    dropped = 0
    kept_materials = []

    for m in materials:
        products = m["products"]
        matches = [p for p in products if title_matches(product_type, p.get("name"))]

        if matches:
            dropped += len(products) - len(matches)
            m["products"] = matches
        # else: zero matches in this group — leave products as-is, untouched.

        kept_materials.append(m)

    return kept_materials, dropped
