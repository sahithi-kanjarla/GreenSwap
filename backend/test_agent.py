"""
Run the GreenSwap agent end to end from the terminal.

Examples:
    python test_agent.py "mug"
    python test_agent.py "microwave safe mug"
    python test_agent.py "kitchen scrubber"
    python test_agent.py "virgin plastic chair"

The first run of a new query uses up to 5 SerpApi credits.
Repeating the same query should hit the disk cache.

The raw result is saved to:
    demo_runs/<query>.json
"""

import re
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing serp_tool.
load_dotenv()

from agent import run_agent
from ranking import finalize


# ============================================================
# TEST SETTINGS
# ============================================================

# Optional ranking preferences.
# Try:
#   []
#   ["low_cost"]
#   ["reusable"]
PREFERENCES = ["low_cost"]

# Optional budget.
# None = no budget filter
# Example:
# MAX_PRICE = 500
MAX_PRICE = None

# Sorting:
#   "recommended"
#   "price_low_high"
#   "price_high_low"
SORT_BY = "recommended"


# ============================================================
# HELPERS
# ============================================================

def safe_claim_text(claim: dict) -> str:
    """
    Read evidence fields safely.

    Different versions of the backend may return slightly
    different evidence schemas, so don't assume every key exists.
    """

    mark = claim.get("mark")

    if not mark:
        if claim.get("validated"):
            mark = "✓"
        else:
            mark = "⚠"

    label = claim.get("label", "Evidence")
    claim_text = claim.get("claim", "")
    snippet = claim.get("evidence_snippet", "")

    if snippet:
        return (
            f'       {mark} {label}: {claim_text} '
            f'<- "{snippet}"'
        )

    return f"       {mark} {label}: {claim_text}"


def print_material(material: dict) -> None:
    """
    Print one material/alternative group safely.
    """

    material_type = material.get("type", "Unknown")
    about = material.get("about", "")

    print(f"\n[{material_type}] {about}")

    # --------------------------------------------------------
    # IMPACT
    # --------------------------------------------------------

    impact = material.get("impact", {})

    print(
        "   Impact — Material: "
        f"{impact.get('material_note', 'n/a')}"
    )

    print(
        "            Reusability: "
        f"{impact.get('reusability', 'n/a')}"
    )

    print(
        "            End of life: "
        f"{impact.get('end_of_life', 'n/a')}"
    )

    print(
        "            Common trade-off: "
        f"{impact.get('common_trade_off', 'n/a')}"
    )

    # --------------------------------------------------------
    # PRODUCTS
    # --------------------------------------------------------

    products = material.get("products", [])

    if not products:
        print("   No products found.")
        return

    for product in products:

        name = product.get("name", "Unnamed product")
        price = product.get("price", "Price unavailable")
        source = product.get("source", "Unknown seller")

        badge = ""

        if product.get("is_lowest_price"):
            badge = "  💰 lowest in group"

        print(
            f"   - {name}{badge} | "
            f"{price} | {source}"
        )

        # ----------------------------------------------------
        # PRODUCT EXPLANATION
        # ----------------------------------------------------

        why = product.get("why_suggested")

        if why:
            print(f"       Why: {why}")

        trade_off = product.get("trade_off")

        if trade_off:
            print(
                f"       Trade-off: {trade_off}"
            )

        # ----------------------------------------------------
        # RATING / REVIEWS
        # ----------------------------------------------------

        rating = product.get("rating")
        reviews = product.get("reviews")

        if rating is not None or reviews is not None:

            rating_text = (
                str(rating)
                if rating is not None
                else "n/a"
            )

            reviews_text = (
                str(reviews)
                if reviews is not None
                else "n/a"
            )

            print(
                f"       Rating: {rating_text} | "
                f"Reviews: {reviews_text}"
            )

        # ----------------------------------------------------
        # EVIDENCE / CLAIMS
        # ----------------------------------------------------

        claims = product.get("claims", [])

        if claims:

            print("       Evidence:")

            for claim in claims:
                print(safe_claim_text(claim))

        # ----------------------------------------------------
        # LINK
        # ----------------------------------------------------

        link = product.get("link")

        if link:
            print(
                f"       🔗 Compare prices / Buy: {link}"
            )


# ============================================================
# MAIN DISPLAY
# ============================================================

def show(result: dict) -> None:

    # ========================================================
    # TRACE
    # ========================================================

    print("\n" + "=" * 70)
    print("--- TRACE ---")
    print("=" * 70)

    for step in result.get("trace", []):

        cached = step.get("cached", False)

        tag = "cache" if cached else "LIVE "

        error = step.get("error")

        error_text = (
            f"  ERROR: {error}"
            if error
            else ""
        )

        print(
            f"  {step.get('step', '?')}. "
            f"[{step.get('engine', '?')}/{tag}] "
            f"\"{step.get('query', '')}\" "
            f"-> {step.get('results_found', 0)} results"
            f"{error_text}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("--- SUMMARY ---")
    print("=" * 70)

    print(
        result.get(
            "summary",
            "No summary available."
        )
    )

    request_sustainability = result.get(
        "request_sustainability",
        "none"
    )

    user_material = result.get(
        "user_specified_material"
    )

    if user_material:

        print(
            f"\n(You asked for: {user_material} "
            f"— classified as {request_sustainability})"
        )

    caution = result.get("caution_note")

    if caution:

        print(
            f"\n⚠ {caution}"
        )

    # ========================================================
    # TOP PICKS
    # ========================================================

    print("\n" + "=" * 70)
    print("--- TOP PICKS ---")
    print("=" * 70)

    top_picks = result.get("top_picks", [])

    if not top_picks:

        print("No top picks returned.")

    else:

        for product in top_picks:

            name = product.get(
                "name",
                "Unnamed product"
            )

            badge = ""

            if product.get("is_lowest_price"):
                badge = "  💰 lowest in group"

            price = product.get(
                "price",
                "Price unavailable"
            )

            source = product.get(
                "source",
                "Unknown seller"
            )

            material = product.get(
                "material_type",
                "Unknown material"
            )

            why_ranked = product.get(
                "why_ranked",
                "No ranking explanation."
            )

            print(
                f"  * {name}{badge}"
            )

            print(
                f"      {price} | "
                f"{source} | "
                f"{material}"
            )

            print(
                f"      why ranked: "
                f"{why_ranked}"
            )

    # ========================================================
    # CHEAPEST FOUND
    # ========================================================

    cheapest = result.get("cheapest_found")

    if cheapest:

        print(
            "\n💰 Cheapest found overall: "
            f"{cheapest.get('name', 'Unknown')} — "
            f"{cheapest.get('price', 'n/a')} "
            f"({cheapest.get('source', 'Unknown')})"
        )

        print(
            "   (cheapest among the alternatives found — "
            "not a same-item price match across sellers)"
        )

    # ========================================================
    # MATERIAL GROUPS
    # ========================================================

    materials = result.get("materials", [])

    matching = [
        material
        for material in materials
        if material.get("matches_request")
    ]

    other = [
        material
        for material in materials
        if not material.get("matches_request")
    ]

    # ========================================================
    # HIGH-IMPACT REQUEST
    # ========================================================

    if request_sustainability == "high_impact":

        print("\n" + "=" * 70)
        print("--- BETTER ALTERNATIVES ---")
        print("=" * 70)

        if not other:

            print(
                "No alternative material groups returned."
            )

        else:

            for material in other:
                print_material(material)

    # ========================================================
    # ECO-LEANING REQUEST
    # ========================================================

    elif (
        request_sustainability == "eco_leaning"
        and matching
    ):

        print("\n" + "=" * 70)
        print("--- WHAT YOU ASKED FOR ---")
        print("=" * 70)

        for material in matching:
            print_material(material)

        print("\n" + "=" * 70)
        print("--- OTHER ALTERNATIVES ---")
        print("=" * 70)

        if other:

            for material in other:
                print_material(material)

        else:

            print("No other alternatives returned.")

    # ========================================================
    # NORMAL DISCOVERY
    # ========================================================

    else:

        print("\n" + "=" * 70)
        print("--- BY MATERIAL / ALTERNATIVE TYPE ---")
        print("=" * 70)

        if not materials:

            print("No material groups returned.")

        else:

            for material in materials:
                print_material(material)

    # ========================================================
    # META
    # ========================================================

    print("\n" + "=" * 70)
    print("--- META ---")
    print("=" * 70)

    meta = result.get("meta", {})

    print(meta)

    # ========================================================
    # SELF CRITIQUE
    # ========================================================

    critique = meta.get("critique")

    if critique:

        print("\n" + "=" * 70)
        print("--- SELF-CRITIQUE ---")
        print("=" * 70)

        print(
            f"  Passed: "
            f"{critique.get('passed', False)}"
        )

        issues = critique.get("issues", [])

        if issues:

            print("  Issues:")

            for issue in issues:
                print(
                    f"  - {issue}"
                )

        else:

            print("  Issues: none")

        print(
            "  Extra search used: "
            f"{critique.get('extra_search_used', False)}"
        )

    # ========================================================
    # NO SUITABLE ALTERNATIVE
    # ========================================================

    if result.get("no_suitable_alternative"):

        print(
            "\n⚠ No suitable alternative found."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # Combine all command-line arguments into one query.
    query = (
        " ".join(sys.argv[1:])
        or "plastic mug"
    )

    print(
        f"\n🌱 GreenSwap"
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Preferences: {PREFERENCES}"
    )

    print(
        f"Max price: "
        f"{MAX_PRICE if MAX_PRICE is not None else 'none'}"
    )

    print(
        f"Sort: {SORT_BY}"
    )

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    raw = run_agent(
        query,
        preferences=PREFERENCES,
        max_price=MAX_PRICE
    )

    # --------------------------------------------------------
    # APPLY RANKING / FINAL UI TRANSFORMATION
    # --------------------------------------------------------

    view = finalize(
        raw,
        preferences=PREFERENCES,
        max_price=MAX_PRICE,
        sort_by=SORT_BY
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    show(view)

    # --------------------------------------------------------
    # SAVE RAW RUN FOR REPLAY
    # --------------------------------------------------------

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        query.lower()
    ).strip("-")

    if not slug:
        slug = "query"

    demo_dir = Path("demo_runs")

    demo_dir.mkdir(
        exist_ok=True
    )

    out_path = (
        demo_dir
        / f"{slug}.json"
    )

    out_path.write_text(
        json.dumps(
            raw,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"\n💾 Saved raw run to:"
        f" {out_path}"
    )