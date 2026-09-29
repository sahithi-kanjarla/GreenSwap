"""
Run the agent end to end from the terminal.

    python test_agent.py "mug"
    python test_agent.py "microwave safe mug"
    python test_agent.py "kitchen scrubber"

The first run of a new query uses up to 5 SerpApi credits. Repeating
the same query hits the disk cache (0 credits). The raw result is saved
to demo_runs/<query>.json so replay mode can use it later.
"""

import re
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # must run before importing serp_tool (it reads SERPAPI_KEY)

from agent import run_agent
from ranking import finalize

PREFERENCES = ["low_cost"]   # try [] or ["reusable"] too
MAX_PRICE = None             # try 500 to see the budget filter


def show(result: dict) -> None:
    print("\n--- TRACE ---")
    for s in result["trace"]:
        tag = "cache" if s.get("cached") else "LIVE "
        err = f"  ERROR: {s['error']}" if s.get("error") else ""
        print(f"  {s['step']}. [{s['engine']}/{tag}] \"{s['query']}\" -> {s['results_found']} results{err}")

    print("\n--- SUMMARY ---")
    print(result["summary"])
    req = result.get("request_sustainability", "none")
    if result.get("user_specified_material"):
        print(f"(You asked for: {result['user_specified_material']} — classified as {req})")
    if result.get("caution_note"):
        print(f"\n⚠ {result['caution_note']}")

    print("\n--- TOP PICKS ---")
    for p in result["top_picks"]:
        print(f"  * {p['name']}")
        print(f"      {p['price']} | {p['source']} | {p['material_type']}")
        print(f"      why ranked: {p['why_ranked']}")

    matching = [m for m in result["materials"] if m.get("matches_request")]
    other = [m for m in result["materials"] if not m.get("matches_request")]

    def print_material(m):
        print(f"\n[{m['type']}] {m['about']}")
        print("   pros:", "; ".join(m.get("pros", [])))
        print("   cons:", "; ".join(m.get("cons", [])))
        for p in m["products"]:
            print(f"   - {p['name']} | {p['price']} | {p['source']}")
            print(f"       {p['why_suggested']}  (trade-off: {p['trade_off']})")
            for c in p.get("claims", []):
                mark = "✓" if c["validated"] else "?"
                print(f"       {mark} [{c['status']}] {c['claim']}  <- \"{c['evidence_snippet']}\"")

    if req == "high_impact":
        print("\n--- BETTER ALTERNATIVES ---")
        for m in other:
            print_material(m)
    elif req == "eco_leaning" and matching:
        print("\n--- WHAT YOU ASKED FOR ---")
        for m in matching:
            print_material(m)
        print("\n--- OTHER ALTERNATIVES ---")
        for m in other:
            print_material(m)
    else:
        print("\n--- BY MATERIAL ---")
        for m in result["materials"]:
            print_material(m)

    print("\n--- META ---")
    print(result["meta"])
    if result["no_suitable_alternative"]:
        print("\nNo suitable alternative found.")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "plastic mug"

    raw = run_agent(query, preferences=PREFERENCES, max_price=MAX_PRICE)
    view = finalize(raw, preferences=PREFERENCES, max_price=MAX_PRICE)
    show(view)

    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    Path("demo_runs").mkdir(exist_ok=True)
    out_path = Path("demo_runs") / f"{slug}.json"
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved raw run to {out_path}")