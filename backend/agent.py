"""
GreenSwap: core agent loop (v4).

Two phases, deliberately separated:

  PHASE 1 — GATHER
    The model decides what to search and when it has enough (real
    agent loop, tools available). Every shopping listing gets an id
    (p1, p2, ...); the model only ever refers to products BY ID, so it
    cannot invent or alter a price/link. Once MAX_SEARCH_CALLS is hit,
    we stop asking the model for more turns in this phase.

  PHASE 2 — FINALIZE
    A brand-new, tool-free conversation (no tool-call history at all)
    that hands the model everything gathered and asks for the final
    JSON. This exists because of a real bug: gpt-oss on Groq will
    still attempt a tool call based on the tool being *described in
    the system prompt text*, even when "tools" is left out of the
    request — and Groq then rejects that as invalid. The fix isn't
    "suppress it harder", it's "give it a conversation with nothing to
    be primed by".

  PHASE 3 — VALIDATE (see evidence.py)
    Any per-product "claims" the model attached get checked in plain
    code against the text we actually retrieved. This never trusts the
    model's self-reported status.

Caps are enforced in code, never trusted to the model:
    MAX_SEARCH_CALLS  real SerpApi calls per run
    MAX_LLM_TURNS     ceiling on phase-1 turns

Ranking/filtering is NOT done here — see ranking.py (pure code).
Preferences (e.g. low_cost) steer RANKING, not the search text.
"""

import os
import json

from groq import Groq
from serp_tool import serp_search
from evidence import validate_claims

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-120b"

MAX_SEARCH_CALLS = 5
MAX_LLM_TURNS = 8

DEBUG = os.environ.get("AGENT_DEBUG", "0") == "1"

AVAILABILITY_NOTE = (
    "Availability shown is based on regional search results and may not "
    "reflect real-time shipping to your exact location. Please verify on "
    "the merchant's page before purchasing."
)

# Shared between the gather prompt and the finalize prompt, so the
# output shape and rules never drift between the two phases.
OUTPUT_RULES = """
Rules about what the user already asked for (READ FIRST):
- Check whether the user's product text already names a specific material or format. Classify "request_sustainability" as one of:
  - "eco_leaning": the named material is a genuine improvement over the conventional default — recycled/upcycled content, reclaimed/reused materials, rapidly renewable natural materials (bamboo, cork, jute, hemp, cane/rattan), or something inherently reusable replacing a single-use item.
  - "high_impact": the named material IS the conventional, higher-impact default — virgin/new plastic, single-use plastic, styrofoam/EPS, or something explicitly disposable/non-recyclable.
  - "none": the user did not name a specific material at all (e.g. just "chair", "mug").
- If "eco_leaning": set "user_specified_material" to that material. That material is the PRIMARY result — search it thoroughly, mark its material object "matches_request": true. Still explore 2-3 OTHER materials as optional alternatives, marked "matches_request": false.
- If "high_impact": set "user_specified_material" to that material (for reference in the summary only), and write a one-sentence factual "caution_note" (e.g. "Virgin plastic is made from new fossil-fuel feedstock and has a higher environmental footprint; here are better options."). Do NOT search for or include this material as a product listing at all — do not spend any search on it. Instead, spend your full search budget finding 3-4 genuinely better materials, and set "matches_request": false on all of them, since none of them is "what was asked for" — they're what you're recommending instead.
- If "none": explore 3-5 different materials as usual, set "request_sustainability": "none", "user_specified_material": null, "caution_note": null, and "matches_request": false for all of them.

Rules about preferences and budget (READ CAREFULLY):
- Preferences like "low cost" or "reusable" and any stated budget are used LATER, in code, to rank and filter products. They are NOT search keywords, and must never appear inside a search query.

Rules about materials:
- Pick genuinely different material or format alternatives, not five variations of one material.
- When relevant, consider recycled-material versions (e.g. recycled plastic, recycled metal, reclaimed wood) as one of the alternatives.
- Only include a product under a material if it actually matches that material and the user's functional qualifiers (e.g. "microwave safe").
- Material-level "about"/"pros"/"cons" is general knowledge about the material itself — it is NOT a claim about any specific product, and must not be presented as verified fact about a listing.

Rules about products and claims:
- Only use product ids that were given to you. Never invent products, prices, ids or links.
- A listing is one product at one seller. Never claim two listings are the same product.
- Do not state that something is "eco-friendly" or "sustainable" as a fact.
- For a product, you may attach 0-2 "claims" — but ONLY if you can quote a short exact phrase (evidence_snippet) that actually appeared in that product's title or in a web result you were shown. If you cannot quote real supporting text, do not add the claim at all. Do not paraphrase the snippet — quote it as it appeared.
- If nothing suitable is found, return an empty "materials" list and explain why in "summary".

Reply with ONLY this JSON and no other text:

{
  "summary": "one sentence on what was found",
  "user_specified_material": "the material/format the user already named, or null",
  "request_sustainability": "eco_leaning | high_impact | none",
  "caution_note": "one factual sentence if high_impact, else null",
  "materials": [
    {
      "type": "short material or format name, e.g. 'cork-base ceramic'",
      "matches_request": true,
      "about": "one sentence on what it is (general knowledge)",
      "pros": ["1-3 short points (general knowledge)"],
      "cons": ["1-3 short points (general knowledge)"],
      "products": [
        {
          "id": "p3",
          "why_suggested": "one short sentence",
          "trade_off": "one short sentence",
          "claims": [
            {"claim": "short claim, e.g. 'made from recycled plastic'",
             "evidence_snippet": "exact short phrase copied from the title/result you saw"}
          ]
        }
      ]
    }
  ]
}

Choose 1-4 products per material.
"""

GATHER_PROMPT = f"""You are a sustainability research agent for GreenSwap, helping shoppers in India.

Given a product the user normally buys (plus optional preferences and budget), find genuinely different, more sustainable alternatives and real products for them.

You have one tool: search(query, engine).
- engine="shopping": real product listings with price and seller. Every listing has an id like "p3".
- engine="web": general web results. Use it to discover current alternatives or check a specific claim.

Strategy:
1. Your own knowledge of sustainable materials may be out of date, since new materials and products appear all the time. START with ONE web search to discover current sustainable alternatives for this product (for example "sustainable alternatives to <product> new materials"). Use the results, including any "Related searches" entry, as candidate materials.
2. Combine those candidates with what you already know. Pick 3-5 genuinely different alternatives.
3. Run about one shopping search per alternative to find real listings. If a candidate has no matching listings, drop it and mention that in "summary".
4. If results are weak, try a different angle instead of repeating the same query. Stop as soon as you have enough — you do not need to use every search.
5. Carry the user's functional qualifiers (for example "microwave safe", a size, a use-case) into the shopping queries. NEVER put price/budget words into a query — see the rules below.
{OUTPUT_RULES}
When you have gathered enough, and ONLY when you are not calling any more tools, reply with the final JSON described above.
"""

FINALIZE_PROMPT = f"""You are finishing a sustainability research task for GreenSwap. All searching is already done — you cannot search again. Use ONLY the product ids and web findings given to you below.
{OUTPUT_RULES}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search live web or shopping data via SerpApi, localized to India.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query to run"},
                    "engine": {
                        "type": "string",
                        "enum": ["shopping", "web"],
                        "description": "'shopping' for product listings, 'web' for general info/discovery",
                    },
                },
                "required": ["query", "engine"],
            },
        },
    }
]


def _tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_msg(msg) -> dict:
    """
    Minimal dict form of a Groq assistant message, safe to send back.
    Deliberately NOT msg.model_dump() — gpt-oss/Groq responses can
    include fields (e.g. "annotations") the API rejects on the way in.
    """
    message = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        message["tool_calls"] = [
            {"id": tc.id, "type": tc.type,
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return message


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _resolve(parsed: dict, catalog: dict) -> tuple[list[dict], int]:
    """Turn the model's id-only answer into full product objects using data the backend fetched itself."""
    materials, dropped, seen = [], 0, set()

    for m in parsed.get("materials", []) or []:
        products = []
        for p in m.get("products", []) or []:
            pid = p.get("id")
            base = catalog.get(pid)
            if base is None:
                dropped += 1
                continue
            if pid in seen:
                continue
            seen.add(pid)
            products.append(
                {
                    "id": pid,
                    "name": base["title"],
                    "price": base["price"],
                    "source": base["source"],
                    "link": base["link"],
                    "image": base["image"],
                    "rating": base["rating"],
                    "reviews": base["reviews"],
                    "material_type": m.get("type"),
                    "why_suggested": p.get("why_suggested"),
                    "trade_off": p.get("trade_off"),
                    "claims": p.get("claims", []) or [],  # validated in phase 3
                }
            )
        if products:
            materials.append(
                {
                    "type": m.get("type"),
                    "matches_request": bool(m.get("matches_request", False)),
                    "about": m.get("about"),
                    "pros": m.get("pros", []),
                    "cons": m.get("cons", []),
                    "info_source": "general knowledge",
                    "products": products,
                }
            )
    return materials, dropped


def _debug(response) -> None:
    if not DEBUG:
        return
    msg = response.choices[0].message
    print("\n--- GROQ RESPONSE ---")
    print("content:", repr(msg.content))
    print("tool_calls:", msg.tool_calls)
    print("finish_reason:", response.choices[0].finish_reason)
    print("---------------------\n")


def _finalize_clean(
    product_query: str, prefs_text: str, budget_text: str,
    catalog: dict, web_findings: list[dict],
) -> dict | None:
    """
    Phase 2. Brand-new conversation, no tools param, no tool-call
    history — nothing for the model to be primed by, which is what
    the gather-phase crash was actually caused by.
    """
    products_seen = [
        {"id": pid, "title": v["title"], "price": v["price"], "source": v["source"],
         "rating": v["rating"], "reviews": v["reviews"]}
        for pid, v in catalog.items()
    ]

    user_content = (
        f"Product: {product_query}\n"
        f"Preferences (for ranking later, NOT search keywords): {prefs_text}\n"
        f"Budget (for filtering later, NOT search keywords): {budget_text}\n\n"
        f"Product listings found (JSON):\n{json.dumps(products_seen, ensure_ascii=False)}\n\n"
        f"Web research notes (JSON):\n{json.dumps(web_findings, ensure_ascii=False)}\n\n"
        "Produce the final JSON now."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FINALIZE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    _debug(response)
    return _parse_json(response.choices[0].message.content)


def run_agent(
    product_query: str,
    preferences: list[str] | None = None,
    max_price: float | None = None,
) -> dict:
    prefs_text = (
        ", ".join(p.replace("_", " ") for p in preferences)
        if preferences else "none stated (give a balanced set)"
    )
    budget_text = f"under ₹{int(max_price)}" if max_price else "none stated"

    messages = [
        {"role": "system", "content": GATHER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Product: {product_query}\n"
                f"Preferences (for ranking later, NOT search keywords): {prefs_text}\n"
                f"Budget (for filtering later, NOT search keywords): {budget_text}"
            ),
        },
    ]

    trace: list[dict] = []
    catalog: dict[str, dict] = {}
    web_findings: list[dict] = []
    search_calls = 0
    next_id = 1
    parsed = None

    # ---------------- PHASE 1: GATHER ----------------
    for _ in range(MAX_LLM_TURNS):
        if search_calls >= MAX_SEARCH_CALLS:
            break  # don't ask the model for another gather-phase turn; go straight to finalize

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.3,
        )
        msg = response.choices[0].message
        _debug(response)
        messages.append(_assistant_msg(msg))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if search_calls >= MAX_SEARCH_CALLS:
                    messages.append(_tool_msg(tc.id, "Search cap reached."))
                    continue

                try:
                    args = json.loads(tc.function.arguments or "{}")
                    query = args["query"]
                    engine = args.get("engine", "shopping")
                    if engine not in ("shopping", "web"):
                        engine = "shopping"
                except (json.JSONDecodeError, KeyError):
                    messages.append(_tool_msg(tc.id, "Invalid arguments. Call search with a query and an engine."))
                    continue

                try:
                    results, cached = serp_search(query, engine)
                except Exception as e:
                    search_calls += 1
                    trace.append({"step": search_calls, "query": query, "engine": engine,
                                  "results_found": 0, "cached": False, "error": str(e)})
                    messages.append(_tool_msg(tc.id, f"Search failed: {e}"))
                    continue

                search_calls += 1
                trace.append({"step": search_calls, "query": query, "engine": engine,
                              "results_found": len(results), "cached": cached})

                if engine == "shopping":
                    model_view = []
                    for item in results:
                        pid = f"p{next_id}"
                        next_id += 1
                        catalog[pid] = item
                        model_view.append({"id": pid, "title": item["title"], "price": item["price"],
                                            "source": item["source"], "rating": item["rating"],
                                            "reviews": item["reviews"]})
                    payload = model_view
                else:
                    web_findings.append({"query": query, "results": results})
                    payload = results

                messages.append(_tool_msg(tc.id, json.dumps(payload, ensure_ascii=False)))
            continue

        parsed = _parse_json(msg.content)
        if parsed is not None:
            break
        messages.append({"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object described in the instructions."})

    # ---------------- PHASE 2: FINALIZE (if gather phase didn't already finish cleanly) ----------------
    if parsed is None:
        parsed = _finalize_clean(product_query, prefs_text, budget_text, catalog, web_findings)

    if parsed is None:
        return {
            "summary": "The agent did not produce a valid answer. Please try again.",
            "materials": [], "availability_note": AVAILABILITY_NOTE,
            "no_suitable_alternative": True, "trace": trace,
            "meta": {"searches": search_calls, "dropped_unknown_ids": 0, "error": "no_valid_json"},
        }

    materials, dropped = _resolve(parsed, catalog)

    # Safety net, not just a prompt request: a high-impact material must
    # NEVER appear as a shown product, even if the model didn't comply.
    if parsed.get("request_sustainability") == "high_impact":
        materials = [m for m in materials if not m.get("matches_request")]

    # ---------------- PHASE 3: VALIDATE (pure code, see evidence.py) ----------------
    materials = validate_claims(materials, web_findings)

    return {
        "summary": parsed.get("summary", ""),
        "user_specified_material": parsed.get("user_specified_material"),
        "request_sustainability": parsed.get("request_sustainability", "none"),
        "caution_note": parsed.get("caution_note"),
        "materials": materials,
        "availability_note": AVAILABILITY_NOTE,
        "no_suitable_alternative": len(materials) == 0,
        "trace": trace,
        "meta": {"searches": search_calls, "dropped_unknown_ids": dropped},
    }