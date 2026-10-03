"""
GreenSwap: core agent loop (v6).

v6 adds:
  - preservation of user context in alternative shopping queries;
  - code-derived shopping signals (Popular / Offer / Lowest price found);
  - stricter separation between real functional mismatches and ordinary trade-offs.


Four phases, deliberately separated:

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

  PHASE 3 — VALIDATE / ENFORCE (see evidence.py, product_match.py)
    Claims are checked against text actually retrieved. Product type
    is checked against the listing's own title. A high-impact material
    is stripped from output even if the model didn't comply. None of
    this is trusted to the model alone.

  PHASE 4 — SELF-CRITIQUE
    One extra Groq call (zero SerpApi cost) reviews the finished draft
    against a checklist. It is explicitly told about our own business
    rules (e.g. a high-impact material being deliberately absent is
    correct, not a gap) so it doesn't flag intended behavior as a bug.
    If it finds a genuine, fixable issue and search budget remains, it
    can trigger exactly ONE more search and one rebuild — never a loop.

Caps are enforced in code, never trusted to the model:
    MAX_SEARCH_CALLS  real SerpApi calls per run
    MAX_LLM_TURNS     ceiling on phase-1 turns

Ranking/filtering is NOT done by the search agent. Retrieved results can be filtered/sorted after the run without another SerpApi call; see ranking.py (pure code). Preferences steer ranking/filtering, not search text.
"""

import os
import json
import re
from urllib.parse import urlparse, parse_qs, unquote

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "Google GenAI SDK is missing. Install it in the backend environment with: "
        "python -m pip install google-genai"
    ) from exc
from serp_tool import serp_search
from evidence import validate_claims
from product_match import filter_by_product_type

# Groq temporarily disabled while testing Gemini.
# from groq import Groq
# client = Groq(api_key=os.environ["GROQ_API_KEY"])
# MODEL = "openai/gpt-oss-120b"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.8-flash"

MAX_SEARCH_CALLS = 5
MAX_LLM_TURNS = 8

# This is the direct fix for two problems that are actually ONE problem:
# "too few products shown" and "evidence search adds latency". Both come
# from evidence searches (purpose="evidence") competing with discovery
# searches (purpose="shopping"/alternative-finding) for the same shared
# budget, with no guaranteed split — the model's own discretion decided
# how much each got, and it was spending too much on evidence, starving
# discovery. This sub-cap guarantees discovery always gets at least
# MAX_SEARCH_CALLS - MAX_EVIDENCE_CALLS calls, regardless of what the
# model tries to do; evidence search literally cannot use more than this,
# enforced in code below (not just requested in the prompt).
MAX_EVIDENCE_CALLS = 1

DEBUG = os.environ.get("AGENT_DEBUG", "0") == "1"

AVAILABILITY_NOTE = (
    "Availability shown is based on regional search results and may not "
    "reflect real-time shipping to your exact location. Please verify on "
    "the merchant's page before purchasing."
)

# Prefer the actual merchant URL returned by the shopping provider.
# A generic shopping/search URL is only a fallback when no direct product URL exists.
LINK_TYPE = "merchant_product"
LINK_NOTE = (
    "Opens the merchant/product listing when an actual product URL was returned. "
    "If no direct listing URL is available, the fallback is marked as a search."
)


def _looks_like_search_url(value: str | None) -> bool:
    """Return True when a URL looks like a generic search rather than a product page."""
    if not value:
        return False
    text = str(value).lower()
    return (
        ("google." in text and ("/search" in text or "tbm=shop" in text))
        or "search?" in text
        or "/search/" in text
    )


def _best_product_link(item: dict) -> tuple[str | None, str, str]:
    """Prefer a real merchant product URL and never pretend a search URL is a product page."""
    direct = item.get("product_link")
    if direct and not _looks_like_search_url(direct):
        return direct, "merchant_product", "Direct product listing"

    generic = item.get("link")
    if generic and not _looks_like_search_url(generic):
        return generic, "merchant_product", "Direct product listing"

    if generic:
        return generic, "shopping_search", "Shopping search fallback — direct product URL was not available"

    return None, "unavailable", "Direct product URL was not available"

AMAZON_HOSTS = {
    "amazon.in", "www.amazon.in", "amazon.com", "www.amazon.com",
    "amazon.co.uk", "www.amazon.co.uk", "amazon.ca", "www.amazon.ca",
    "amazon.de", "www.amazon.de", "amazon.fr", "www.amazon.fr",
    "amazon.it", "www.amazon.it", "amazon.es", "www.amazon.es",
    "amazon.co.jp", "www.amazon.co.jp", "amazon.com.au", "www.amazon.com.au",
}

FLIPKART_HOSTS = {"flipkart.com", "www.flipkart.com"}


def _clean_input(value: str) -> str:
    return (value or "").strip()


def _is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _host(value: str) -> str:
    try:
        return urlparse(value).netloc.lower().split(":")[0]
    except Exception:
        return ""


def _extract_amazon_asin(value: str) -> str | None:
    if not _is_url(value):
        return None
    host = _host(value)
    if host not in AMAZON_HOSTS:
        return None
    parsed = urlparse(value)
    path = unquote(parsed.path)
    patterns = [
        r"/(?:dp|gp/product|gp/aw/d|product)/([A-Z0-9]{10})(?:[/?]|$)",
        r"/([A-Z0-9]{10})(?:[/?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    query = parse_qs(parsed.query)
    for key in ("asin", "ASIN"):
        values = query.get(key)
        if values:
            candidate = values[0].strip().upper()
            if re.fullmatch(r"[A-Z0-9]{10}", candidate):
                return candidate
    return None


def _url_input_context(value: str) -> dict:
    if not _is_url(value):
        return {"is_url": False, "source": None, "asin": None, "url": None}
    host = _host(value)
    source = "amazon" if host in AMAZON_HOSTS else ("flipkart" if host in FLIPKART_HOSTS else "web")
    return {"is_url": True, "source": source, "asin": _extract_amazon_asin(value), "url": value}


def _url_search_query(value: str) -> str:
    context = _url_input_context(value)
    asin = context.get("asin")
    source = context.get("source")
    if source == "amazon" and asin:
        return f'Amazon product {asin} product details material specifications'
    parsed = urlparse(value)
    path_text = unquote(parsed.path).replace("/", " ").replace("-", " ").replace("_", " ")
    path_text = re.sub(r"\s+", " ", path_text).strip()
    if path_text:
        return f'"{path_text[:180]}" product material specifications'
    return f'"{value}" product details'


def _augment_shopping_query(query: str, user_request: str) -> str:
    """Preserve meaningful user context when the model drops it from a shopping query."""
    if not query or not user_request or _is_url(user_request):
        return query.strip()

    # Buying/ranking words are handled separately and should not leak into
    # discovery queries. Functional/use-case words are preserved.
    ignored = {
        "a", "an", "the", "for", "to", "of", "and", "or", "with",
        "my", "me", "i", "need", "want", "looking", "find", "show",
        "get", "buy", "best", "good", "cheap", "cheapest", "budget",
        "affordable", "price", "priced", "under", "below", "less",
        "around", "cost", "costing", "rs", "inr",
    }

    def tokens(value: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"[a-zA-Z0-9]+", value)
            if len(token) > 1
        ]

    query_tokens = set(tokens(query))
    missing = []
    for token in tokens(user_request):
        if token in ignored:
            continue
        if token not in query_tokens and token not in missing:
            missing.append(token)

    if not missing:
        return query.strip()

    # This is only a safety net. The LLM still chooses the alternative/approach.
    return f"{query.strip()} {' '.join(missing[:8])}".strip()


def _numeric_reviews(value) -> int | None:
    if value in (None, "", "N/A"):
        return None
    match = re.search(r"[\d,]+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _numeric_price(value) -> float | None:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _offer_signal(item: dict) -> dict | None:
    """Detect an offer only from actual current + old prices returned by shopping."""
    current = _numeric_price(item.get("extracted_price", item.get("price")))
    old = _numeric_price(item.get("extracted_old_price", item.get("old_price")))

    if current is None or old is None or old <= current:
        return None

    savings = old - current
    return {
        "is_offer": True,
        "current_price": current,
        "old_price": old,
        "savings": savings,
        "savings_percent": round((savings / old) * 100) if old else None,
        "basis": "current and previous prices returned by shopping data",
    }


def _add_shopping_signals(catalog: dict[str, dict]) -> None:
    """
    Compute buying signals after discovery.

    Popular = high review volume relative to the listings found in this run.
    It is deliberately NOT called 'most sold' because review count is not
    sales volume.
    """
    review_values = [
        _numeric_reviews(item.get("reviews"))
        for item in catalog.values()
    ]
    review_values = [v for v in review_values if v is not None and v >= 0]

    threshold = None
    if review_values:
        ordered = sorted(review_values)
        index = max(0, int((len(ordered) - 1) * 0.75))
        threshold = ordered[index]

    valid_prices = [
        _numeric_price(item.get("extracted_price", item.get("price")))
        for item in catalog.values()
    ]
    valid_prices = [v for v in valid_prices if v is not None]
    lowest_price = min(valid_prices) if valid_prices else None

    for item in catalog.values():
        reviews = _numeric_reviews(item.get("reviews"))
        badges = []

        if threshold is not None and reviews is not None and reviews >= threshold:
            badges.append("Popular")

        explicit_bestseller = bool(
            item.get("is_bestseller")
            or item.get("bestseller")
            or item.get("best_seller")
        )
        if explicit_bestseller:
            badges.append("Bestseller")

        offer = _offer_signal(item)
        if offer:
            badges.append("Offer")

        current = _numeric_price(item.get("extracted_price", item.get("price")))
        if lowest_price is not None and current == lowest_price:
            badges.append("Lowest price found")

        item["shopping_signals"] = {
            "badges": badges,
            "popular": "Popular" in badges,
            "bestseller": "Bestseller" in badges,
            "bestseller_basis": (
                "explicit bestseller signal returned by shopping data"
                if "Bestseller" in badges else None
            ),
            "popular_basis": (
                "high review volume among listings found in this run"
                if "Popular" in badges else None
            ),
            "offer": offer,
            "lowest_price_in_run": "Lowest price found" in badges,
        }


OUTPUT_RULES = """
Rules about understanding the user's request (READ FIRST):

- If the user input is a product URL, treat it as a BEFORE-YOU-BUY request.
  First identify the product/category from retrieved information, then analyze the
  characteristics that matter for that product, discover relevant alternatives,
  and explain evidence-backed considerations and trade-offs. Do not invent product
  details merely from the URL.

- Identify the core product_type from the user's natural-language request.
  product_type means what the user is actually trying to buy, not its
  material, sustainability qualifier, use case, or budget.

  Examples:
    "mug" -> "mug"
    "microwave safe mug" -> "mug"
    "ceramic mug under ₹500" -> "mug"
    "recycled plastic chair for outdoor use" -> "chair"
    "something to clean my toilet without harsh chemicals" -> "toilet cleaner"

- Preserve important functional qualifiers and user requirements for search.
  Examples include microwave-safe, outdoor use, size, capacity, intended use,
  fragrance-free, etc.

- Extract the important functional requirements implied by the user's request.
  These are separate from sustainability preferences. Functional requirements
  can include intended use, setting/context, performance, durability, longevity,
  safety, compatibility, size, capacity, appearance, maintenance, frequency of
  use, replacement frequency, exposure/contact considerations, or any other
  requirement that materially affects whether a product is suitable.

- When the intended use creates an additional safety, compatibility, or exposure
  requirement, treat that requirement explicitly. Do not infer suitability merely
  because the product belongs to the right broad category. If the available
  listing/evidence cannot establish an important requirement, say that it is
  unclear or exclude the product when the uncertainty materially affects use.

- Do NOT use a fixed category-specific list to determine functional requirements.
  Infer them from the user's actual wording and context.

- Keep these requirements in mind throughout search, product selection, evidence
  investigation, and finalization. An environmental characteristic must never
  override an important functional requirement.

- Check whether the user explicitly names a material, format, or environmental
  preference.

- Classify "request_sustainability" as one of:

  - "eco_leaning":
    The user explicitly selected a material, format, or characteristic that
    appears environmentally preferable in the context of THIS product.
    Determine this from the product context and the meaning of the request.
    Do NOT use a fixed list of "green" materials or keywords.

  - "high_impact":
    The user explicitly selected a material, format, or approach that the
    agent reasonably identifies as comparatively high-impact for THIS product
    category. Reason from the product context; do not match against a fixed
    hardcoded list.

  - "none":
    The user did not explicitly select a material, format, or environmental
    preference.

- If "eco_leaning":
  set "user_specified_material" to the material/format/environmental preference
  the user explicitly requested. Make it the PRIMARY search direction.
  You may still discover genuinely different alternatives if useful.

- If "high_impact":
  set "user_specified_material" to the requested material/format for reference.
  Write a short factual "caution_note".
  Do NOT recommend the requested high-impact option as the environmentally
  preferable result. Discover better alternatives instead.

- If "none":
  dynamically discover environmentally relevant alternatives for the specific
  product category.

Rules for environmental reasoning:

- GreenSwap is an environmental-alternative discovery agent, NOT a scientific
  sustainability scoring system.

- There is NO universal fixed list of things that count as sustainable.

- For every request, first reason about which environmental dimensions could
  actually matter for THIS specific product.

- Possible dimensions include, but are NOT limited to:
    material origin and resource use
    recycled/recovered/reclaimed material
    renewable material or resource
    reuse/reusability
    refill/concentrated formats
    packaging and material waste
    end-of-life characteristics
    recyclability
    biodegradability/compostability
    formulation or ingredient origin
    organic production/formulation
    plant-derived alternatives
    longevity
    repairability
    replacement frequency
    another product-specific environmental dimension the agent identifies

- These are reasoning dimensions, NOT a mandatory checklist and NOT universal
  search keywords.

- Decide which 1-3 dimensions are actually meaningful for the current product
  before deciding what alternatives to search.

- Do NOT force material substitution when material is not the meaningful
  environmental dimension. For some products, formulation, packaging,
  concentration, refillability, longevity, repairability, or end-of-life may
  matter more.

- An attribute is NOT automatically an environmental benefit just because it
  sounds positive.

  In particular:
    recycled does not automatically make the whole product environmentally
    preferable;
    refillable does not automatically make the whole product environmentally
    preferable;
    reusable does not automatically make the whole product environmentally
    preferable;
    organic does not automatically make the whole product environmentally
    preferable;
    plant-based does not automatically make the whole product environmentally
    preferable;
    biodegradable does not automatically make the whole product
    environmentally preferable;
    non-toxic does not automatically make the whole product environmentally
    preferable;
    durable does not automatically make the whole product environmentally
    preferable.

- Do NOT automatically exclude organic, plant-based, biodegradable,
  refillable, or non-toxic options. They may be highly relevant for some
  categories. Determine their relevance from the actual request and product
  context.

- Never infer one property from another:
    plant-based != automatically non-toxic
    organic != automatically biodegradable
    non-toxic != automatically environmentally preferable
    refillable != automatically environmentally preferable
    durable != automatically environmentally preferable

- Consider the overall product/alternative rather than recommending a product
  solely because it contains one "green-sounding" feature.

- For material-based products, recovered/recycled/reclaimed materials can be
  relevant because they may reduce reliance on virgin material. Still evaluate
  the actual product and available evidence.

- For consumable or formulation products, environmental considerations may be
  more about formulation, ingredients, concentration, refill format,
  packaging, waste, or end-of-life than about physical material substitution.

- The agent may identify environmental dimensions not listed above. This list
  is not exhaustive.

- If no meaningful environmental advantage can be reasonably identified or
  supported by the available evidence, do not invent one. It is acceptable to
  return fewer alternatives or no suitable alternative.

Rules about search planning:

- The LLM decides which environmental dimensions are worth spending SerpApi
  calls on for the current request.

- Do NOT search every possible dimension.

- Do NOT append a universal set of words such as "organic", "non-toxic",
  "biodegradable", "plant-based", "eco-friendly", "recycled", or "refillable"
  to every query.

- First reason about the product, then choose the most useful alternative
  approaches, then generate search queries from those approaches.

- Search queries should combine the actual product_type with the selected
  environmental approach and preserve important functional qualifiers.

- IMPORTANT: Every shopping/discovery query must preserve the user's relevant
  use-case and functional context. For example, if the request is
  "water bottle for office", a query such as "recycled plastic water bottle"
  must still preserve the office/use context. Do not drop a qualifier merely
  because the alternative name changed.

- The backend also checks discovery shopping queries for dropped user context
  and appends missing meaningful context as a safety net. This does not choose
  the alternative for you.

- Use observations from previous searches to decide what to search next.
  If a search produces strong alternatives, move on. If an important
  dimension is weak or missing, use the next search to investigate it.

- Carry functional qualifiers into shopping queries whenever relevant.

- When the request contains important functional requirements, preserve the
  requirements that materially affect suitability in shopping queries whenever
  doing so improves the search. Do not search only for an environmental or
  material characteristic when the user's functional need is equally important.

- If a product's intended use implies a safety, compatibility, contact, exposure,
  performance, or other non-environmental constraint, investigate that constraint
  when it is material to choosing between alternatives. Keep the reasoning
  product-specific and evidence-based; do not use a hardcoded category checklist.

- When durability, longevity, or reduced replacement frequency is a requirement,
  treat it as a functional requirement. Search for product-specific evidence
  relevant to useful life when possible rather than relying only on generic
  material-level claims.

Rules about preferences and budget:

- Preferences and budget are used LATER for ranking/filtering.
- They are NOT sustainability dimensions unless the user explicitly makes
  them part of the environmental request.
- Never put price/budget words into a search query.

Rules about alternative groups:

- Pick genuinely different alternative types, materials, or approaches.
  Do not create several near-duplicates merely to fill slots.

- The number and type of alternatives should depend on what the agent finds.
  Do NOT force exactly 3, 4, or 5 alternatives when evidence does not
  support that many.

- Only include a product under an alternative if it actually matches that
  alternative AND is the SAME product type the user asked for.

- Functional qualifiers from the user's request must also be respected where
  they are relevant.

- If a promising environmental alternative has no suitable real product
  listing, do not force in a mismatched product. Either drop it and explain
  why, or use one targeted search if that could realistically find a suitable
  product.

- From each shopping search, include EVERY genuinely matching, genuinely
  distinct listing under its alternative — not just one or two. The search
  already cost a call; do not discard good inventory you already paid for.
  Only exclude a listing if it fails the product-type match or an important
  functional requirement, or is a near-exact duplicate of one already
  included.

Rules about functional suitability of alternatives:

- Before selecting an alternative, identify the important functional requirements
  implied by the user's request.

- Evaluate every candidate against those requirements before including it in the
  final answer.

- An environmental characteristic alone is NOT enough reason to recommend an
  alternative. The alternative must also reasonably satisfy the user's actual
  need.

- Use the available product title, listing information, specifications, and
  targeted web evidence to determine whether an important requirement is
  supported, contradicted, or unclear.

- Do not assume that a product satisfies an important requirement merely because
  the requirement is not contradicted by the listing.

- If an important functional requirement cannot reasonably be established, do
  not present the product as a confident match. Exclude it when the available
  evidence indicates a mismatch; otherwise make the limitation clear.

- Do not keep a product merely because it has stronger environmental/material
  evidence if it appears unsuitable for the user's actual need.

- Do not force recycled, reclaimed, renewable, organic, plant-based, non-toxic,
  refillable, biodegradable, reusable, durable, plastic-free, or any other
  positive-sounding characteristic into the search simply because it sounds
  environmentally preferable. Decide whether it is relevant to THIS product and
  use case.

- Do not create category-specific suitability rules. Determine suitability from
  the user's actual requirements and the evidence gathered for the current run.

- The same characteristic can be relevant for one product and irrelevant or unsuitable for another. For example, recycled or reclaimed materials may be especially relevant for furniture such as chairs or stools, while a recycled-material version of a food-contact item should only be considered when its intended use and available product information make that application appropriate. This is an example of reasoning, NOT a hardcoded category rule.

- Health/safety characteristics and environmental characteristics are different dimensions. For example, "non-toxic" may matter primarily because of human exposure or product safety; it should not automatically be treated as proof that a product is environmentally preferable. Likewise, "organic" or "plant-based" should not automatically be treated as an environmental verdict.

- If the intended use is ambiguous and that ambiguity materially changes whether an alternative is suitable, prefer broadly suitable alternatives or communicate the limitation rather than making a strong unsupported assumption.

- Do not force a fixed number of alternatives. Return only alternatives that are meaningfully relevant and supported by real listings. If only two make sense, return two; if none make sense, return none.

Rules about material and Impact descriptions:

- Each material/alternative has an "impact" object with FOUR short fields.
  These describe general contextual information, NOT independent verification
  of a specific product listing.

- "material_note": describe a specific environmental characteristic or
  potential effect, not an overall verdict.

- "reusability": describe the actual use pattern of the product or format.
  Only use "single-use" when the product is genuinely intended to be discarded
  after one use or one short use cycle. Products designed for repeated use or
  repeated wearing should be described as designed for repeated use/wearing.
  Do not infer durability merely from reusability.

- "end_of_life": describe end-of-life characteristics only when reasonably
  supported by general knowledge. Otherwise write exactly "Not verified".

- "common_trade_off": give one relevant downside or trade-off.

- Never write "this is sustainable", "this is eco-friendly", "this is green",
  or equivalent as an established fact.

- Prefer specific wording such as:
    "uses recovered material"
    "can reduce reliance on virgin material"
    "uses a refill format that may reduce packaging"
    "designed for reuse"
    "biodegradability was not verified"

- When the environmental effect depends on conditions, say so.

Rules about products and claims:

- Only use product ids that were given to you. Never invent products, prices,
  ids, ratings, reviews, or links.

- Do not write or fabricate a product URL in the JSON. Product links are attached
  later by code from the original shopping listing. If a listing has no direct
  product URL, the code must mark the link as unavailable or a search fallback.

- A listing is one product at one seller. Never claim two listings are the
  same product unless the available evidence supports that identity.

- Do not state that something is "eco-friendly" or "sustainable" as a fact.

- For a product, attach 0-2 important factual claims. Prefer claims that
  materially help the user understand why the product was selected.

- A claim must have an exact short evidence_snippet copied from the product
  listing/title or a web result that you were actually shown.

- Do not paraphrase the evidence_snippet.

- If a claim appears only in the shopping listing, it is still allowed, but
  it will be labeled "Stated by seller" by the code validator.

- If a claim is supported by a targeted web result, cite the exact snippet
  from that result. The validator will label it "Evidence found".

- A seller/manufacturer statement is still a seller/manufacturer claim.
  Finding the same claim on a search result does not automatically make it
  independently proven.

- Prefer official manufacturer documentation, certification bodies,
  technical/specification documents, ingredient information, or reputable
  independent sources when investigating important claims.

- Do not infer one claim from another.

  For example:
    "non-toxic" does not prove "environmentally friendly"
    "organic" does not prove "biodegradable"
    "refillable" does not prove "low environmental impact"
    "recycled" does not prove every environmental claim about the product

- If evidence is insufficient, do not create the claim. The system should be
  able to say that something was not verified.

- Evidence investigation (purpose="evidence") is capped separately from
  discovery and is RESERVED for your single highest-value product overall —
  not every product, not every material. Spend it where it will most change
  what the user sees, or skip it entirely if listing text + claims already
  make the strongest case for your top pick. Every other product is still
  shown with whatever claims its own listing text already supports
  (labeled "Stated by seller") — that is a complete, honest answer on its
  own and does not require a dedicated verification search.

- If nothing suitable is found, return an empty "materials" list and explain
  why in "summary".

Reply with ONLY this JSON and no other text:

{
  "summary": "one sentence on what was found",
  "product_type": "the core product category, 1-3 words, e.g. 'chair', 'mug', 'food container'",
  "functional_requirements": [
    "short, concrete requirement inferred from the user's request"
  ],
  "user_specified_material": "the material/format/environmental preference the user explicitly named, or null",
  "request_sustainability": "eco_leaning | high_impact | none",
  "caution_note": "one factual sentence if high_impact, else null",
  "materials": [
    {
      "type": "short alternative type/material/approach name",
      "matches_request": true,
      "about": "one sentence describing the alternative",
      "impact": {
        "material_note": "specific environmental characteristic or potential effect",
        "reusability": "specific statement or 'Not verified'",
        "end_of_life": "specific statement or 'Not verified'",
        "common_trade_off": "specific trade-off"
      },
      "products": [
        {
          "id": "p3",
          "why_suggested": "one short sentence explaining why this product fits",
          "trade_off": "one short sentence specific to this listing",
          "claims": [
            {
              "claim": "short specific claim",
              "evidence_snippet": "exact short phrase copied from the title/result you saw"
            }
          ]
        }
      ]
    }
  ]
}

Include every genuinely matching, genuinely distinct product found for each
alternative — do not arbitrarily cap how many you include. Do not force
products or alternatives that are not supported by search results.
"""

GATHER_PROMPT = f"""You are a sustainability research agent for GreenSwap, helping shoppers in India.

Given a natural-language user request, understand what the user is trying to
buy and discover environmentally preferable alternatives and real products.

The user may give only a product ("chair"), a qualified product
("microwave safe mug"), an environmentally preferred material
("recycled plastic chair"), a potentially high-impact material
("virgin plastic chair"), a longer natural-language description, or a pasted
product URL.

If the user gives a URL:
- Treat the URL as the product they are considering buying, not as the product
  category itself.
- Use web search first to identify the product and retrieve relevant details.
- Do not infer material, ingredients, certifications, price, or sustainability
  properties from the URL slug alone.
- Once the product is identified, reason about which characteristics matter for
  that specific product and discover genuinely suitable alternatives.
- The final answer should explain the product considered, relevant evidence,
  alternatives, and important trade-offs. This is a before-you-buy workflow.

You have one tool: search(query, engine, purpose).

- engine="shopping": real product listings with price and seller. Every
  listing has an id like "p3".
- engine="web": general web results. Use it to discover current alternatives,
  understand the category, or investigate a specific claim.
- purpose="discovery" for finding alternatives/categories/products.
  purpose="evidence" for verifying a specific claim about a specific product.

SEARCH BUDGET — READ CAREFULLY:
You have a maximum of {MAX_SEARCH_CALLS} searches in total, but they are NOT
interchangeable. At most {MAX_EVIDENCE_CALLS} of them may be purpose="evidence"
— the rest MUST go toward discovery (finding alternatives and real products).
This is enforced in code: an evidence search beyond the cap will be rejected.
Breadth of real products comes first. Spend discovery searches generously —
one call per genuinely distinct alternative — before considering any evidence
search at all. Evidence search is a bonus for your single strongest claim, not
a routine step for every product.

IMPORTANT: You are an agent, not a fixed keyword searcher.

Before choosing alternatives, remember: there is no universal list of materials or keywords that GreenSwap must search for every product. The agent must determine applicability from the user's actual need and context.

Before choosing alternatives:

1. Identify the core product_type and preserve the user's functional
   qualifiers.

2. Determine whether the user has already specified a material, format, or
   environmental preference.

3. Determine whether that request is eco_leaning, high_impact, or none based
   on the product context and the meaning of the request.

4. Reason about which environmental dimensions could actually matter for THIS
   product.

   Possible dimensions include material origin/resource use,
   recycled/recovered/reclaimed material, renewable material, reuse,
   refill/concentration, packaging/waste, end-of-life, recyclability,
   biodegradability/compostability, formulation/ingredients,
   organic/plant-derived alternatives, longevity, repairability,
   replacement frequency, or another dimension you identify.

   These are NOT a checklist.

   Decide which dimensions are actually meaningful for THIS request.

5. Do not force material substitution when another environmental dimension
   is more relevant. For example, some products may be better explored
   through formulation, packaging, concentration, refillability, longevity,
   repairability, or end-of-life.

6. Use those dimensions to discover genuinely different alternative
   types/materials/approaches.

7. Decide which alternatives are worth searching given the limited search
   budget. Prioritize covering MORE distinct alternatives over deeply
   investigating one — breadth of real, genuinely different options is the
   primary goal.

8. Generate natural search queries from your reasoning.

   Do NOT automatically append universal keywords such as:
   "organic", "non-toxic", "biodegradable", "plant-based", "eco-friendly",
   "recycled", or "refillable".

   These are possible search directions, not mandatory keywords.

   Choose them only when they are relevant to the actual product and request.

9. Search and inspect the results. Include every genuinely matching,
   genuinely distinct listing from each shopping search under its
   alternative — do not narrow a full page of good results down to just
   one or two.

10. EVIDENCE INVESTIGATION — optional, tightly rationed.

    After discovery, you may have at most {MAX_EVIDENCE_CALLS} purpose="evidence"
    search(es) remaining — this is enforced in code regardless of what you
    attempt. Use it (if at all) ONLY on your single highest-value product and
    claim overall, not spread across many products.

    A good evidence query identifies the actual product/brand and the
    specific claim:
      "Brand Product Name" manufacturer material
      "Brand Product Name" ingredients
      "Brand Product Name" certification
      "Brand Product Name" recycled content
      "Brand Product Name" technical specifications

    Do not treat generic seller words such as "durable", "premium", "strong",
    or "long-lasting" as independent proof of useful life.

    A search result repeating a seller's claim is still only a claim unless
    the source provides meaningful supporting information.

    Every other product is still shown with whatever its own listing text
    supports (labeled "Stated by seller" by the validator) — this is a
    complete, honest answer on its own; it does not require a dedicated
    search to be valid.

11. After observing each result, decide what to do next.

    If an alternative has strong results, move on.

    If an important environmental direction is weak or missing, try a
    different search angle.

    If the search reveals a better alternative approach that you had not
    considered, you may investigate it.

    Do not repeat searches merely to use the available search budget.

12. Carry important functional qualifiers from the user's request into
    shopping queries whenever relevant.

13. Never put budget/price words into search queries. Budget and preferences
    are handled later by ranking/filtering code.

14. Before finalizing, check that:
    - products match the requested product_type
    - important functional requirements are respected
    - alternatives are genuinely different
    - an explicit user material/environmental preference is handled correctly
    - environmental claims are supported by actual evidence
    - unsupported claims are omitted
    - one "green-sounding" feature is not treated as proof that the entire
      product is environmentally preferable
    - every genuinely matching product from your searches is included, not
      just one or two per alternative

The reasoning loop should be:

REASON -> SEARCH -> OBSERVE -> REASON -> SEARCH -> OBSERVE -> FINALIZE

not:

PRODUCT -> FIXED KEYWORDS -> SEARCH.

The category-specific environmental reasoning must be decided fresh for every
user input. Do not rely on product-specific mappings written by the developer.

{OUTPUT_RULES}

When you have gathered enough information, and ONLY when you are not calling
any more tools, reply with the final JSON described above.
"""

FINALIZE_PROMPT = f"""You are finishing a GreenSwap environmental-alternative
research task.

All searching is already done. You CANNOT search again.

Use ONLY the product ids and web findings provided to you.

Assemble the best-supported final answer from the research that was actually
gathered. Include every genuinely matching, genuinely distinct product from
the listings you were given — do not arbitrarily narrow them down.

Do not invent missing evidence.

Shopping signals supplied by backend data (Popular, Offer, Lowest price found)
are buying signals only. Never use them as evidence that a product is more
sustainable or environmentally preferable.

Do not turn a seller/manufacturer claim into independent verification.

Do not infer that one environmental attribute automatically implies another.

Do not describe a product or material as "sustainable", "eco-friendly", or
"green" as an established fact.

Impact descriptions should explain specific characteristics or potential
environmental effects rather than giving an unsupported overall verdict.

If an environmental claim is not supported by the gathered evidence, omit it
or make clear that it was not verified.

Preserve the user's product_type, inferred functional requirements, and any
explicit material/environmental preference. Do not silently drop important
functional requirements when selecting products.

If a product cannot reasonably satisfy an important functional requirement from
the available evidence, do not present it as a confident match.

If the user supplied a product URL, treat the identified product as the product
being considered for purchase. Explain relevant evidence and trade-offs for that
product, then provide alternatives that still satisfy the same underlying need.
Do not claim that the URL itself proves any product attribute.

For a high-impact request, the requested high-impact material is intentionally
excluded from the environmentally preferable product listings. Its absence
is NOT a failure.

{OUTPUT_RULES}
"""


CRITIQUE_PROMPT = """You are doing a quality check on a DRAFT GreenSwap answer
before it is shown to a user.

You are reviewing someone else's work, not writing it.

IMPORTANT CONTEXT:

The agent is NOT required to use a fixed sustainability taxonomy.

Different product categories can have different meaningful environmental
dimensions.

A material-based product may be best explored through material origin,
recycled/recovered content, renewable resources, reuse, longevity, etc.

A formulation/consumable product may instead be better explored through
ingredients, formulation, concentration, refill format, packaging, waste,
end-of-life, etc.

Do NOT penalize the draft simply because it did not include a particular
keyword such as organic, non-toxic, biodegradable, recycled, plant-based,
refillable, or reusable.

Do NOT penalize the draft for having few or no purpose="evidence" searches —
evidence search is deliberately rationed to at most one product, and every
other product being shown with only "Stated by seller" claims is the
INTENDED, complete behavior, not a gap.

Do NOT flag "too few products" as an issue unless a material/alternative that
IS shown has only one product while the underlying search very likely
returned more that were wrongly excluded — judge by what's plausible, not by
an arbitrary target count.

The question is whether the selected alternatives make sense for THIS
specific product and request.

You cannot search again yourself, but you may request exactly ONE follow-up
search if it would clearly fix the single biggest genuine problem.

Checklist:

1. PRODUCT MATCH
Does every product actually match the user's core product_type?
Do not accept a different product category merely because it is related.

2. FUNCTIONAL REQUIREMENTS
First identify the important functional, safety, compatibility, exposure/contact,
and intended-use requirements that are actually stated or genuinely necessary
from the user's request. Do NOT invent a new requirement merely because a
product has a known trade-off.

Then inspect EVERY recommended product against those requirements.

For each important requirement, determine whether the available information
indicates that it is supported, contradicted, or unclear.

A trade-off by itself is NOT a functional mismatch. For example, "glass may
break if dropped" is a trade-off, not a reason to reject glass unless the user
actually requires high impact resistance/durability or the intended use makes
that requirement materially necessary.

If the user did not state or materially imply a requirement, do not create one
just to eliminate an alternative.

A product with strong environmental/material evidence should still fail the
suitability check if it does not reasonably satisfy an important functional
requirement. Do not assume suitability merely because the product belongs to
the correct general category.

If a product appears unsuitable for an important requirement, identify that
specific product by its product id and explain the mismatch. Do not treat missing
information as proof of suitability; when uncertainty is material, require
clarification or exclude the product rather than making a confident assumption.

3. FUNCTIONAL + ENVIRONMENTAL REASONING
Do the selected alternative types both make functional sense for THIS product
and represent plausible environmental or otherwise relevant improvements?

Do not require a fixed material list. Do not require material diversity when
material is not the meaningful environmental dimension. Do not reward the
draft merely for containing words such as recycled, organic, non-toxic,
plant-based, refillable, biodegradable, reusable, or eco-friendly.

4. NO GREEN-WORD ASSUMPTIONS
Check that the answer does not treat:
- organic
- plant-based
- non-toxic
- biodegradable
- refillable
- reusable
- durable
- recycled
- "eco-friendly"

as automatic proof that the whole product is environmentally preferable.

5. USER-SPECIFIED MATERIAL/PREFERENCE
If the user explicitly requested an environmentally preferable material,
format, or characteristic, it should be treated as a primary direction.

If the user explicitly requested a high-impact material/format, its absence
from the recommended environmentally preferable products is intentional and
MUST NOT be treated as a failure.

6. EVIDENCE
Spot-check claims.

Does each evidence_snippet actually support the specific claim?

A seller/manufacturer claim should not be presented as independent
verification.

Do not infer one environmental claim from another.

7. ALTERNATIVE DIVERSITY
Are the alternatives genuinely different rather than several variations of
the same approach?

Do not require a specific number of alternatives if the evidence does not
support them.

8. SHOPPING SIGNALS
Popularity, offers, ratings, review counts, and price are shopping signals.
Do not treat them as environmental evidence and do not require a popular or
discounted product to be selected.

9. SEARCH QUALITY
Is there an obvious important environmental direction that the agent should
have investigated based on the user's request?

Only identify this as an issue if one additional targeted search could
realistically fix it.

Reply with ONLY this JSON:

{
  "pass": true or false,
  "issues": [
    {
      "type": "functional_mismatch | evidence_gap | search_quality | other",
      "product_id": "p3 or null",
      "requirement": "the specific requirement involved or null",
      "reason": "short factual explanation",
      "action": "remove | clarify | search | none"
    }
  ],
  "extra_search": {"query": "...", "engine": "shopping" or "web"} or null
}

Only set "extra_search" if you are confident ONE more search would
meaningfully fix the single biggest genuine issue.

A noted limitation with "extra_search": null is better than a wasted search.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search live web or shopping data via SerpApi, localized to India. "
                "Use shopping to find purchasable product listings. Use web to investigate "
                "a specific product claim, manufacturer information, certification, ingredients, "
                "technical documentation, or category context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query to run"},
                    "engine": {
                        "type": "string",
                        "enum": ["shopping", "web"],
                        "description": (
                            "'shopping' for product listings. "
                            "'web' for evidence research or general web information."
                        ),
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["discovery", "evidence"],
                        "description": (
                            "Why this search is being made. Use 'evidence' when "
                            "investigating a specific product/claim; otherwise use 'discovery'. "
                            "Evidence searches are capped separately and more tightly than "
                            "discovery searches."
                        ),
                    },
                },
                "required": ["query", "engine"],
            },
        },
    }
]


def _gemini_tools():
    """Convert the existing OpenAI-style search schema into Gemini function declarations."""
    fn = TOOLS[0]["function"]
    params = fn["parameters"]

    properties = {}
    for name, spec in params.get("properties", {}).items():
        item = {
            "type": getattr(types.Type, spec.get("type", "string").upper(), types.Type.STRING),
            "description": spec.get("description", ""),
        }
        if "enum" in spec:
            item["enum"] = spec["enum"]
        properties[name] = item

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=properties,
                        required=params.get("required", []),
                    ),
                )
            ]
        )
    ]


def _gemini_contents(messages):
    """Convert the agent's existing message history into Gemini contents."""
    contents = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            # System instructions are handled separately by Gemini.
            continue

        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message.get("content", ""))]
                )
            )

        elif role == "assistant":
            # IMPORTANT for Gemini 3.x:
            # If the model returned a function_call, its Part may contain a
            # thought_signature. Reconstructing the function call from only
            # name/arguments loses that signature and Gemini rejects the next
            # turn with INVALID_ARGUMENT. Reuse the original Parts verbatim.
            preserved_parts = message.get("gemini_parts")
            if preserved_parts:
                contents.append(types.Content(role="model", parts=preserved_parts))
                continue

            # Fallback for old/internal history that has no preserved Gemini parts.
            parts = []
            content = message.get("content") or ""
            if content:
                parts.append(types.Part.from_text(text=content))

            for tc in message.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                parts.append(
                    types.Part.from_function_call(
                        name=fn.get("name", "search"),
                        args=args,
                    )
                )

            if parts:
                contents.append(types.Content(role="model", parts=parts))

        elif role == "tool":
            # Gemini represents tool results as function responses.
            try:
                result = json.loads(message.get("content", "{}"))
            except Exception:
                result = {"result": message.get("content", "")}

            # Gemini FunctionResponse.response must be a JSON object/dict.
            # Our search tool can legitimately return a top-level list of
            # products, so wrap non-dict results instead of passing the list
            # directly to the Gemini SDK.
            if not isinstance(result, dict):
                result = {"result": result}

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name="search",
                            response=result,
                        )
                    ],
                )
            )

    return contents


def _gemini_generate(messages, use_tools=False, temperature=0.3):
    """Gemini replacement for client.chat.completions.create()."""
    system_instruction = ""
    for message in messages:
        if message.get("role") == "system":
            system_instruction += (message.get("content") or "") + "\n"

    config_kwargs = {
        "temperature": temperature,
        "system_instruction": system_instruction.strip() or None,
    }

    if use_tools:
        config_kwargs["tools"] = _gemini_tools()

    response = client.models.generate_content(
        model=MODEL,
        contents=_gemini_contents(messages),
        config=types.GenerateContentConfig(**config_kwargs),
    )

    return response


def _gemini_message(response):
    """Normalize a Gemini response to the small interface used by run_agent."""
    text_content = getattr(response, "text", None) or ""
    tool_calls = []

    candidate = None
    try:
        candidate = response.candidates[0]
    except Exception:
        pass

    if candidate is not None:
        for part in getattr(candidate.content, "parts", []) or []:
            fc = getattr(part, "function_call", None)
            if fc:
                args = dict(fc.args or {})
                tool_calls.append(
                    {
                        "id": f"gemini_call_{len(tool_calls) + 1}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )

    class Message:
        pass

    msg = Message()
    msg.content = text_content
    msg.tool_calls = tool_calls

    # Keep the ORIGINAL Gemini parts, including thought_signature. These must
    # be sent back unchanged on the next tool-calling turn for Gemini 3.x.
    try:
        msg.gemini_parts = list(candidate.content.parts) if candidate is not None else []
    except Exception:
        msg.gemini_parts = []

    return msg



def _tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_msg(msg) -> dict:
    """Convert the normalized Gemini message into our internal history format."""
    return {
        "role": "assistant",
        "content": getattr(msg, "content", "") or "",
        "tool_calls": list(getattr(msg, "tool_calls", None) or []),
        # Preserve Gemini's original response Parts so thought_signature is
        # not lost when the conversation is sent back for the next turn.
        "gemini_parts": list(getattr(msg, "gemini_parts", None) or []),
    }


def _tool_call_parts(tc, default_id="call_0"):
    """Return (tool_call_id, function_name, arguments) for Gemini/OpenAI call formats."""
    if isinstance(tc, dict):
        tool_id = tc.get("id") or default_id
        function_data = tc.get("function")
        if isinstance(function_data, dict):
            name = function_data.get("name", "")
            arguments = function_data.get("arguments", "{}")
        else:
            name = tc.get("name", "")
            arguments = tc.get("arguments", "{}")
    else:
        tool_id = getattr(tc, "id", default_id)
        function_data = getattr(tc, "function", None)
        if function_data is not None:
            name = getattr(function_data, "name", "")
            arguments = getattr(function_data, "arguments", "{}")
        else:
            name = getattr(tc, "name", "")
            arguments = getattr(tc, "arguments", "{}")

    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif not isinstance(arguments, str):
        arguments = str(arguments or "{}")

    return tool_id, name, arguments


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
            product_link, link_type, link_note = _best_product_link(base)
            products.append(
                {
                    "id": pid,
                    "name": base["title"],
                    "price": base["price"],
                    "source": base["source"],
                    "link": product_link,
                    "link_type": link_type,
                    "link_note": link_note,
                    "image": base["image"],
                    "rating": base["rating"],
                    "reviews": base["reviews"],
                    "product_id": base.get("product_id"),
                    "product_link": base.get("product_link"),
                    "immersive_product_page_token": base.get("immersive_product_page_token"),
                    "serpapi_immersive_product_api": base.get("serpapi_immersive_product_api"),
                    "material_type": m.get("type"),
                    "why_suggested": p.get("why_suggested"),
                    "trade_off": p.get("trade_off"),
                    "claims": p.get("claims", []) or [],
                    "shopping_signals": base.get("shopping_signals", {}),
                }
            )
        if products:
            materials.append(
                {
                    "type": m.get("type"),
                    "matches_request": bool(m.get("matches_request", False)),
                    "about": m.get("about"),
                    "impact": m.get("impact", {}) or {},
                    "info_source": "general knowledge",
                    "products": products,
                }
            )
    return materials, dropped


def _debug(response) -> None:
    if not DEBUG:
        return
    msg = _gemini_message(response)
    print("\n--- GEMINI RESPONSE ---")
    print("content:", repr(msg.content))
    print("tool_calls:", msg.tool_calls)
    print("---------------------\n")


def _finalize_clean(
    product_query: str, prefs_text: str, budget_text: str,
    catalog: dict, web_findings: list[dict],
) -> dict | None:
    products_seen = [
        {
            "id": pid,
            "title": v["title"],
            "price": v["price"],
            "source": v["source"],
            "rating": v["rating"],
            "reviews": v["reviews"],
            "delivery": v.get("delivery"),
            "snippet": v.get("snippet"),
            "extracted_price": v.get("extracted_price"),
            "old_price": v.get("old_price"),
            "extracted_old_price": v.get("extracted_old_price"),
            "shopping_signals": v.get("shopping_signals", {}),
            "has_direct_product_link": bool(
                v.get("product_link") and not _looks_like_search_url(v.get("product_link"))
            ),
        }
        for pid, v in catalog.items()
    ]

    url_context = _url_input_context(product_query)
    user_content = (
        f"User input: {product_query}\n"
        f"Input context: {json.dumps(url_context, ensure_ascii=False)}\n"
        f"Preferences (for ranking later, NOT search keywords): {prefs_text}\n"
        f"Budget (for filtering later, NOT search keywords): {budget_text}\n\n"
        "First infer and preserve the important functional requirements from the user's request. "
        "Do not recommend a product that appears unsuitable for an important requirement.\n\n"
        f"Product listings found (JSON):\n{json.dumps(products_seen, ensure_ascii=False)}\n\n"
        f"Web research notes (JSON):\n{json.dumps(web_findings, ensure_ascii=False)}\n\n"
        "Produce the final JSON now."
    )

    response = _gemini_generate(
        [
            {"role": "system", "content": FINALIZE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        use_tools=False,
        temperature=0.3,
    )
    _debug(response)
    return _parse_json(_gemini_message(response).content)


def _lite_draft(
    product_type: str | None, functional_requirements: list[str] | None,
    materials: list[dict], request_sustainability: str | None,
    user_specified_material: str | None,
) -> dict:
    return {
        "product_type": product_type,
        "functional_requirements": functional_requirements or [],
        "request_sustainability": request_sustainability,
        "user_specified_material": user_specified_material,
        "materials": [
            {
                "type": m["type"],
                "matches_request": m.get("matches_request"),
                "products": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "shopping_signals": p.get("shopping_signals", {}),
                        "claims": [
                            {"claim": c["claim"], "evidence_snippet": c["evidence_snippet"], "validated": c["validated"]}
                            for c in p.get("claims", [])
                        ],
                    }
                    for p in m["products"]
                ],
            }
            for m in materials
        ],
    }


def _run_critique(
    product_query: str, product_type: str | None,
    functional_requirements: list[str] | None, materials: list[dict],
    request_sustainability: str | None, user_specified_material: str | None,
) -> dict:
    draft = _lite_draft(
        product_type, functional_requirements, materials,
        request_sustainability, user_specified_material
    )
    response = _gemini_generate(
        [
            {"role": "system", "content": CRITIQUE_PROMPT},
            {
                "role": "user",
                "content": f"User asked for: {product_query}\n\nDraft (JSON):\n{json.dumps(draft, ensure_ascii=False)}\n\nReview this now.",
            },
        ],
        use_tools=False,
        temperature=0.2,
    )
    _debug(response)
    parsed = _parse_json(_gemini_message(response).content)
    if parsed is None:
        return {"pass": True, "issues": [], "extra_search": None}
    return {
        "pass": bool(parsed.get("pass", True)),
        "issues": parsed.get("issues") or [],
        "extra_search": parsed.get("extra_search"),
    }


def _apply_functional_critique(materials: list[dict], critique: dict) -> tuple[list[dict], int]:
    remove_ids = set()
    for issue in critique.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        if issue.get("type") != "functional_mismatch":
            continue
        if issue.get("action") != "remove":
            continue
        pid = issue.get("product_id")
        if pid:
            remove_ids.add(pid)

    if not remove_ids:
        return materials, 0

    removed = 0
    filtered_materials = []
    for material in materials:
        kept = []
        for product in material.get("products", []):
            if product.get("id") in remove_ids:
                removed += 1
            else:
                kept.append(product)
        if kept:
            material = dict(material)
            material["products"] = kept
            filtered_materials.append(material)
    return filtered_materials, removed


def _price_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _build_filter_options(materials: list[dict]) -> dict:
    material_options = []
    prices = []
    ratings = []
    reviews = []

    for material in materials:
        material_type = material.get("type")
        if material_type and material_type not in material_options:
            material_options.append(material_type)

        for product in material.get("products", []):
            price = _price_number(product.get("price"))
            if price is not None:
                prices.append(price)
            try:
                rating = product.get("rating")
                if rating not in (None, "", "N/A"):
                    ratings.append(float(rating))
            except (TypeError, ValueError):
                pass
            reviews_value = product.get("reviews")
            if reviews_value not in (None, "", "N/A"):
                match = re.search(r"[\d,]+", str(reviews_value))
                if match:
                    try:
                        reviews.append(int(match.group(0).replace(",", "")))
                    except ValueError:
                        pass

    return {
        "material": material_options,
        "price": {"currency": "INR", "min": min(prices) if prices else None, "max": max(prices) if prices else None},
        "rating": {"min": min(ratings) if ratings else None, "max": max(ratings) if ratings else None},
        "reviews": {"min": min(reviews) if reviews else None, "max": max(reviews) if reviews else None},
        "sort": ["relevance", "price_low_to_high", "price_high_to_low", "rating", "reviews"],
    }


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

    input_context = _url_input_context(product_query)

    messages = [
        {"role": "system", "content": GATHER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Product / URL: {product_query}\n"
                f"Input context: {json.dumps(input_context, ensure_ascii=False)}\n"
                f"Preferences (for ranking later, NOT search keywords): {prefs_text}\n"
                f"Budget (for filtering later, NOT search keywords): {budget_text}"
            ),
        },
    ]

    trace: list[dict] = []
    catalog: dict[str, dict] = {}
    web_findings: list[dict] = []
    search_calls = 0
    evidence_calls = 0
    next_id = 1
    parsed = None

    if input_context["is_url"] and search_calls < MAX_SEARCH_CALLS:
        identification_query = _url_search_query(product_query)
        try:
            results, cached = serp_search(identification_query, "web")
            search_calls += 1
            web_findings.append({
                "query": identification_query,
                "purpose": "url_product_identification",
                "results": results,
            })
            trace.append({
                "step": search_calls, "query": identification_query, "engine": "web",
                "purpose": "url_product_identification", "results_found": len(results), "cached": cached,
            })
        except Exception as exc:
            search_calls += 1
            web_findings.append({
                "query": identification_query, "purpose": "url_product_identification",
                "results": [], "error": str(exc),
            })
            trace.append({
                "step": search_calls, "query": identification_query, "engine": "web",
                "purpose": "url_product_identification", "results_found": 0,
                "cached": False, "error": str(exc),
            })

    # ---------------- PHASE 1: GATHER ----------------
    for _ in range(MAX_LLM_TURNS):
        if search_calls >= MAX_SEARCH_CALLS:
            break

        response = _gemini_generate(messages, use_tools=True, temperature=0.3)
        msg = _gemini_message(response)
        _debug(response)
        messages.append(_assistant_msg(msg))

        if msg.tool_calls:
            for index, tc in enumerate(msg.tool_calls):
                tool_id, function_name, raw_arguments = _tool_call_parts(
                    tc, f"gemini_call_{index + 1}"
                )

                if search_calls >= MAX_SEARCH_CALLS:
                    messages.append(_tool_msg(tool_id, "Search cap reached."))
                    continue

                try:
                    args = json.loads(raw_arguments or "{}")
                    query = str(args["query"]).strip()
                    engine = args.get("engine", "shopping")
                    purpose = args.get("purpose", "discovery")
                    if engine not in ("shopping", "web"):
                        engine = "shopping"
                    if purpose not in ("discovery", "evidence"):
                        purpose = "discovery"
                except (json.JSONDecodeError, KeyError, TypeError):
                    messages.append(_tool_msg(
                        tool_id,
                        "Invalid arguments. Call search with a query and an engine.",
                    ))
                    continue

                if not query:
                    messages.append(_tool_msg(tool_id, "Search query cannot be empty."))
                    continue

                original_query = query
                if engine == "shopping" and purpose == "discovery":
                    query = _augment_shopping_query(query, product_query)

                if purpose == "evidence" and evidence_calls >= MAX_EVIDENCE_CALLS:
                    messages.append(_tool_msg(
                        tool_id,
                        f"Evidence search cap reached ({MAX_EVIDENCE_CALLS} max). "
                        "Use a discovery search instead, or finalize using listing-based claims.",
                    ))
                    continue

                try:
                    results, cached = serp_search(query, engine)
                except Exception as e:
                    search_calls += 1
                    if purpose == "evidence":
                        evidence_calls += 1
                    trace.append({
                        "step": search_calls, "query": query, "engine": engine,
                        "purpose": purpose, "results_found": 0, "cached": False,
                        "error": str(e),
                    })
                    messages.append(_tool_msg(tool_id, f"Search failed: {e}"))
                    continue

                search_calls += 1
                if purpose == "evidence":
                    evidence_calls += 1

                trace.append({
                    "step": search_calls, "query": query, "engine": engine,
                    "purpose": purpose, "results_found": len(results), "cached": cached,
                    **({"model_query": original_query} if original_query != query else {}),
                })

                if engine == "shopping":
                    model_view = []
                    for item in results:
                        pid = f"p{next_id}"
                        next_id += 1
                        catalog[pid] = item
                        model_view.append({
                            "id": pid, "title": item["title"], "price": item["price"],
                            "source": item["source"], "rating": item["rating"],
                            "reviews": item["reviews"],
                        })
                    payload = model_view
                else:
                    web_findings.append({"query": query, "purpose": purpose, "results": results})
                    payload = results

                messages.append(_tool_msg(
                    tool_id, json.dumps(payload, ensure_ascii=False)
                ))
            continue

        parsed = _parse_json(msg.content)
        if parsed is not None:
            break
        messages.append({"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object described in the instructions."})

    # Compute shopping signals from the complete discovered catalog before
    # finalization. The model can use these as factual buying metadata.
    _add_shopping_signals(catalog)

    # ---------------- PHASE 2: FINALIZE ----------------
    if parsed is None:
        parsed = _finalize_clean(product_query, prefs_text, budget_text, catalog, web_findings)

    if parsed is None:
        return {
            "summary": "The agent did not produce a valid answer. Please try again.",
            "input_mode": "url" if input_context["is_url"] else "text",
            "input_source": input_context.get("source"),
            "input_url": input_context.get("url"),
            "input_asin": input_context.get("asin"),
            "materials": [],
            "filter_options": _build_filter_options([]),
            "availability_note": AVAILABILITY_NOTE,
            "no_suitable_alternative": True, "trace": trace,
            "meta": {"searches": search_calls, "evidence_searches": evidence_calls,
                     "dropped_unknown_ids": 0, "error": "no_valid_json"},
        }

    materials, dropped = _resolve(parsed, catalog)
    materials, dropped_type_mismatch = filter_by_product_type(materials, parsed.get("product_type"))

    if parsed.get("request_sustainability") == "high_impact":
        materials = [m for m in materials if not m.get("matches_request")]

    # ---------------- PHASE 3: VALIDATE ----------------
    materials = validate_claims(materials, web_findings)

    # ---------------- PHASE 4: SELF-CRITIQUE ----------------
    functional_requirements = parsed.get("functional_requirements") or []

    critique = _run_critique(
        product_query, parsed.get("product_type"), functional_requirements, materials,
        parsed.get("request_sustainability"), parsed.get("user_specified_material"),
    )

    materials, dropped_functional_mismatch = _apply_functional_critique(materials, critique)
    critique_meta = {
        "ran": True, "passed": critique["pass"], "issues": critique["issues"],
        "extra_search_used": False,
    }

    extra = critique.get("extra_search")
    if not critique["pass"] and extra and search_calls < MAX_SEARCH_CALLS:
        q, eng = extra.get("query"), extra.get("engine", "shopping")
        purpose = "evidence" if eng == "web" else "discovery"
        if eng not in ("shopping", "web"):
            eng = "shopping"
            purpose = "discovery"
        # Respect the evidence sub-cap here too, even for the critique's own
        # follow-up — no exception swallows the budget split.
        if purpose == "evidence" and evidence_calls >= MAX_EVIDENCE_CALLS:
            purpose = "discovery"
        if q:
            try:
                results, cached = serp_search(q, eng)
                search_calls += 1
                if purpose == "evidence":
                    evidence_calls += 1
                trace.append({"step": search_calls, "query": q, "engine": eng,
                              "purpose": purpose, "results_found": len(results),
                              "cached": cached, "from_critique": True})

                if eng == "shopping":
                    for item in results:
                        pid = f"p{next_id}"
                        next_id += 1
                        catalog[pid] = item
                    _add_shopping_signals(catalog)
                else:
                    web_findings.append({"query": q, "purpose": purpose, "results": results})

                parsed2 = _finalize_clean(product_query, prefs_text, budget_text, catalog, web_findings)
                if parsed2 is not None:
                    parsed = parsed2
                    materials, dropped = _resolve(parsed, catalog)
                    materials, dropped_type_mismatch = filter_by_product_type(materials, parsed.get("product_type"))
                    if parsed.get("request_sustainability") == "high_impact":
                        materials = [m for m in materials if not m.get("matches_request")]
                    materials = validate_claims(materials, web_findings)
                    materials, extra_removed = _apply_functional_critique(materials, critique)
                    dropped_functional_mismatch += extra_removed
                    critique_meta["extra_search_used"] = True
            except Exception as e:
                trace.append({"step": search_calls + 1, "query": q, "engine": eng,
                              "results_found": 0, "cached": False, "error": str(e), "from_critique": True})

    return {
        "summary": parsed.get("summary", ""),
        "input_mode": "url" if input_context["is_url"] else "text",
        "input_source": input_context.get("source"),
        "input_url": input_context.get("url"),
        "input_asin": input_context.get("asin"),
        "product_type": parsed.get("product_type"),
        "functional_requirements": functional_requirements,
        "user_specified_material": parsed.get("user_specified_material"),
        "request_sustainability": parsed.get("request_sustainability", "none"),
        "caution_note": parsed.get("caution_note"),
        "materials": materials,
        "filter_options": _build_filter_options(materials),
        "availability_note": AVAILABILITY_NOTE,
        "no_suitable_alternative": len(materials) == 0,
        "trace": trace,
        "meta": {
            "searches": search_calls,
            "evidence_searches": evidence_calls,
            "dropped_unknown_ids": dropped,
            "dropped_type_mismatch": dropped_type_mismatch,
            "dropped_functional_mismatch": dropped_functional_mismatch,
            "shopping_signals": {
                "popular_definition": "high review volume among listings found in this run",
                "offer_definition": "current price lower than a retrieved previous/old price",
                "not_sales_data": True,
            },
            "critique": critique_meta,
        },
    }