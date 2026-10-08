"""
GreenSwap: core agent loop (v7).

v7 adds:
  - provider layer (llm.py): Gemini runs the multi-turn gather loop, Groq
    runs the self-critique and is the automatic fallback;
  - compact prompts (the old rules were repeated 3-4 times and pushed a
    single request past Groq's 8k-token limit);
  - `avoid_terms` on the search tool, so the context safety net never
    re-adds a high-impact material the model deliberately replaced;
  - a single link policy, stricter evidence validation, fail-closed
    critique that is re-run after its own follow-up search;
  - an `on_event` callback so the API can stream the agent's progress.

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
    the prompt or history*, even when "tools" is left out of the
    request — and Groq then rejects that as invalid. The fix isn't
    "suppress it harder", it's "give it a conversation with nothing to
    be primed by". FINALIZE_PROMPT therefore never mentions searching.

  PHASE 3 — VALIDATE / ENFORCE (see evidence.py, product_match.py)
    Claims are checked against text actually retrieved. Product type
    is checked against the listing's own title. A high-impact material
    is stripped from output even if the model didn't comply. None of
    this is trusted to the model alone.

  PHASE 4 — SELF-CRITIQUE
    One extra LLM call (zero SerpApi cost) reviews the finished draft
    against a checklist. It is explicitly told about our own business
    rules (e.g. a high-impact material being deliberately absent is
    correct, not a gap) so it doesn't flag intended behavior as a bug.
    If it finds a genuine, fixable issue and search budget remains, it
    can trigger exactly ONE more search and one rebuild — never a loop.
    The rebuilt draft is critiqued once more (no search) before output.

Caps are enforced in code, never trusted to the model:
    MAX_SEARCH_CALLS    real SerpApi calls per run
    MAX_EVIDENCE_CALLS  of those, how many may be purpose="evidence"
    MAX_LLM_TURNS       ceiling on phase-1 turns

Ranking/filtering is NOT done by the search agent. Retrieved results can
be filtered/sorted after the run without another SerpApi call; see
ranking.py (pure code). Preferences steer ranking/filtering, not search text.
"""

import os
import json
import re
from typing import Callable
from urllib.parse import urlparse, parse_qs, unquote

import llm
from serp_tool import serp_search
from evidence import validate_claims
from product_match import filter_by_product_type, _stem
from ranking import extract_price, parse_number, SORT_OPTIONS
from research import (
    normalize_research_state, enforce_user_requirements, guard_conclusion,
    absolute_language, answer_texts,
)

MAX_SEARCH_CALLS = 5
MAX_LLM_TURNS = 8

# Evidence searches (purpose="evidence") compete with discovery searches
# for the same budget. Left to the model's discretion, evidence was
# starving discovery ("too few products shown"). This sub-cap guarantees
# discovery always gets at least MAX_SEARCH_CALLS - MAX_EVIDENCE_CALLS
# calls; it is enforced in code below, not just requested in the prompt.
MAX_EVIDENCE_CALLS = 1

# How many listings from one shopping search the model sees, in gather and
# finalize alike. Offering all 20 at finalize pushed requests past Groq's
# 8k tokens-per-minute limit (live failure: 9,398 tokens requested).
GATHER_VIEW_LIMIT = 10

DEBUG = os.environ.get("AGENT_DEBUG", "0") == "1"

AVAILABILITY_NOTE = (
    "Availability shown is based on regional search results and may not "
    "reflect real-time shipping to your exact location. Please verify on "
    "the merchant's page before purchasing."
)

EventSink = Callable[[dict], None] | None


# =====================================================================
# Links
# =====================================================================

def _is_google_host(value: str | None) -> bool:
    try:
        return "google." in urlparse(value or "").netloc.lower()
    except Exception:
        return False


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
    """
    One link policy for the whole app:
      1. a real merchant URL (not Google, not a search page) -> merchant_product;
      2. else the stable Google Shopping search for this title + seller built
         by serp_tool -> shopping_search;
      3. else SerpApi's Google product page (can be session-dependent).
    A Google URL is never labelled as a direct merchant listing.
    """
    for key in ("merchant_link", "product_link"):
        url = item.get(key)
        if url and not _is_google_host(url) and not _looks_like_search_url(url):
            return url, "merchant_product", "Direct merchant listing"

    if item.get("link"):
        return item["link"], "shopping_search", "Opens Google Shopping for this product and seller"

    if item.get("product_link"):
        return item["product_link"], "google_product_page", "Google Shopping product page (may expire)"

    return None, "unavailable", "Direct product URL was not available"


# =====================================================================
# URL input
# =====================================================================

AMAZON_HOSTS = {
    "amazon.in", "www.amazon.in", "amazon.com", "www.amazon.com",
    "amazon.co.uk", "www.amazon.co.uk", "amazon.ca", "www.amazon.ca",
    "amazon.de", "www.amazon.de", "amazon.fr", "www.amazon.fr",
    "amazon.it", "www.amazon.it", "amazon.es", "www.amazon.es",
    "amazon.co.jp", "www.amazon.co.jp", "amazon.com.au", "www.amazon.com.au",
}

FLIPKART_HOSTS = {"flipkart.com", "www.flipkart.com"}


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


# =====================================================================
# Query context safety net
# =====================================================================

# Words that never carry use-case context. Buying/ranking words are handled
# separately and must not leak into discovery queries either.
_CONTEXT_STOPWORDS = {
    "a", "an", "the", "for", "to", "of", "and", "or", "with", "without", "in",
    "on", "at", "by", "from", "as", "so", "but", "not", "no", "is", "are", "be",
    "it", "its", "this", "that", "these", "those", "will", "would", "should",
    "can", "could", "do", "does", "doesn", "don", "isn", "won", "my", "me", "i",
    "we", "our", "you", "need", "needs", "want", "wants", "looking", "find",
    "show", "get", "buy", "suggest", "recommend", "please", "help", "some",
    "any", "something", "thing", "things", "one", "very", "really", "much",
    "many", "more", "most", "also", "just", "like", "which", "who", "what",
    "when", "where", "how", "use", "using", "used", "carry", "carrying",
    "take", "taking", "keep", "make", "long", "time", "lot", "last", "lasts",
    "frequent", "replacement", "alternative", "alternatives", "option",
    "options", "eco", "friendly", "sustainable", "green", "best", "good",
    "cheap", "cheapest", "budget", "affordable", "price", "priced", "under",
    "below", "less", "around", "cost", "costing", "rs", "inr",
}
MAX_CONTEXT_TOKENS = 3


def _tokens(value: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", value or "") if len(t) > 1]


def _augment_shopping_query(query: str, user_request: str, avoid_terms: list[str] | None = None) -> str:
    """
    Preserve meaningful user context when the model drops it from a
    shopping query ("water bottle for office" -> keep "office").

    Never re-adds a word the model listed in `avoid_terms` — e.g. the
    high-impact material it deliberately replaced ("virgin plastic chair"
    must not turn "bamboo chair" into "bamboo chair virgin plastic").
    Appends at most MAX_CONTEXT_TOKENS words. This is only a safety net;
    the LLM still chooses the alternative/approach.
    """
    if not query or not user_request or _is_url(user_request):
        return (query or "").strip()

    present = {_stem(t) for t in _tokens(query)}
    avoid = {_stem(t) for term in (avoid_terms or []) for t in _tokens(term)}

    missing = []
    for token in _tokens(user_request):
        stem = _stem(token)
        if token in _CONTEXT_STOPWORDS or token.isdigit():
            continue
        if stem in present or stem in avoid or token in missing:
            continue
        missing.append(token)

    if not missing:
        return query.strip()
    return f"{query.strip()} {' '.join(missing[:MAX_CONTEXT_TOKENS])}".strip()


# =====================================================================
# Shopping signals (buying metadata, never environmental evidence)
# =====================================================================

def _item_price(item: dict) -> float | None:
    price = extract_price(item.get("extracted_price"))
    return price if price is not None else extract_price(item.get("price"))


def _offer_signal(item: dict) -> dict | None:
    """Detect an offer only from actual current + old prices returned by shopping."""
    current = _item_price(item)
    old = extract_price(item.get("extracted_old_price"))
    if old is None:
        old = extract_price(item.get("old_price"))

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
    review_values = [parse_number(item.get("reviews")) for item in catalog.values()]
    review_values = [v for v in review_values if v is not None and v >= 0]

    threshold = None
    if review_values:
        ordered = sorted(review_values)
        threshold = ordered[max(0, int((len(ordered) - 1) * 0.75))]

    prices = [p for p in (_item_price(item) for item in catalog.values()) if p is not None]
    lowest_price = min(prices) if prices else None

    for item in catalog.values():
        reviews = parse_number(item.get("reviews"))
        badges = []

        if threshold is not None and reviews is not None and reviews >= threshold:
            badges.append("Popular")

        if item.get("is_bestseller") or item.get("bestseller") or item.get("best_seller"):
            badges.append("Bestseller")

        offer = _offer_signal(item)
        if offer:
            badges.append("Offer")

        if lowest_price is not None and _item_price(item) == lowest_price:
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


def _listing_text(item: dict) -> str:
    """Seller-provided listing text (besides the title) used as claim evidence."""
    parts = [item.get("snippet") or ""]
    parts += [str(e) for e in (item.get("extensions") or [])]
    parts.append(item.get("tag") or "")
    return " | ".join(p for p in parts if p)


# =====================================================================
# Prompts
# =====================================================================

# Shared by GATHER and FINALIZE. It must not mention searching or tools:
# FINALIZE has to stay free of anything that could prime a tool call.
OUTPUT_RULES = """
PRIORITIES, IN ORDER: (1) what the user is actually buying; (2) the user's
explicit requirements; (3) functional, safety, compatibility and intended-use
suitability; (4) only then environmental differences among SUITABLE options;
evidence throughout. Environmental reasoning never justifies recommending a
product that does not meet the need.

UNDERSTAND THE REQUEST
- product_type: what the user is actually buying, 1-3 words, without material,
  eco qualifier, use case or budget ("ceramic mug under ₹500" -> "mug").
- user_requirements: what the user EXPLICITLY asked for, in their words (e.g.
  "organic", "plant-based", "for office"). Keep them; never silently replace one
  with "something environmentally better". Generic words like "eco-friendly"
  or "sustainable" express the goal, not a product attribute to match.
- functional_requirements / safety_requirements: what the product must do or be
  safe for, inferred from this request (intended use, contact, exposure,
  compatibility...). Do not invent needs the request does not imply.
- request_sustainability: "eco_leaning" if the user named a material/format that is
  environmentally preferable here, "high_impact" if they named a comparatively
  high-impact one (set user_specified_material, write a factual caution_note,
  recommend alternatives and mark groups of that material matches_request=true;
  code removes them), else "none".
- A pasted URL is a before-you-buy request: identify the product from retrieved
  information only (never from the URL slug), then give suitable alternatives.

DECIDE WHAT MATTERS (yours to decide; there is no fixed list)
- environmental_dimensions: the characteristics that could meaningfully
  DISTINGUISH the suitable alternatives for THIS request, given what research
  actually found. Each needs why_relevant and a priority. It may be one, several,
  or none. A dimension is something to compare, never a reason to reject a
  product: if every suitable option shares a trait, it does not distinguish them.
- focus_note: one plain sentence telling the user what you focused on and why
  ("Focused on refill format because it was the main difference among suitable
  products."). No hidden reasoning, just the decision.
- USER WANT vs PRODUCT CLAIM vs EVIDENCE: what the user wants, what a product
  claims, and what retrieved text supports are different things. "Organic",
  "natural", "recycled", "plant-based", "biodegradable", "recyclable", "reusable",
  "refillable", "plastic-free" can be requirements or claims; none is automatic
  proof of overall environmental superiority. Never infer one property from
  another. Read claims precisely: an outer paper box around a plastic tube is not
  plastic-free packaging.
- conclusion: the overall verdict the evidence allows: "clear_advantage" (only
  with product-specific evidence), "potential_advantage" (seller statements or
  general reasoning), or "no_clear_winner". "No clear winner based on available
  evidence" is a successful, honest outcome. Code caps this verdict at what the
  validated evidence supports.

ALTERNATIVES AND PRODUCTS
- Alternatives must be genuinely different types/materials/approaches; no fixed
  count, no near-duplicates to fill slots.
- A product belongs under an alternative only if it matches that alternative,
  is the SAME product_type and reasonably meets the important functional
  requirements. Missing information is not proof of suitability: exclude clear
  mismatches; otherwise state the limitation in trade_off. A trade-off (e.g. glass
  can break) is not a mismatch unless the user's need makes it one.
- Include EVERY genuinely matching, distinct listing you were given for an
  alternative, not just one or two. Exclude only type/requirement mismatches and
  near-exact duplicates. A listing is one product at one seller; never claim two
  listings are the same item.
- Use only product ids you were given. Never invent products, prices, ratings,
  reviews or links, and never write URLs (code attaches links).
- Shopping signals (Popular, Offer, Lowest price found, rating, reviews, price)
  are buying signals, never environmental evidence. Green-sounding words in a
  title are claims to check, not a reason to include or rank a product.
- requirement_checks: for each product, judge every user/functional/safety
  requirement as "met" (cite an exact snippet), "unclear", or "contradicted".
  Code removes contradicted products and shows uncited "met" as Not verified.

IMPACT FIELDS (general context, not verification of a listing)
- material_note: a specific characteristic or potential effect, not a verdict.
- reusability: the real use pattern; "single-use" only if genuinely discarded
  after one use. Reusable does not imply durable.
- end_of_life: only if reasonably supported, else exactly "Not verified".
- common_trade_off: one relevant downside.
- Never state "sustainable", "eco-friendly", "green" or "the most sustainable" as
  fact; prefer "worth considering because...", "potential advantage on...",
  "seller states...", "evidence found for...", "not verified", "trade-off...".

CLAIMS
- 0-2 claims per product that explain why it was chosen; kind is
  "environmental", "requirement" or "functional".
- If retrieved text contradicts or qualifies a claim, cite it as counter_snippet
  (code then labels the claim "Conflicting / unclear").
- evidence_snippet: an EXACT phrase of at least 3 words copied (not paraphrased)
  from that product's title/listing text or from a web result you were shown, and
  it must mention what the claim is about. Code checks this: listing text is
  labelled "Stated by seller", product-specific web text "Evidence found",
  anything else "Not verified".
- A seller claim repeated on a web page is still a seller claim. If evidence is
  insufficient, omit the claim.

Reply with ONLY this JSON, no other text:
{
  "summary": "one sentence on what was found",
  "product_type": "1-3 words",
  "user_requirements": ["explicit user requirement"],
  "functional_requirements": ["short concrete requirement"],
  "safety_requirements": ["short requirement"],
  "environmental_dimensions": [{"dimension": "...", "why_relevant": "...", "priority": "high | medium | low"}],
  "focus_note": "one sentence",
  "conclusion": {"verdict": "clear_advantage | potential_advantage | no_clear_winner", "explanation": "one or two sentences"},
  "user_specified_material": "string or null",
  "request_sustainability": "eco_leaning | high_impact | none",
  "caution_note": "one factual sentence if high_impact, else null",
  "materials": [
    {
      "type": "short alternative approach name",
      "matches_request": true,
      "about": "one sentence",
      "why_different": "how this approach differs from the others",
      "impact": {"material_note": "...", "reusability": "...", "end_of_life": "...", "common_trade_off": "..."},
      "products": [
        {
          "id": "p3",
          "why_suggested": "one short sentence",
          "trade_off": "one short sentence specific to this listing",
          "requirement_checks": [{"requirement": "as listed above", "status": "met | unclear | contradicted", "evidence_snippet": "exact copied phrase or null"}],
          "claims": [{"claim": "short specific claim", "kind": "environmental", "evidence_snippet": "exact copied phrase", "counter_snippet": null}]
        }
      ]
    }
  ]
}
If nothing suitable was found, return "materials": [] and explain in summary.
"""

GATHER_PROMPT = f"""You are GreenSwap, a research agent that helps shoppers in
India find environmentally preferable alternatives as real, buyable products.

TOOL: search(query, engine, purpose, avoid_terms)
- engine="shopping": product listings, each with an id like "p3".
- engine="web": web results, to understand a category or check a specific claim.
- purpose="discovery" finds alternatives/products; purpose="evidence" checks one
  claim about one specific product.
- avoid_terms: words from the user's request you are deliberately NOT searching
  for. ALWAYS list a high-impact material you are replacing (user asked "virgin
  plastic chair", you search "bamboo chair" -> avoid_terms ["virgin plastic"]).
  The backend re-adds dropped use-case words (e.g. "office") to shopping queries
  unless you list them here.

BUDGET (enforced in code): at most {MAX_SEARCH_CALLS} searches, of which at most
{MAX_EVIDENCE_CALLS} may be purpose="evidence". Spend discovery first: about one
shopping search per genuinely distinct alternative. Use an evidence search, if at
all, only for your single highest-value product claim, with a query naming the
brand/product and the claim ("Brand Product" recycled content). Generic seller
words (durable, premium) are not proof.

YOU RUN THE RESEARCH. You decide what you need to know, which characteristics
could distinguish suitable options, which alternative approaches are worth
comparing, what to search, which claims need checking, whether more research is
needed, and when you have enough. Let results change your direction: if a search
reveals a difference you had not considered, you may pursue it.
Constraints on queries: build them from the user's need and explicit
requirements (keep use-case qualifiers), not from generic sustainability words;
never put price/budget words in a query (code applies budget later); never
search just to use budget. For a URL input, identify the product from retrieved
data first.

When you have enough, reply with the final JSON.
{OUTPUT_RULES}"""

FINALIZE_PROMPT = f"""You are finishing a GreenSwap environmental-alternative
research task. The research is complete. Use ONLY the product listings and web
notes provided in the user message; do not invent missing evidence.

If the user supplied a product URL, treat the identified product as the one being
considered, explain relevant evidence and trade-offs, then give alternatives that
meet the same need. The URL itself proves no attribute.

For a high_impact request the requested material is intentionally excluded from
the recommendations; its absence is not a failure.
{OUTPUT_RULES}"""

CRITIQUE_PROMPT = """You quality-check a DRAFT GreenSwap answer before a user sees it.
You review; you do not rewrite.

INTENDED BEHAVIOUR — never flag these:
- No fixed sustainability taxonomy: different products have different meaningful
  dimensions (materials vs formulation/refill/packaging). Do not demand keywords
  such as organic, recycled, biodegradable or refillable.
- Evidence search is rationed to at most one product; "Stated by seller" claims on
  the rest are the complete, intended behaviour. Unvalidated claims are already
  labelled honestly for the user and are not by themselves an issue.
- For a high_impact request the requested material is deliberately absent. That
  material is NOT a requirement: never flag a product for not being made of it,
  and never request a search for it.
- "Too few products" is an issue only if a shown alternative very likely had more
  matching listings that were wrongly excluded.
- Popularity, offers, ratings and price are buying signals, not evidence.

CHECK (use only requirements the user stated or the draft lists; never invent one):
1. PRODUCT: product_type is what the user is buying; every product is that type.
2. USER REQUIREMENTS: every explicit user requirement was preserved, not swapped
   for a generic "greener" goal. Words listed under
   request_terms_not_covered_by_any_requirement may be dropped requirements.
3. FUNCTION: for EACH product, each stated requirement is supported / contradicted /
   unclear. A trade-off is not a mismatch unless the stated need makes it one.
   Flag a contradicted product by its id.
4. PACKAGING/DIMENSIONS: no suitable product was rejected merely for a trait
   (e.g. plastic packaging) that the user did not rule out; the chosen dimensions
   fit THIS request; no obvious differentiator found during research was missed.
5. GREEN WORDS: no product was chosen or ranked just for a green keyword; organic,
   natural, recycled, plant-based etc. are not treated as automatic superiority.
6. CLAIM PRECISION: each evidence_snippet supports its claim exactly; outer
   packaging is not confused with the product's own container; seller claims are
   not presented as independent verification.
7. CONCLUSION: the overall verdict and wording do not claim more than the
   evidence. If they do, set conclusion_override (usually "no_clear_winner").
8. SEARCH QUALITY: an obvious important direction was missed AND one targeted
   search could realistically fix it.

Reply with ONLY this JSON:
{
  "pass": true or false,
  "issues": [
    {
      "type": "functional_mismatch | requirement_not_preserved | green_keyword | claim_precision | unsupported_conclusion | evidence_gap | search_quality | other",
      "product_id": "p3 or null",
      "requirement": "the requirement involved or null",
      "reason": "short factual explanation",
      "action": "remove | clarify | search | none"
    }
  ],
  "extra_search": {"query": "...", "engine": "shopping" or "web"} or null,
  "conclusion_override": "potential_advantage | no_clear_winner" or null
}
Set "extra_search" only if ONE more search would clearly fix the single biggest
genuine issue; a noted limitation is better than a wasted search.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search live Google Shopping or Google web results via SerpApi, localized to India. "
                "Use shopping for purchasable listings; web for category context or one specific product claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query to run"},
                    "engine": {
                        "type": "string",
                        "enum": ["shopping", "web"],
                        "description": "'shopping' for product listings, 'web' for web results.",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["discovery", "evidence"],
                        "description": (
                            "'evidence' when checking a specific product claim (tightly capped), "
                            "otherwise 'discovery'."
                        ),
                    },
                    "avoid_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Words from the user's request deliberately left out of this query "
                            "(e.g. a high-impact material being replaced). The backend will not re-add them."
                        ),
                    },
                },
                "required": ["query", "engine"],
            },
        },
    }
]


# =====================================================================
# Helpers
# =====================================================================

def _tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": "search", "content": content}


def _assistant_msg(msg: llm.Message) -> dict:
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": list(msg.tool_calls),
        # Gemini's original Parts keep thought_signature for the next turn.
        "gemini_parts": list(msg.gemini_parts),
    }


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


def _debug(msg: llm.Message) -> None:
    if DEBUG:
        print(f"\n--- LLM RESPONSE ({msg.provider}) ---")
        print("content:", repr(msg.content[:2000]))
        print("tool_calls:", msg.tool_calls)
        print("---------------------\n")


def _short(text: str | None, limit: int) -> str | None:
    if not text:
        return text
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _gather_view(pid: str, item: dict) -> dict:
    """What the gather model sees for one listing: compact, ids only."""
    view = {
        "id": pid, "title": item["title"], "price": item["price"],
        "source": item["source"], "rating": item["rating"], "reviews": item["reviews"],
    }
    if item.get("snippet"):
        view["listing_text"] = _short(item["snippet"], 140)
    return {k: v for k, v in view.items() if v not in (None, "", [])}


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
            link, link_type, link_note = _best_product_link(base)
            products.append(
                {
                    "id": pid,
                    "name": base["title"],
                    "price": base["price"],
                    "extracted_price": _item_price(base),
                    "source": base["source"],
                    "link": link,
                    "link_type": link_type,
                    "link_note": link_note,
                    "image": base["image"],
                    "rating": base["rating"],
                    "reviews": base["reviews"],
                    "listing_text": _listing_text(base),
                    "product_id": base.get("product_id"),
                    "immersive_product_page_token": base.get("immersive_product_page_token"),
                    "material_type": m.get("type"),
                    "why_suggested": p.get("why_suggested"),
                    "trade_off": p.get("trade_off"),
                    "claims": p.get("claims", []) or [],
                    # Checked and replaced by research.enforce_user_requirements.
                    "raw_requirement_checks": p.get("requirement_checks", []) or [],
                    "shopping_signals": base.get("shopping_signals", {}),
                }
            )
        if products:
            materials.append(
                {
                    "type": m.get("type"),
                    "matches_request": bool(m.get("matches_request", False)),
                    "about": m.get("about"),
                    "why_different": m.get("why_different"),
                    "impact": m.get("impact", {}) or {},
                    "info_source": "model general knowledge (not verified per listing)",
                    "products": products,
                }
            )
    return materials, dropped


def _finalize_clean(
    product_query: str, prefs_text: str, budget_text: str,
    catalog: dict, web_findings: list[dict], notify=None,
) -> dict | None:
    products_seen = []
    for pid, v in catalog.items():
        item = {
            "id": pid,
            "title": v["title"],
            "price": v["price"],
            "source": v["source"],
            "rating": v["rating"],
            "reviews": v["reviews"],
            "listing_text": _short(_listing_text(v), 140),
            "old_price": v.get("old_price"),
            "badges": (v.get("shopping_signals") or {}).get("badges"),
        }
        products_seen.append({k: x for k, x in item.items() if x not in (None, "", [])})

    web_notes = [
        {
            "query": f.get("query"),
            "results": [
                {"title": r.get("title"), "snippet": r.get("snippet")}
                for r in f.get("results", []) or []
            ],
        }
        for f in web_findings
    ]

    user_content = (
        f"User input: {product_query}\n"
        f"Input context: {json.dumps(_url_input_context(product_query), ensure_ascii=False)}\n"
        f"Preferences (applied later by code): {prefs_text}\n"
        f"Budget (applied later by code): {budget_text}\n\n"
        f"Product listings (JSON):\n{json.dumps(products_seen, ensure_ascii=False)}\n\n"
        f"Web notes (JSON):\n{json.dumps(web_notes, ensure_ascii=False)}\n\n"
        "Produce the final JSON now."
    )

    messages = [
        {"role": "system", "content": FINALIZE_PROMPT},
        {"role": "user", "content": user_content},
    ]
    msg = llm.generate(messages, tools=None, temperature=0.3, task="finalize", notify=notify)
    _debug(msg)
    parsed = _parse_json(msg.content)
    if parsed is None:
        # One corrective retry: models occasionally wrap or truncate the JSON.
        messages += [
            {"role": "assistant", "content": msg.content},
            {"role": "user", "content": "That was not valid JSON. Reply with ONLY the complete JSON object."},
        ]
        msg = llm.generate(messages, tools=None, temperature=0.2, task="finalize", notify=notify)
        _debug(msg)
        parsed = _parse_json(msg.content)
    return parsed


def _unaddressed_request_terms(product_query: str, parsed: dict) -> list[str]:
    """
    Meaningful words from the user's request that no stated requirement, the
    product type or a named material covers ("organic shampoo" answered with
    no "organic" requirement). Generic word comparison, no category rules;
    surfaced to the critique and the user, never auto-applied.
    """
    if _is_url(product_query):
        return []
    covered = set()
    for key in ("user_requirements", "functional_requirements", "safety_requirements"):
        for text in parsed.get(key) or []:
            covered |= {_stem(t) for t in _tokens(text)}
    for text in (parsed.get("product_type"), parsed.get("user_specified_material")):
        covered |= {_stem(t) for t in _tokens(text or "")}
    out = []
    for token in _tokens(product_query):
        if token in _CONTEXT_STOPWORDS or token.isdigit() or _stem(token) in covered or token in out:
            continue
        out.append(token)
    return out


def _lite_draft(parsed: dict, materials: list[dict], product_query: str = "") -> dict:
    state = normalize_research_state(parsed)
    return {
        "request_terms_not_covered_by_any_requirement": _unaddressed_request_terms(product_query, parsed),
        "summary": parsed.get("summary"),
        "product_type": state["product_type"],
        "user_requirements": state["user_requirements"],
        "functional_requirements": state["functional_requirements"],
        "safety_requirements": state["safety_requirements"],
        "environmental_dimensions": state["environmental_dimensions"],
        "focus_note": state["focus_note"],
        "conclusion": parsed.get("conclusion"),
        "request_sustainability": parsed.get("request_sustainability"),
        "user_specified_material": parsed.get("user_specified_material"),
        "materials": [
            {
                "type": m["type"],
                "matches_request": m.get("matches_request"),
                "why_different": m.get("why_different"),
                "products": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "why_suggested": p.get("why_suggested"),
                        "trade_off": p.get("trade_off"),
                        "requirement_checks": [
                            {"requirement": c["requirement"], "label": c["label"]}
                            for c in p.get("requirement_checks", [])
                        ],
                        "claims": [
                            {"claim": c["claim"], "evidence_snippet": c["evidence_snippet"], "label": c["label"]}
                            for c in p.get("claims", [])
                        ],
                    }
                    for p in m["products"]
                ],
            }
            for m in materials
        ],
    }


def _run_critique(product_query: str, parsed: dict, materials: list[dict], notify=None) -> dict:
    """
    Returns {"ran", "pass", "issues", "extra_search"}. Fails CLOSED: if the
    critique call errors or returns unparseable JSON it is reported as not
    run, never as passed.
    """
    try:
        msg = llm.generate(
            [
                {"role": "system", "content": CRITIQUE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User asked for: {product_query}\n\n"
                        f"Draft (JSON):\n{json.dumps(_lite_draft(parsed, materials, product_query), ensure_ascii=False)}\n\n"
                        "Review this now."
                    ),
                },
            ],
            tools=None, temperature=0.2, task="critique", notify=notify,
        )
    except llm.LLMError as exc:
        return {"ran": False, "pass": None, "issues": [], "extra_search": None, "error": str(exc)}

    _debug(msg)
    result = _parse_json(msg.content)
    if result is None:
        return {"ran": False, "pass": None, "issues": [], "extra_search": None, "error": "unparseable critique"}
    return {
        "ran": True,
        "pass": bool(result.get("pass", False)),
        "issues": [i for i in (result.get("issues") or []) if isinstance(i, dict)],
        "extra_search": result.get("extra_search"),
        # research.guard_conclusion applies this only if it is more conservative.
        "conclusion_override": result.get("conclusion_override"),
        "provider": msg.provider,
    }


def _roots(text: str | None) -> set[str]:
    """4-char word roots, so 'resistant'/'resistance' compare equal."""
    return {t[:4] for t in _tokens(text or "") if t not in _CONTEXT_STOPWORDS and len(t) >= 3}


def _high_impact_roots(parsed: dict) -> set[str]:
    if parsed.get("request_sustainability") != "high_impact":
        return set()
    return _roots(parsed.get("user_specified_material"))


def _functional_removals(critique: dict, parsed: dict, product_query: str) -> tuple[set[str], list[dict]]:
    """
    Product ids the critique may remove, plus the removals code refused.

    The critique decides; code enforces. A removal is accepted only when its
    requirement is grounded in the user's words or the stated functional
    requirements, and is never "not made of the high-impact material".
    (A live run removed every bamboo chair for "virgin plastic chair" because
    the critique treated the material as a requirement.)
    """
    material = _high_impact_roots(parsed)
    grounded = _roots(product_query)
    for key in ("user_requirements", "functional_requirements", "safety_requirements"):
        for requirement in parsed.get(key) or []:
            grounded |= _roots(requirement)
    grounded -= material

    accepted, refused = set(), []
    for issue in critique.get("issues", []):
        if not (issue.get("type") in ("functional_mismatch", "requirement_not_preserved")
                and issue.get("action") == "remove" and issue.get("product_id")):
            continue
        requirement = _roots(issue.get("requirement"))
        if requirement and requirement <= material:
            refused.append({**issue, "refused_because": "the high-impact material is not a requirement"})
        elif not (requirement & grounded):
            refused.append({**issue, "refused_because": "requirement not stated by the user"})
        else:
            accepted.add(issue["product_id"])
    return accepted, refused


def _remove_products(materials: list[dict], remove_ids: set[str]) -> tuple[list[dict], int]:
    if not remove_ids:
        return materials, 0
    removed, kept_materials = 0, []
    for material in materials:
        kept = [p for p in material.get("products", []) if p.get("id") not in remove_ids]
        removed += len(material.get("products", [])) - len(kept)
        if kept:
            kept_materials.append({**material, "products": kept})
    return kept_materials, removed


def _build_filter_options(materials: list[dict]) -> dict:
    material_options, prices, ratings, reviews = [], [], [], []

    for material in materials:
        if material.get("type") and material["type"] not in material_options:
            material_options.append(material["type"])
        for product in material.get("products", []):
            for bucket, value in (
                (prices, product.get("extracted_price")),
                (ratings, parse_number(product.get("rating"))),
                (reviews, parse_number(product.get("reviews"))),
            ):
                if value is not None:
                    bucket.append(value)

    def span(values):
        return {"min": min(values) if values else None, "max": max(values) if values else None}

    return {
        "material": material_options,
        "price": {"currency": "INR", **span(prices)},
        "rating": span(ratings),
        "reviews": span(reviews),
        "sort": SORT_OPTIONS,
    }


def _claim_counts(materials: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for m in materials:
        for p in m["products"]:
            for c in p.get("claims", []):
                counts[c["status"]] = counts.get(c["status"], 0) + 1
    return counts


# =====================================================================
# Orchestration
# =====================================================================

def run_agent(
    product_query: str,
    preferences: list[str] | None = None,
    max_price: float | None = None,
    on_event: EventSink = None,
) -> dict:
    """
    Run the full agent. `on_event`, if given, receives progress dicts
    ({"type": "phase" | "search" | "search_done" | "validate" | "critique", ...})
    so a UI can show the agent working live.
    """
    def emit(event_type: str, **data) -> None:
        if on_event:
            try:
                on_event({"type": event_type, **data})
            except Exception:
                pass  # a broken listener must never break the agent

    def on_llm_issue(provider: str, error: str) -> None:
        emit("llm_wait", provider=provider, error=error)

    product_query = (product_query or "").strip()
    prefs_text = (
        ", ".join(p.replace("_", " ") for p in preferences)
        if preferences else "none stated (give a balanced set)"
    )
    budget_text = f"under ₹{int(max_price)}" if max_price else "none stated"
    input_context = _url_input_context(product_query)

    trace: list[dict] = []
    catalog: dict[str, dict] = {}
    web_findings: list[dict] = []
    providers_used: set[str] = set()
    fallbacks: list[str] = []
    state = {"search_calls": 0, "evidence_calls": 0, "next_id": 1}

    def run_search(query: str, engine: str, purpose: str, model_query: str | None = None,
                   from_critique: bool = False) -> tuple[list[dict], str | None]:
        """One budgeted SerpApi call. Failures count against the budget too."""
        state["search_calls"] += 1
        if purpose == "evidence":
            state["evidence_calls"] += 1
        step = state["search_calls"]
        emit("search", step=step, query=query, engine=engine, purpose=purpose)

        entry = {"step": step, "query": query, "engine": engine, "purpose": purpose}
        if model_query and model_query != query:
            entry["model_query"] = model_query
        if from_critique:
            entry["from_critique"] = True

        try:
            results, cached = serp_search(query, engine)
        except Exception as exc:
            trace.append({**entry, "results_found": 0, "cached": False, "error": str(exc)})
            emit("search_done", step=step, results_found=0, cached=False, error=str(exc))
            return [], str(exc)

        trace.append({**entry, "results_found": len(results), "cached": cached})
        emit("search_done", step=step, results_found=len(results), cached=cached)
        if engine == "web":
            web_findings.append({"query": query, "purpose": purpose, "results": results})
        return results, None

    shown: dict[str, dict] = {}

    def register(results: list[dict]) -> list[str]:
        ids = []
        for item in results:
            pid = f"p{state['next_id']}"
            state["next_id"] += 1
            catalog[pid] = item
            ids.append(pid)
        for pid in ids[:GATHER_VIEW_LIMIT]:
            shown[pid] = catalog[pid]
        return ids

    # ---------------- PHASE 0: URL identification ----------------
    if input_context["is_url"]:
        emit("phase", phase="identify", message="Identifying the product behind the URL")
        run_search(_url_search_query(product_query), "web", "url_product_identification")

    # ---------------- PHASE 1: GATHER ----------------
    emit("phase", phase="gather", message="Planning searches and gathering real listings")
    messages = [
        {"role": "system", "content": GATHER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Product / URL: {product_query}\n"
                f"Input context: {json.dumps(input_context, ensure_ascii=False)}\n"
                f"Preferences (applied later by code, NOT search keywords): {prefs_text}\n"
                f"Budget (applied later by code, NOT search keywords): {budget_text}"
                + (
                    f"\nURL identification results (JSON): {json.dumps(web_findings, ensure_ascii=False)}"
                    if web_findings else ""
                )
            ),
        },
    ]
    parsed = None
    llm_error = None

    for _ in range(MAX_LLM_TURNS):
        if state["search_calls"] >= MAX_SEARCH_CALLS:
            break

        try:
            msg = llm.generate(messages, tools=TOOLS, temperature=0.3, task="gather", notify=on_llm_issue)
        except llm.LLMError as exc:
            llm_error = str(exc)
            break
        providers_used.add(msg.provider)
        if msg.fallback_errors:
            fallbacks.extend(msg.fallback_errors)
            emit("fallback", provider=msg.provider, errors=msg.fallback_errors)
        _debug(msg)
        messages.append(_assistant_msg(msg))

        if msg.content and msg.tool_calls:
            emit("thought", text=_short(msg.content, 400))

        if not msg.tool_calls:
            parsed = _parse_json(msg.content)
            if parsed is not None:
                break
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object described in the instructions."})
            continue

        for index, tc in enumerate(msg.tool_calls):
            tool_id = tc.get("id") or f"call_{index + 1}"

            if state["search_calls"] >= MAX_SEARCH_CALLS:
                messages.append(_tool_msg(tool_id, "Search cap reached."))
                continue

            try:
                args = json.loads(tc.get("function", {}).get("arguments") or "{}")
                query = str(args["query"]).strip()
                engine = args.get("engine", "shopping")
                purpose = args.get("purpose", "discovery")
                avoid_terms = [str(t) for t in (args.get("avoid_terms") or []) if t]
            except (json.JSONDecodeError, KeyError, TypeError):
                messages.append(_tool_msg(tool_id, "Invalid arguments. Call search with a query and an engine."))
                continue

            if engine not in ("shopping", "web"):
                engine = "shopping"
            if purpose not in ("discovery", "evidence"):
                purpose = "discovery"
            if not query:
                messages.append(_tool_msg(tool_id, "Search query cannot be empty."))
                continue

            if purpose == "evidence" and state["evidence_calls"] >= MAX_EVIDENCE_CALLS:
                messages.append(_tool_msg(
                    tool_id,
                    f"Evidence search cap reached ({MAX_EVIDENCE_CALLS} max). "
                    "Use a discovery search instead, or answer using listing-based claims.",
                ))
                continue

            model_query = query
            if engine == "shopping" and purpose == "discovery":
                query = _augment_shopping_query(query, product_query, avoid_terms)

            results, error = run_search(query, engine, purpose, model_query=model_query)
            if error:
                messages.append(_tool_msg(tool_id, f"Search failed: {error}"))
                continue

            if engine == "shopping":
                ids = register(results)
                payload = [_gather_view(pid, catalog[pid]) for pid in ids[:GATHER_VIEW_LIMIT]]
            else:
                payload = [{"title": r.get("title"), "snippet": r.get("snippet")} for r in results]

            messages.append(_tool_msg(tool_id, json.dumps(payload, ensure_ascii=False)))

    # Shopping signals come from the complete catalog, before finalization.
    _add_shopping_signals(catalog)

    # ---------------- PHASE 2: FINALIZE ----------------
    if parsed is None:
        emit("phase", phase="finalize", message="Assembling the answer from gathered research")
        try:
            parsed = _finalize_clean(product_query, prefs_text, budget_text, shown, web_findings, on_llm_issue)
        except llm.LLMError as exc:
            llm_error = str(exc)
            parsed = None

    base_meta = {"searches": state["search_calls"], "evidence_searches": state["evidence_calls"]}

    if parsed is None:
        return {
            "query": product_query,
            "summary": "The agent did not produce a valid answer. Please try again.",
            "input_mode": "url" if input_context["is_url"] else "text",
            "input_source": input_context.get("source"),
            "input_url": input_context.get("url"),
            "input_asin": input_context.get("asin"),
            "materials": [],
            "filter_options": _build_filter_options([]),
            "availability_note": AVAILABILITY_NOTE,
            "no_suitable_alternative": True,
            "trace": trace,
            "meta": {**base_meta, "dropped_unknown_ids": 0,
                     "error": llm_error or "no_valid_json"},
        }

    # ---------------- PHASE 3: VALIDATE / ENFORCE ----------------
    def build(parsed_answer: dict) -> tuple[list[dict], dict]:
        mats, dropped_ids = _resolve(parsed_answer, catalog)
        mats, dropped_type = filter_by_product_type(mats, parsed_answer.get("product_type"))
        high_impact_removed = 0
        if parsed_answer.get("request_sustainability") == "high_impact":
            high_impact_removed = sum(len(m["products"]) for m in mats if m.get("matches_request"))
            mats = [m for m in mats if not m.get("matches_request")]
        # Function first: requirement suitability is enforced before any
        # environmental claim is even validated.
        mats, requirement_removed = enforce_user_requirements(
            mats, normalize_research_state(parsed_answer), web_findings,
        )
        mats = validate_claims(mats, web_findings)
        stats = {
            "dropped_unknown_ids": dropped_ids,
            "dropped_type_mismatch": dropped_type,
            "removed_high_impact": high_impact_removed,
            "removed_requirement_mismatch": len(requirement_removed),
            "requirement_removals": requirement_removed,
        }
        emit("validate", **{k: v for k, v in stats.items() if k != "requirement_removals"},
             claims=_claim_counts(mats), products=sum(len(m["products"]) for m in mats))
        return mats, stats

    emit("phase", phase="validate", message="Checking product types and evidence in code")
    materials, stats = build(parsed)

    # ---------------- PHASE 4: SELF-CRITIQUE ----------------
    emit("phase", phase="critique", message="Self-critique of the draft")
    critique = _run_critique(product_query, parsed, materials, on_llm_issue)
    removal_ids, refused = _functional_removals(critique, parsed, product_query)
    materials, removed = _remove_products(materials, removal_ids)
    emit("critique", ran=critique["ran"], passed=critique["pass"],
         issues=critique["issues"], removed=removed, refused=len(refused))

    critique_meta = {
        "ran": critique["ran"], "passed": critique["pass"], "issues": critique["issues"],
        "refused_removals": refused, "extra_search_used": False, "rechecked": False,
    }

    extra = critique.get("extra_search")
    if isinstance(extra, dict) and extra.get("query"):
        material = _high_impact_roots(parsed)
        if material and material <= _roots(str(extra["query"])):
            critique_meta["extra_search_refused"] = "it searched for the high-impact material"
            extra = None
    if (
        critique["ran"] and not critique["pass"] and isinstance(extra, dict)
        and extra.get("query") and state["search_calls"] < MAX_SEARCH_CALLS
    ):
        eng = extra.get("engine") if extra.get("engine") in ("shopping", "web") else "shopping"
        purpose = "evidence" if eng == "web" else "discovery"
        # The evidence sub-cap applies to the critique's follow-up too.
        if purpose == "evidence" and state["evidence_calls"] >= MAX_EVIDENCE_CALLS:
            purpose = "discovery"
        q = str(extra["query"]).strip()
        if eng == "shopping":
            q = _augment_shopping_query(q, product_query, [parsed.get("user_specified_material") or ""]
                                        if parsed.get("request_sustainability") == "high_impact" else None)

        emit("phase", phase="critique_search", message="Running one follow-up search the critique asked for")
        results, error = run_search(q, eng, purpose, from_critique=True)
        if not error:
            if eng == "shopping":
                register(results)
                _add_shopping_signals(catalog)
            try:
                parsed2 = _finalize_clean(product_query, prefs_text, budget_text, shown, web_findings, on_llm_issue)
            except llm.LLMError:
                parsed2 = None
            if parsed2 is not None:
                parsed = parsed2
                materials, stats = build(parsed)
                critique_meta["extra_search_used"] = True

                # Re-check the rebuilt draft once (no further searches).
                recheck = _run_critique(product_query, parsed, materials, on_llm_issue)
                more_ids, more_refused = _functional_removals(recheck, parsed, product_query)
                removal_ids |= more_ids
                refused += more_refused
                materials, removed = _remove_products(materials, removal_ids)
                critique_meta.update({
                    "rechecked": recheck["ran"],
                    "recheck_passed": recheck["pass"],
                    "recheck_issues": recheck["issues"],
                })
                emit("critique", ran=recheck["ran"], passed=recheck["pass"],
                     issues=recheck["issues"], removed=removed, recheck=True)

    research = normalize_research_state(parsed)
    override = critique.get("conclusion_override")
    if critique_meta.get("rechecked"):
        override = recheck.get("conclusion_override") or override
    conclusion = guard_conclusion(parsed.get("conclusion"), materials, override)
    language_flags = absolute_language(answer_texts(parsed.get("summary"), conclusion, materials))
    if language_flags and conclusion["verdict"] != "clear_advantage":
        original = conclusion.get("downgraded_from") or conclusion["verdict"]
        conclusion = guard_conclusion(conclusion, materials, "no_clear_winner")
        if conclusion["verdict"] != original:
            conclusion["downgraded_from"] = original
            conclusion["downgrade_reason"] = "the answer used verdict language the evidence does not support"
    emit("conclusion", verdict=conclusion["verdict"], label=conclusion["label"])
    emit("phase", phase="done", message="Done")

    return {
        "query": product_query,
        "summary": parsed.get("summary", ""),
        "input_mode": "url" if input_context["is_url"] else "text",
        "input_source": input_context.get("source"),
        "input_url": input_context.get("url"),
        "input_asin": input_context.get("asin"),
        **research,
        "conclusion": conclusion,
        "language_flags": language_flags,
        "unaddressed_request_terms": _unaddressed_request_terms(product_query, parsed),
        "user_specified_material": parsed.get("user_specified_material"),
        "request_sustainability": parsed.get("request_sustainability", "none"),
        "caution_note": parsed.get("caution_note"),
        "materials": materials,
        "filter_options": _build_filter_options(materials),
        "availability_note": AVAILABILITY_NOTE,
        "no_suitable_alternative": len(materials) == 0,
        "trace": trace,
        "meta": {
            **base_meta,
            "searches": state["search_calls"],
            "evidence_searches": state["evidence_calls"],
            **stats,
            "dropped_functional_mismatch": removed,
            "llm_fallbacks": fallbacks[:5],
            "providers": sorted(providers_used | ({critique.get("provider")} if critique.get("provider") else set())),
            "shopping_signals": {
                "popular_definition": "high review volume among listings found in this run",
                "offer_definition": "current price lower than a retrieved previous/old price",
                "not_sales_data": True,
            },
            "critique": critique_meta,
        },
    }
