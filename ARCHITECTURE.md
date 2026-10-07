# GreenSwap — Architecture

_Snapshot of the codebase as of commit 8a73094 (before the hackathon refactor). Section 6 lists known issues at that point._

## 1. Purpose
GreenSwap is an AI shopping agent for Indian shoppers. A user types a product ("chair"), a qualified
need ("lunch box for college"), a material preference ("virgin plastic chair") or pastes a product URL.
The agent works out what is actually being bought and which environmental dimensions matter for
*that* product (material origin, refill format, longevity, formulation…). It then searches live Google
Shopping and Google web results through SerpApi, groups real purchasable listings into genuinely
different "alternative" types, and attaches claims that code checks against text actually retrieved.
It deliberately avoids an "eco score". It labels each claim with its source ("Stated by seller",
"Evidence found", "Not verified") and keeps buying signals (Popular, Offer, Lowest price) separate
from environmental reasoning.

## 2. The four-phase agent loop (`backend/agent.py::run_agent`)
| Phase | What happens | Why it exists |
|---|---|---|
| 0. URL pre-step | If input is a URL, one web search identifies the product (`_url_search_query`, ASIN extraction for Amazon). | The model must not guess material or price from a URL slug. |
| 1. GATHER | Tool-calling loop (≤ `MAX_LLM_TURNS`=8). The model calls `search(query, engine, purpose)`. Every shopping listing goes into `catalog` under an id `p1, p2…`, and the model only ever sees and refers to ids. | Gives the model real agency (reason → search → observe) while making it impossible to invent a price, link or product: those come from `catalog` later. |
| 2. FINALIZE | If phase 1 did not end with valid JSON (usually because the search cap was hit), `_finalize_clean` opens a **new, tool-free conversation** with only the gathered listings and web notes and asks for the final JSON. | **Bug workaround:** with Groq's `openai/gpt-oss-120b`, simply leaving `tools` out of the request was not enough. The model still emitted a tool call because the tool was *described in the system prompt and history*, and Groq rejected the request as invalid. The fix is a clean conversation with no tool-call history to prime it, not "telling it harder". |
| 3. VALIDATE / ENFORCE | `_resolve` maps ids to real listing data. `filter_by_product_type` drops wrong-category items. High-impact groups are stripped. `validate_claims` relabels every claim from retrieved text. | The model's output is treated as a proposal. Code decides what survives. |
| 4. SELF-CRITIQUE | One extra LLM call (no SerpApi cost) reviews a slim draft against a checklist. `functional_mismatch` + `remove` issues delete those product ids. If it fails, asks for an `extra_search` and budget remains, the agent runs **exactly one** more search and one re-finalize, never a loop. | Catches functional mismatches (for example a glass bottle for a toddler) that per-field validators can't see. The prompt explicitly lists intended behaviour so the critique does not "fix" it. |

## 3. Files and how they connect
```
test_agent.py (CLI) ──► agent.run_agent(query, prefs, max_price) ──► ranking.finalize(raw, prefs, max_price, sort)
                              │                                            (pure code, re-runnable for free)
                              ├─► serp_tool.serp_search(query, engine)  → SerpApi google_shopping / google (+ disk cache)
                              ├─► product_match.filter_by_product_type
                              ├─► evidence.validate_claims
                              └─► Gemini (google-genai) via _gemini_generate / _gemini_message adapters
```
- **agent.py** (~2000 lines): prompts (`OUTPUT_RULES`, `GATHER_PROMPT`, `FINALIZE_PROMPT`, `CRITIQUE_PROMPT`), tool schema, the Gemini adapter layer (converts OpenAI-style history into Gemini `Content` and keeps `thought_signature` parts), URL parsing, query augmentation, shopping signals, the orchestration, and the output shape (`filter_options`, `meta`, `trace`).
- **serp_tool.py**: SerpApi wrapper, India-localized (`gl=in`, `google.co.in`). It normalizes shopping results (≤20) and web results (≤5) and uses a SHA1-keyed disk cache in `backend/.cache/`. It also has an **Immersive Product API** (`get_product_offers`, `normalize_product_offers`) for multi-store price comparison, but nothing calls it yet. Debug helpers are included.
- **evidence.py**: pure-code claim validator. It does a normalized substring match of `evidence_snippet` against listing fields and then against retrieved web results, and assigns a status and label.
- **product_match.py**: pure-code product-type filter (substring of `product_type` in the title, with an "at least one match in group" gate).
- **ranking.py**: pure-code ranking, budget filter, sorting, `top_picks` and `cheapest_found`. It has no eco score; each product gets a plain-language `why_ranked`.
- **test_agent.py**: CLI runner. It prints trace, summary, top picks, groups and critique, and saves the raw run to `demo_runs/<slug>.json`.
- **main.py**: `uv init` placeholder. **test_connections.py**, **README.md**: empty.
- **demo_runs/**: 12 saved runs from *earlier* agent versions (see risks).

## 4. Core principle: "the LLM decides, code enforces"
| Guardrail | Where |
|---|---|
| Total SerpApi cap `MAX_SEARCH_CALLS=5` (checked before every call, failed calls count too) | agent.py `run_agent` phase 1 |
| Evidence sub-cap `MAX_EVIDENCE_CALLS=1`: excess evidence calls are rejected with a tool message | agent.py `run_agent` phase 1, also applied to the critique follow-up |
| LLM turn ceiling `MAX_LLM_TURNS=8` | agent.py |
| Critique gets at most one extra search and one rebuild | agent.py phase 4 |
| Products referenced only by id; unknown ids dropped (`dropped_unknown_ids`) and duplicates de-duped | agent.py `_resolve` |
| Price, link, image and rating always come from catalog data, never from the model | agent.py `_resolve` |
| Search URLs never labelled as product pages | agent.py `_best_product_link`, `_looks_like_search_url` |
| Evidence-snippet validation: model status ignored, label recomputed from retrieved text | evidence.py `validate_claims` |
| Product-type matching against the listing title | product_match.py |
| High-impact material stripping (`matches_request` groups removed when `high_impact`) | agent.py `run_agent` (both build paths) |
| Functional-mismatch removal by product id from the critique | agent.py `_apply_functional_critique` |
| User-context safety net appends dropped request words to discovery queries | agent.py `_augment_shopping_query` |
| Shopping signals computed in code (Popular = top-quartile reviews *in this run*, never "most sold"; Offer only from real old/new prices) | agent.py `_add_shopping_signals`, `_offer_signal` |
| Budget/preferences kept out of search queries and applied post-hoc | prompts + ranking.py |
| Arg sanitising: bad engine/purpose coerced, empty query rejected | agent.py phase 1 |

## 5. Not built yet
- No API server (FastAPI and uvicorn are in `pyproject.toml` but unused; `main.py` is a stub).
- No frontend.
- No automated tests. `test_agent.py` is a live CLI script and `test_connections.py` is empty.
- Immersive Product price comparison exists in `serp_tool.py` but is not wired in.
- No Groq↔Gemini fallback. Groq is commented out and Gemini is the only provider.
- No README or setup docs.

## 6. Inconsistencies, dead code and risks (not fixed)
**Correctness bugs**
1. **`_augment_shopping_query` re-injects the high-impact material.** For "virgin plastic chair", a model query "bamboo chair" becomes "bamboo chair virgin plastic". This works against the high-impact logic. Long requests also leak filler words ("that", "will", "long", "time", "doesn"), because the stop-list is short and up to 8 tokens get appended.
2. **Link contradiction.** `serp_tool` says the raw SerpApi `product_link` is session-dependent and must *not* be the user-facing link, so it sets `link` to a stable Shopping search URL. `agent._best_product_link` does the opposite: it prefers `product_link`. Google `/shopping/product/...` URLs don't match `_looks_like_search_url`, so they get labelled "Direct product listing" (`merchant_product`) even though they aren't merchant pages.
3. **Evidence validator blind spots.**
   (a) The finalize model is shown listing `snippet`, but `_find_listing_evidence` checks only `name/title/source/description/listing_text`. Claims quoted from the snippet are wrongly marked "Not verified".
   (b) Matching against `source` (merchant name) lets a trivial snippet validate.
   (c) There is no minimum snippet length, so "steel" validates "made of 100% stainless steel, BPA-free".
   (d) Web evidence is not tied to the product. Any retrieved web result containing the snippet validates the claim for *any* product.
4. **Critique result is not re-checked after the rebuild.** After the extra search, the new draft is filtered with the *old* critique only.
5. **Critique fails open.** Unparseable critique JSON is treated as `pass: True`.
6. **Search-count inconsistency.** Phase 1 counts failed searches against the budget; the critique follow-up does not, and its error trace uses `search_calls + 1`.
7. **Product-type substring matching is fragile.** "cup" matches "cupboard"; "knife" misses "knives"; "lunch box" misses "Lunchbox"; "food container" misses "Food Storage Container". The last two drop correct items when a sibling does match.
8. **The phase-2 workaround is only partial.** `FINALIZE_PROMPT` embeds `OUTPUT_RULES`, which still talks about `purpose="evidence"` searches. That is the same priming pattern the workaround was meant to remove. It doesn't matter on Gemini today, but it will on Groq.

**Inconsistencies / dead code**

9. Docstrings say "Groq", but the code is Gemini-only. The user brief says "Groq with Gemini fallback"; no fallback exists.
10. `pyproject.toml` lists `groq` but **not `google-genai`**, which is what actually runs. A fresh `uv sync` will crash on import.
11. The uncommitted change `MODEL = "gemini-3.7-flash"` (was 3.8) is a hard-coded model id with no config/env override.
12. The `filter_options.sort` values (`relevance`, `price_low_to_high`, `rating`, `reviews`) don't match `ranking.py` sort keys (`recommended`, `price_low_high`, `price_high_low`). `rating` and `reviews` sorting isn't implemented.
13. There are three separate price parsers: `agent._numeric_price`, `agent._price_number`, `ranking.extract_price`.
14. Two different "lowest price" notions: `shopping_signals.lowest_price_in_run` (whole run) and `ranking.is_lowest_price` (per group).
15. Dead code: `LINK_TYPE` and `LINK_NOTE` in agent.py; the `conflicting_evidence` status is never produced; `link_type`/`link_note` set in serp_tool are overwritten; the high-impact branches in `ranking.build_top_picks`/`finalize` can never trigger because those groups were already stripped; `ranking._score` says "added in the next build step" (it is already built); `info_source` is hard-coded to "general knowledge".
16. `test_agent.py` suggests a `"reusable"` preference, but ranking only understands `low_cost`.
17. `max_price` reaches the agent only as prompt text. That's fine, but it's not obvious.

**Risks**

18. **The `demo_runs/` files are stale.** They predate current guardrails: they show 3 evidence searches against a cap of 1, ≤8 results, no `purpose`, and a critique flagging "virgin plastic missing" as a bug. They should not be shown to judges as-is.
19. `agent.py` reads `GEMINI_API_KEY` at import and `serp_tool` reads `SERPAPI_KEY` at import, so importing either module without keys crashes. That matters for an API server and for tests.
20. Everything is synchronous and one run means 6–9 LLM calls plus 5 SerpApi calls. There's no streaming, so a UI would sit blank for 30–90 s.
21. Prompt size: `OUTPUT_RULES` is ~430 lines and is sent on every turn. That costs latency and tokens, and invites instruction drift.
22. The cache never expires (prices go stale), and the cache key ignores the `gl`/`hl` params.
23. Currency is assumed to be INR everywhere.


## 7. Status after the hackathon refactor

Items from section 6 that have been addressed:

| # | Issue | Resolution |
|---|---|---|
| 1 | Augmentation re-injected the high-impact material | New `avoid_terms` tool argument; code never re-adds those words. Larger stop-list, plural-aware, max 3 words (`agent._augment_shopping_query`) |
| 2 | Link contradiction | One policy in `agent._best_product_link`: a real merchant URL, else the stable Shopping search, else the Google product page. Google URLs are never labelled "merchant". |
| 3 | Evidence blind spots | Listing snippet/extensions count as listing text; the merchant name doesn't. Snippets must be ≥3 words or ≥15 chars and must mention the claim's topic. Web evidence must be about the product, otherwise it is labelled `general_evidence`. |
| 4, 5 | Critique not re-checked; failed open | The rebuilt draft is critiqued again. Critique failure is reported as `ran: false`, never as passed. |
| — | Critique removed every bamboo chair for "virgin plastic chair" (found in a live run) | `agent._functional_removals` accepts only removals grounded in the user's words or the stated requirements, and never for the high-impact material. Critique follow-up searches for that material are refused. |
| 6 | Search counting | A single `run_search` helper counts every attempt, including failures. |
| 7 | Product-type matching | Whole-word, plural-aware, any-order and compound matching (`product_match.title_matches`) |
| 8, 9, 19, 21 | Prompt priming, provider mismatch, import-time keys, prompt size | `llm.py`: Gemini is primary, Groq handles the critique and fallback, with retry and cooldown. Clients start lazily. The prompts are rewritten without duplication, and `FINALIZE_PROMPT` never mentions searching. |
| 10, 11 | Missing dependency, hard-coded model | `google-genai` added to `pyproject.toml`. Model ids come from env (`GEMINI_MODEL`, `GROQ_MODEL`). |
| 12, 13 | Sort keys, three price parsers | `ranking.SORT_OPTIONS` is used everywhere, `rating`/`reviews` sorts were added, and `ranking.extract_price` is the single parser (now handles Indian digit grouping). |
| 18 | Stale demo runs | Regenerated through the new API |
| 20 | No streaming | `run_agent(on_event=...)` plus SSE in `api.py` |
| — | Not built | FastAPI server (`api.py`), React frontend (`frontend/`), offline pytest suite (`backend/tests/`), README |

Still open: per-group vs per-run "lowest price" (14), the cache has no expiry (22), INR assumed (23), and Google Lens photo input (deferred).
