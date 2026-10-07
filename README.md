# GreenSwap 🌱

**Before you buy, swap smarter.** Type what you're about to buy, or paste a product URL. GreenSwap's AI agent finds environmentally preferable alternatives as **real products you can buy in India**, and labels every claim with how well it is backed.

Built for the SerpApi India Hackathon 2026.

## Why it's different

- **A real agent, not a keyword list.** For each product it works out which environmental dimensions actually matter (material, refill format, longevity, formulation…). It then plans its own Google Shopping and Google web searches through SerpApi, reads the results and decides what to search next.
- **The LLM decides, code enforces.**
  - Code caps the number of searches.
  - The model can only pick products by id, so it can't invent prices or links.
  - Wrong-type listings are dropped.
  - A high-impact material you asked for ("virgin plastic chair") is withheld, and you get a caution note.
  - Every claim is checked against text actually retrieved.
- **Honest evidence labels.** Each claim is labelled *Evidence found* (product-specific web source), *Stated by seller* (listing text), *General info* or *Not verified*. There is no made-up eco score.
- **Self-critique with guardrails.** A second model reviews the draft and can remove unsuitable products. Code refuses removals that aren't grounded in what you asked for.
- **Live agent timeline.** You watch every search, guardrail and critique step as it happens.
- **Compare stores.** One click checks other sellers for the same product using SerpApi's Immersive Product API.

## Run it

Requirements: Python 3.11+ with [uv](https://docs.astral.sh/uv/), and Node 20+.

```bash
# backend
cd backend
uv sync
# create backend/.env with: SERPAPI_KEY, GEMINI_API_KEY and/or GROQ_API_KEY
uv run uvicorn api:app --port 8000

# frontend (dev, proxies /api to :8000)
cd frontend
npm install
npm run dev        # http://localhost:5173
```

For a one-process demo, run `npm run build` in `frontend/`. FastAPI then serves `frontend/dist` at http://localhost:8000.

Optional environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LLM_PRIMARY` | `gemini` | Which provider runs the search loop |
| `LLM_CRITIQUE_PROVIDER` | `groq` (if its key is set) | Which provider runs the self-critique |
| `GEMINI_MODEL` / `GROQ_MODEL` | `gemini-3.7-flash` / `openai/gpt-oss-120b` | Model ids |
| `SERP_CACHE` | `1` | Set `0` to bypass the on-disk SerpApi cache |
| `AGENT_DEBUG` | `0` | Set `1` to print raw LLM responses |

If one provider is overloaded or rate limited, the other takes over automatically.

## Demo without credits

Saved runs in `backend/demo_runs/` appear in the UI under **Replay a saved run**. They replay the full agent timeline with no SerpApi or LLM calls.

## Tests

```bash
cd backend
uv run pytest -q
```

The tests cover the pure-code guardrails: evidence validation, product-type matching, ranking, query context, link policy and critique refusals. They need no keys.

## Layout

```
backend/
  agent.py         four-phase agent: gather → finalize → validate → self-critique
  llm.py           Gemini + Groq with retry, cooldown and automatic fallback
  serp_tool.py     SerpApi: Google Shopping, Google web, Immersive Product (+ disk cache)
  evidence.py      claim validation (pure code)
  product_match.py product-type enforcement (pure code)
  ranking.py       budget, sorting, top picks (pure code, re-runnable for free)
  api.py           FastAPI: SSE search stream, re-rank, store offers, demo replay
  test_agent.py    CLI runner
  tests/           offline pytest suite
frontend/          React + Vite + Tailwind UI
ARCHITECTURE.md    design notes and guardrail inventory
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the full list of guardrails.
