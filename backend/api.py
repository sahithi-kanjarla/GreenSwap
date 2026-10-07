"""
GreenSwap HTTP API (FastAPI).

    GET  /api/health                 which LLM providers / SerpApi are configured
    GET  /api/search?q=...           runs the agent, streams progress as Server-Sent Events
         &preferences=low_cost&max_price=500&sort_by=recommended
         &replay=1                   replay a saved demo run instead (no credits, works offline)
    POST /api/rank                   re-rank / re-filter a finished run (pure code, free)
    GET  /api/offers?token=...       compare sellers for one product (SerpApi Immersive Product)
    GET  /api/demos                  saved demo runs
    GET  /api/demo/{slug}            one saved demo run (raw + ranked view)

Run:  uvicorn api:app --reload --port 8000
If ../frontend/dist exists it is served at "/" so one process can host the demo.
"""

import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm  # noqa: E402
from agent import run_agent  # noqa: E402
from ranking import finalize, SORT_OPTIONS  # noqa: E402
from serp_tool import get_normalized_product_offers  # noqa: E402

BASE_DIR = Path(__file__).parent
DEMO_DIR = BASE_DIR / "demo_runs"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

app = FastAPI(title="GreenSwap API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Finished runs kept in memory so re-ranking never re-runs the agent.
RUNS: dict[str, dict] = {}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:120] or "query"


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _parse_prefs(preferences: str | None) -> list[str]:
    return [p.strip() for p in (preferences or "").split(",") if p.strip()]


def _view(raw: dict, prefs: list[str], max_price: float | None, sort_by: str) -> dict:
    if sort_by not in SORT_OPTIONS:
        sort_by = "recommended"
    return finalize(raw, preferences=prefs, max_price=max_price, sort_by=sort_by)


def _load_demo(slug: str) -> dict:
    path = DEMO_DIR / f"{_slug(slug)}.json"
    if not path.exists():
        raise HTTPException(404, f"No saved demo run for '{slug}'.")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/api/health")
def health():
    import os
    return {
        "ok": True,
        "llm_providers": llm.available_providers(),
        "serpapi": bool(os.environ.get("SERPAPI_KEY")),
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    preferences: str | None = None,
    max_price: float | None = None,
    sort_by: str = "recommended",
    replay: bool = False,
):
    prefs = _parse_prefs(preferences)

    def finish(raw: dict) -> dict:
        run_id = uuid.uuid4().hex[:12]
        RUNS[run_id] = raw
        return {"type": "result", "run_id": run_id, "raw": raw, "view": _view(raw, prefs, max_price, sort_by)}

    if replay:
        raw = _load_demo(q)

        def replay_stream():
            yield _sse({"type": "phase", "phase": "gather", "message": "Replaying a saved run (no credits used)", "replay": True})
            for step in raw.get("trace", []):
                time.sleep(0.6)
                yield _sse({"type": "search", **{k: step.get(k) for k in ("step", "query", "engine", "purpose")}})
                time.sleep(0.4)
                yield _sse({"type": "search_done", "step": step.get("step"), "results_found": step.get("results_found", 0),
                            "cached": True, "error": step.get("error")})
            for phase, message in (("finalize", "Assembling the answer"), ("validate", "Checking types and evidence in code"),
                                   ("critique", "Self-critique of the draft")):
                time.sleep(0.5)
                yield _sse({"type": "phase", "phase": phase, "message": message})
            critique = (raw.get("meta") or {}).get("critique") or {}
            yield _sse({"type": "critique", "ran": critique.get("ran"), "passed": critique.get("passed"),
                        "issues": critique.get("issues", []), "removed": (raw.get("meta") or {}).get("dropped_functional_mismatch", 0)})
            yield _sse({"type": "phase", "phase": "done", "message": "Done"})
            yield _sse(finish(raw))

        return StreamingResponse(replay_stream(), media_type="text/event-stream")

    events: queue.Queue = queue.Queue()
    DONE = object()

    def worker():
        try:
            raw = run_agent(q, preferences=prefs, max_price=max_price, on_event=events.put)
            DEMO_DIR.mkdir(exist_ok=True)
            (DEMO_DIR / f"{_slug(q)}.json").write_text(
                json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            events.put(finish(raw))
        except Exception as exc:  # surface any failure to the client
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if event is DONE:
                break
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RankRequest(BaseModel):
    run_id: str | None = None
    demo: str | None = None
    preferences: list[str] = []
    max_price: float | None = None
    sort_by: str = "recommended"


@app.post("/api/rank")
def rank(req: RankRequest):
    if req.run_id:
        raw = RUNS.get(req.run_id)
        if raw is None:
            raise HTTPException(404, "Unknown run_id (the server may have restarted).")
    elif req.demo:
        raw = _load_demo(req.demo)
    else:
        raise HTTPException(400, "Provide run_id or demo.")
    return _view(raw, req.preferences, req.max_price, req.sort_by)


@app.get("/api/offers")
def offers(token: str = Query(..., min_length=10)):
    try:
        items, cached = get_normalized_product_offers(token)
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch store offers: {exc}")
    return {"offers": items, "cached": cached}


@app.get("/api/demos")
def demos():
    out = []
    for path in sorted(DEMO_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "slug": path.stem,
            "query": raw.get("query") or path.stem.replace("-", " "),
            "summary": raw.get("summary"),
            "products": sum(len(m.get("products", [])) for m in raw.get("materials", [])),
        })
    return out


@app.get("/api/demo/{slug}")
def demo(slug: str, sort_by: str = "recommended"):
    raw = _load_demo(slug)
    run_id = f"demo-{_slug(slug)}"
    RUNS[run_id] = raw
    return {"run_id": run_id, "raw": raw, "view": _view(raw, [], None, sort_by)}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
