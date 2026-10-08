"""
Offline stand-ins for the LLM and SerpApi so a case runs through the REAL
agent pipeline with no network calls.

The scripted model searches once per entry in case["searches"], then gives
case["answer"]; the critique replies with case["critique"] (default: pass).
Listing ids are assigned by the agent in fixture order (p1, p2, ...).
"""

import json

import agent
import llm


def listing(title: str, price: str = "₹499", snippet: str | None = None,
            source: str = "Amazon.in", rating: float | None = 4.2, reviews: int | None = 120) -> dict:
    """A shopping result in serp_tool's normalized shape."""
    return {
        "title": title, "price": price, "extracted_price": None,
        "old_price": None, "extracted_old_price": None, "source": source,
        "product_link": None, "merchant_link": None,
        "link": f"https://www.google.co.in/search?q={title.replace(' ', '+')}&tbm=shop",
        "image": None, "rating": rating, "reviews": reviews, "delivery": None,
        "snippet": snippet, "extensions": [], "tag": None, "badge": None,
        "multiple_sources": None, "product_id": None,
        "immersive_product_page_token": None, "serpapi_immersive_product_api": None,
    }


class ScriptedLLM:
    def __init__(self, case: dict):
        self.case = case
        self.searches = list(case.get("searches") or [{"query": case["query"], "engine": "shopping"}])
        self.calls: list[str] = []

    def generate(self, messages, tools=None, temperature=0.3, task="gather", notify=None):
        self.calls.append(task)
        if task == "critique":
            return llm.Message(json.dumps(self.case.get("critique") or {"pass": True, "issues": []}), provider="fake")
        if task == "gather" and self.searches:
            search = self.searches.pop(0)
            args = {"query": search["query"], "engine": search.get("engine", "shopping"),
                    "purpose": search.get("purpose", "discovery")}
            call = {"id": f"call_{len(self.calls)}", "type": "function",
                    "function": {"name": "search", "arguments": json.dumps(args)}}
            return llm.Message("", [call], provider="fake")
        return llm.Message(json.dumps(self.case["answer"]), provider="fake")


def fake_serp(case: dict):
    shopping = list(case.get("listings") or [])
    web = list(case.get("web") or [])

    def serp_search(query: str, engine: str = "shopping"):
        if engine == "shopping":
            results, shopping[:] = list(shopping), []  # every listing returned once
            return results, True
        return web, True
    return serp_search


def run_offline(case: dict, monkeypatch) -> dict:
    from ranking import finalize
    scripted = ScriptedLLM(case)
    monkeypatch.setattr(agent.llm, "generate", scripted.generate)
    monkeypatch.setattr(agent, "serp_search", fake_serp(case))
    raw = agent.run_agent(case["query"], preferences=case.get("preferences"))
    return finalize(raw, preferences=case.get("preferences"))
