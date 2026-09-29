import os
import json
import hashlib
from pathlib import Path
from urllib.parse import quote as requote
from dotenv import load_dotenv

from serpapi import GoogleSearch

load_dotenv()

SERPAPI_KEY = os.environ["SERPAPI_KEY"]

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

USE_CACHE = os.environ.get("SERP_CACHE", "1") != "0"

MAX_SHOPPING_RESULTS = 8
MAX_WEB_RESULTS = 5


def _cache_file(engine: str, query: str) -> Path:
    key = hashlib.sha1(
        f"{engine}|{query.strip().lower()}".encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{key}.json"


def serp_search(
    query: str,
    engine: str = "shopping"
) -> tuple[list[dict], bool]:

    path = _cache_file(engine, query)

    # Return cached results if available
    if USE_CACHE and path.exists():
        return (
            json.loads(path.read_text(encoding="utf-8")),
            True
        )

    # Make live SerpApi request
    results = _fetch(query, engine)

    # Cache non-empty results
    if USE_CACHE and results:
        path.write_text(
            json.dumps(results, ensure_ascii=False),
            encoding="utf-8"
        )

    return results, False


def _fetch(query: str, engine: str) -> list[dict]:

    params = {
        "api_key": SERPAPI_KEY,
        "q": query,
        "gl": "in",
        "google_domain": "google.co.in",
        "hl": "en",
        "engine": (
            "google_shopping"
            if engine == "shopping"
            else "google"
        ),
    }

    data = GoogleSearch(params).get_dict()

    # Handle SerpApi errors
    error = data.get("error")

    if error and "hasn't returned any results" not in error:
        raise RuntimeError(error)

    # -------------------------
    # SHOPPING RESULTS
    # -------------------------

    if engine == "shopping":

        out = []

        for item in data.get("shopping_results", [])[
            :MAX_SHOPPING_RESULTS
        ]:

            title = item.get("title", "")
            source = item.get("source", "")

            # Temporary Google Shopping search link.
            # We will improve this later to use a direct merchant link
            # if the raw SerpApi response provides one.
            link = (
                f"https://www.google.com/search?"
                f"q={requote(title + ' ' + source)}"
                f"&tbm=shop"
                if title
                else None
            )

            out.append(
                {
                    "title": title,
                    "price": item.get("price"),
                    "source": source,
                    "link": link,
                    "image": item.get("thumbnail"),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                }
            )

        return out

    # -------------------------
    # WEB RESULTS
    # -------------------------

    return [
        {
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link"),
        }
        for item in data.get("organic_results", [])[
            :MAX_WEB_RESULTS
        ]
    ]


def debug_raw_shopping(query: str) -> None:

    params = {
        "api_key": SERPAPI_KEY,
        "q": query,
        "gl": "in",
        "google_domain": "google.co.in",
        "hl": "en",
        "engine": "google_shopping",
    }

    data = GoogleSearch(params).get_dict()

    print(
        json.dumps(
            data.get("shopping_results", [])[:2],
            indent=2
        )
    )

if __name__ == "__main__":
    results, from_cache = serp_search("mug", "shopping")

    print("From cache:", from_cache)
    print(json.dumps(results, indent=2, ensure_ascii=False))