import os
import json
import hashlib
from pathlib import Path
from urllib.parse import quote as requote

from dotenv import load_dotenv
from serpapi import GoogleSearch


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SERPAPI_KEY = os.environ["SERPAPI_KEY"]

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

USE_CACHE = os.environ.get("SERP_CACHE", "1") != "0"


# ============================================================
# LIMITS
# ============================================================

# Raised from 8 -> 20: this is the SAME SerpApi call either way (SerpApi
# already fetched this many results server-side), so keeping more of what
# we already paid for is free breadth. The earlier cap of 8 was silently
# throttling "how many products can the user choose among" even when
# 15-20 good, genuinely distinct listings existed in the same response.
MAX_SHOPPING_RESULTS = 20
MAX_WEB_RESULTS = 5

# Immersive Product API can return up to ~13 stores when
# more_stores=true, depending on product availability.
MAX_IMMERSIVE_STORES = 13


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_file(engine: str, query: str) -> Path:
    """
    Create a deterministic cache filename for a request.

    The actual query/token is hashed so we don't create
    extremely long filenames.
    """
    key = hashlib.sha1(
        f"{engine}|{query.strip().lower()}".encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{key}.json"


# ============================================================
# GENERAL SERP SEARCH
# ============================================================

def serp_search(
    query: str,
    engine: str = "shopping"
) -> tuple[list[dict], bool]:
    """
    Search using SerpApi.

    Supported engines:
        - shopping
        - web

    Returns:
        (results, from_cache)
    """

    path = _cache_file(engine, query)

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if USE_CACHE and path.exists():
        return (
            json.loads(path.read_text(encoding="utf-8")),
            True
        )

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    results = _fetch(query, engine)

    # --------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------

    if USE_CACHE and results:
        path.write_text(
            json.dumps(
                results,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    return results, False


# ============================================================
# FETCH SHOPPING / WEB
# ============================================================

def _fetch(
    query: str,
    engine: str
) -> list[dict]:

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

    # --------------------------------------------------------
    # SERPAPI ERROR
    # --------------------------------------------------------

    error = data.get("error")

    if error and "hasn't returned any results" not in error:
        raise RuntimeError(error)

    # ========================================================
    # GOOGLE SHOPPING
    # ========================================================

    if engine == "shopping":

        out = []

        for item in data.get(
            "shopping_results",
            []
        )[:MAX_SHOPPING_RESULTS]:

            title = item.get("title", "")
            source = item.get("source", "")

            # ------------------------------------------------
            # SAFE PRODUCT-SPECIFIC SHOPPING LINK
            # ------------------------------------------------
            #
            # SerpApi's `product_link` can be session-bound and, in
            # some responses, can point back to the original Shopping
            # search rather than reliably opening this exact listing.
            #
            # GreenSwap therefore uses its own stable Google Shopping
            # search URL built from THIS product title + merchant.
            # This prevents a product card from accidentally opening
            # the query that found a different product.
            # ------------------------------------------------

            shopping_query = f'"{title}" "{source}"'.strip()

            fallback_link = (
                f"https://www.google.co.in/search?"
                f"q={requote(shopping_query)}"
                f"&tbm=shop"
                if title
                else None
            )

            # Keep SerpApi's raw product link for debugging/future use,
            # but DO NOT expose it as GreenSwap's primary product link.
            product_link = item.get("product_link")

            # ------------------------------------------------
            # Immersive Product token
            #
            # THIS is what we need later for price comparison.
            # ------------------------------------------------

            immersive_product_page_token = item.get(
                "immersive_product_page_token"
            )

            serpapi_immersive_product_api = item.get(
                "serpapi_immersive_product_api"
            )

            # ------------------------------------------------
            # NORMALIZED PRODUCT
            # ------------------------------------------------

            out.append({

                # Basic product information
                "title": title,

                # Pricing
                "price": item.get("price"),
                "extracted_price": item.get(
                    "extracted_price"
                ),

                "old_price": item.get("old_price"),
                "extracted_old_price": item.get(
                    "extracted_old_price"
                ),

                # Merchant
                "source": source,

                # Links
                #
                # Keep product_link when available.
                # Otherwise use our Google Shopping fallback.
                "product_link": product_link,
                "raw_product_link": product_link,

                # IMPORTANT: always use the stable product-specific
                # Shopping search URL as the user-facing link.
                # The raw SerpApi product_link is retained separately.
                "link": fallback_link,

                "link_type": "google_shopping_search",

                "link_note": (
                    "Opens Google Shopping for this product and merchant. "
                    "The direct SerpApi product link is intentionally not used "
                    "because it can be session-dependent."
                ),

                # Image
                "image": item.get("thumbnail"),

                # Rating / reviews
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),

                # Additional Shopping information
                "delivery": item.get("delivery"),
                "snippet": item.get("snippet"),

                "extensions": item.get(
                    "extensions",
                    []
                ),

                "tag": item.get("tag"),
                "badge": item.get("badge"),

                "multiple_sources": item.get(
                    "multiple_sources"
                ),

                # Product identity
                #
                # These are important for the lazy
                # price-comparison step.
                "product_id": item.get(
                    "product_id"
                ),

                "immersive_product_page_token": (
                    immersive_product_page_token
                ),

                "serpapi_immersive_product_api": (
                    serpapi_immersive_product_api
                ),
            })

        return out

    # ========================================================
    # GOOGLE WEB SEARCH
    # ========================================================

    return [
        {
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link"),
        }
        for item in data.get(
            "organic_results",
            []
        )[:MAX_WEB_RESULTS]
    ]


# ============================================================
# IMMERSIVE PRODUCT API
# ============================================================

def get_product_offers(
    page_token: str,
    more_stores: bool = True
) -> tuple[dict, bool]:
    """
    Fetch merchant/store offers for one specific Google
    Shopping product using its Immersive Product token.

    This is intentionally separate from serp_search().

    Why?
        The initial GreenSwap agent search should remain
        focused on discovery/research.

        Price comparison happens lazily when the user asks
        to compare buying options for a particular product.

    Args:
        page_token:
            immersive_product_page_token returned by
            Google Shopping.

        more_stores:
            If True, ask SerpApi for more merchant offers.

    Returns:
        (raw_response, from_cache)
    """

    if not page_token:
        raise ValueError(
            "A valid immersive_product_page_token is required."
        )

    # --------------------------------------------------------
    # Cache key
    # --------------------------------------------------------

    cache_query = (
        f"{page_token}|"
        f"more_stores={str(more_stores).lower()}"
    )

    path = _cache_file(
        "google_immersive_product",
        cache_query
    )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if USE_CACHE and path.exists():
        return (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            ),
            True
        )

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google_immersive_product",
        "page_token": page_token,
        "more_stores": "true" if more_stores else "false",
    }

    data = GoogleSearch(params).get_dict()

    # --------------------------------------------------------
    # SERPAPI ERROR
    # --------------------------------------------------------

    error = data.get("error")

    if error:
        raise RuntimeError(error)

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if USE_CACHE:
        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    return data, False


# ============================================================
# NORMALIZE IMMERSIVE PRODUCT OFFERS
# ============================================================

def normalize_product_offers(
    data: dict
) -> list[dict]:
    """
    Convert the raw Immersive Product API response into
    a simple structure that the FastAPI backend/frontend
    can consume easily.

    SerpApi's Immersive Product API exposes store offers
    under:

        product_results -> stores

    Each store can contain:
        - name
        - logo
        - link
        - title
        - price
        - extracted_price
        - original_price
        - extracted_original_price
        - shipping
        - extracted shipping
        - total
        - extracted_total
        - discount
        - rating
        - reviews
        - details_and_offers
        - tag
        - payment_methods
    """

    product_results = data.get(
        "product_results",
        {}
    )

    stores = product_results.get(
        "stores",
        []
    )

    offers = []

    for store in stores[:MAX_IMMERSIVE_STORES]:

        offers.append({

            # ------------------------------------------------
            # Merchant
            # ------------------------------------------------

            "merchant": store.get("name"),

            "logo": store.get("logo"),

            # ------------------------------------------------
            # Product
            # ------------------------------------------------

            "title": store.get("title"),

            "link": store.get("link"),

            # ------------------------------------------------
            # Price
            # ------------------------------------------------

            "price": store.get("price"),

            "extracted_price": store.get(
                "extracted_price"
            ),

            "original_price": store.get(
                "original_price"
            ),

            "extracted_original_price": store.get(
                "extracted_original_price"
            ),

            # ------------------------------------------------
            # Shipping
            # ------------------------------------------------

            "shipping": store.get("shipping"),

            "shipping_extracted": store.get(
                "shipping_extracted"
            ),

            # ------------------------------------------------
            # Total
            # ------------------------------------------------

            "total": store.get("total"),

            "extracted_total": store.get(
                "extracted_total"
            ),

            # ------------------------------------------------
            # Discount / tags
            # ------------------------------------------------

            "discount": store.get("discount"),

            "tag": store.get("tag"),

            # ------------------------------------------------
            # Product metadata
            # ------------------------------------------------

            "rating": store.get("rating"),

            "reviews": store.get("reviews"),

            "payment_methods": store.get(
                "payment_methods"
            ),

            "details_and_offers": store.get(
                "details_and_offers",
                []
            ),

            "coupon": store.get("coupon"),

            # ------------------------------------------------
            # Installment information
            # ------------------------------------------------

            "monthly_payment_duration": store.get(
                "monthly_payment_duration"
            ),

            "installments_description": store.get(
                "installments_description"
            ),

            "down_payment": store.get(
                "down_payment"
            ),

            # ------------------------------------------------
            # Tax
            # ------------------------------------------------

            "estimated_tax": store.get(
                "estimated_tax"
            ),

            "extracted_estimated_tax": store.get(
                "extracted_estimated_tax"
            ),
        })

    return offers


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def get_normalized_product_offers(
    page_token: str,
    more_stores: bool = True
) -> tuple[list[dict], bool]:
    """
    Convenience wrapper.

    Fetches the Immersive Product response and immediately
    converts it into normalized merchant offers.

    Returns:
        (offers, from_cache)
    """

    data, from_cache = get_product_offers(
        page_token=page_token,
        more_stores=more_stores
    )

    offers = normalize_product_offers(data)

    return offers, from_cache


# ============================================================
# DEBUG RAW SHOPPING
# ============================================================

def debug_raw_shopping(
    query: str
) -> None:
    """
    Debug helper.

    Prints the first two raw Google Shopping results so
    we can inspect exactly what SerpApi is returning.
    """

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
            data.get(
                "shopping_results",
                []
            )[:2],
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# DEBUG IMMERSIVE PRODUCT
# ============================================================

def debug_product_offers(
    page_token: str
) -> None:
    """
    Debug helper for the price-comparison API.
    """

    data, from_cache = get_product_offers(
        page_token=page_token,
        more_stores=True
    )

    print(
        "From cache:",
        from_cache
    )

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TEST 1: NORMAL SHOPPING SEARCH
    # --------------------------------------------------------

    results, from_cache = serp_search(
        "mug",
        "shopping"
    )

    print(
        "From cache:",
        from_cache
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # If you want to test price comparison:
    #
    # 1. Look at the printed shopping result.
    # 2. Copy its:
    #
    #       immersive_product_page_token
    #
    # 3. Then uncomment the following:
    #
    # debug_product_offers(
    #     "PASTE_TOKEN_HERE"
    # )
    #
    # --------------------------------------------------------