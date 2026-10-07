from evidence import validate_claims


def _run(product, claim, snippet, web=None):
    materials = [{"type": "x", "products": [{**product, "claims": [{"claim": claim, "evidence_snippet": snippet}]}]}]
    return validate_claims(materials, web or [])[0]["products"][0]["claims"][0]


PRODUCT = {
    "name": "Milton Steel Water Bottle 1L",
    "source": "Amazon.in",
    "listing_text": "Made from food grade stainless steel, BPA free",
}


def test_listing_snippet_counts_as_seller_statement():
    c = _run(PRODUCT, "Food grade stainless steel", "food grade stainless steel")
    assert c["status"] == "stated_in_listing" and c["validated"]


def test_too_short_snippet_is_unverified():
    c = _run(PRODUCT, "Made of 100% stainless steel and BPA free", "Steel")
    assert c["status"] == "unverified"


def test_merchant_name_is_not_evidence():
    c = _run(PRODUCT, "Sold on Amazon.in marketplace", "Amazon.in marketplace seller")
    assert c["status"] == "unverified"


def test_snippet_must_relate_to_claim():
    c = _run(PRODUCT, "Recyclable at end of life", "Milton Steel Water Bottle")
    assert c["status"] == "unverified"


def test_web_evidence_tied_to_product():
    web = [{"results": [{"title": "Milton bottles", "snippet": "Milton uses 18/8 stainless steel in its bottles", "link": "u"}]}]
    c = _run(PRODUCT, "Uses 18/8 stainless steel", "uses 18/8 stainless steel", web)
    assert c["status"] == "supported_by_search" and c["validated"]


def test_web_evidence_not_about_product_is_general():
    web = [{"results": [{"title": "Steel guide", "snippet": "Stainless steel is fully recyclable in most cities", "link": "u"}]}]
    c = _run(PRODUCT, "Steel is fully recyclable", "stainless steel is fully recyclable", web)
    assert c["status"] == "general_evidence" and not c["validated"]
