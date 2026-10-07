from agent import _augment_shopping_query, _best_product_link, _add_shopping_signals


def test_high_impact_material_not_reinjected():
    q = _augment_shopping_query("bamboo chair", "virgin plastic chair", ["virgin plastic"])
    assert q == "bamboo chair"


def test_use_case_context_preserved():
    assert _augment_shopping_query("steel water bottle", "water bottle for office") == "steel water bottle office"


def test_filler_words_do_not_leak():
    q = _augment_shopping_query(
        "cotton office shirt",
        "I need an everyday office shirt that will last a long time and doesn't need frequent replacement",
    )
    assert q == "cotton office shirt everyday"


def test_plural_counts_as_present():
    assert _augment_shopping_query("glass bottles", "glass bottle") == "glass bottles"


def test_link_policy():
    shop = "https://www.google.co.in/search?q=x&tbm=shop"
    gpage = "https://www.google.com/shopping/product/123"
    assert _best_product_link({"merchant_link": "https://www.amazon.in/dp/B0", "link": shop})[1] == "merchant_product"
    assert _best_product_link({"product_link": gpage, "link": shop}) == (shop, "shopping_search", "Opens Google Shopping for this product and seller")
    assert _best_product_link({"product_link": gpage})[1] == "google_product_page"
    assert _best_product_link({})[1] == "unavailable"


def test_shopping_signals():
    catalog = {
        "p1": {"price": "₹500", "extracted_price": 500, "old_price": "₹800", "reviews": 1000},
        "p2": {"price": "₹300", "extracted_price": 300, "reviews": 5},
    }
    _add_shopping_signals(catalog)
    assert "Offer" in catalog["p1"]["shopping_signals"]["badges"]
    assert "Popular" in catalog["p1"]["shopping_signals"]["badges"]
    assert catalog["p2"]["shopping_signals"]["lowest_price_in_run"]


def test_critique_cannot_remove_for_high_impact_material_or_invented_requirement():
    from agent import _functional_removals
    critique = {"issues": [
        {"type": "functional_mismatch", "product_id": "p1", "requirement": "virgin plastic", "action": "remove"},
        {"type": "functional_mismatch", "product_id": "p2", "requirement": "weather resistance", "action": "remove"},
        {"type": "functional_mismatch", "product_id": "p3", "requirement": "outdoor use", "action": "remove"},
    ]}
    parsed = {"request_sustainability": "high_impact", "user_specified_material": "virgin plastic",
              "functional_requirements": ["seating for outdoor use"]}
    accepted, refused = _functional_removals(critique, parsed, "virgin plastic chair")
    assert accepted == {"p3"}
    assert {r["product_id"] for r in refused} == {"p1", "p2"}
