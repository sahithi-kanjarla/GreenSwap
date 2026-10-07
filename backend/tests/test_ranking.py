from ranking import extract_price, parse_number, rank_products, finalize


def test_price_parsing():
    assert extract_price("₹1,29,999") == 129999.0
    assert extract_price("Rs. 1,299.50") == 1299.5
    assert extract_price(599) == 599.0
    assert extract_price(None) is None


def test_number_parsing():
    assert parse_number("1.2K") == 1200
    assert parse_number("4.5") == 4.5


def test_budget_keeps_unknown_prices_and_sorts():
    products = [
        {"name": "a", "price": "₹900", "rating": 4.0, "reviews": 10},
        {"name": "b", "price": "₹300", "rating": 4.8, "reviews": 500},
        {"name": "c", "price": None},
    ]
    ranked = rank_products(products, max_price=500, sort_by="price_low_high")
    assert [p["name"] for p in ranked] == ["b", "c"]
    assert [p["name"] for p in rank_products(products, sort_by="reviews")][0] == "b"


def test_finalize_shape():
    raw = {"request_sustainability": "none", "materials": [
        {"type": "steel", "matches_request": False, "products": [{"name": "a", "price": "₹100", "material_type": "steel"}]}
    ]}
    view = finalize(raw)
    assert view["top_picks"][0]["name"] == "a"
    assert view["cheapest_found"]["name"] == "a"
    assert not view["no_suitable_alternative"]
