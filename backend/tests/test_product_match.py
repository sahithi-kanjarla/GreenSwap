from product_match import title_matches, filter_by_product_type


def test_whole_words_only():
    assert not title_matches("cup", "Wooden Cupboard Organizer")
    assert title_matches("cup", "Steel Cups Set of 4")


def test_plurals_and_compounds():
    assert title_matches("knife", "Bamboo Knives Set")
    assert title_matches("lunch box", "Steel Lunchbox 3 Tier")
    assert title_matches("food container", "Glass Food Storage Container 1L")
    assert title_matches("chair", "Recycled Plastic Chairs")


def test_group_with_no_match_is_left_alone():
    materials = [
        {"type": "powder", "products": [{"name": "Citric Acid Cleaning Powder"}]},
        {"type": "liquid", "products": [{"name": "Herbal Toilet Cleaner"}, {"name": "Floor Mop"}]},
    ]
    kept, dropped = filter_by_product_type(materials, "toilet cleaner")
    assert dropped == 1
    assert [p["name"] for p in kept[0]["products"]] == ["Citric Acid Cleaning Powder"]
    assert [p["name"] for p in kept[1]["products"]] == ["Herbal Toilet Cleaner"]
