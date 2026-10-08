"""
Evaluation cases. Each case = a user request, fixture listings/web results,
the answer a model might plausibly give (including mistakes the guardrails
must catch), and BEHAVIOUR expectations.

The category examples here (lip balm, cleaner, bottle...) are TEST DATA only.
Nothing in the agent reads them; the agent has no category rules.

`expect` keys (all optional):
  product_type, user_requirements      -> also checked in live runs
  kept / removed                       -> product ids present / absent
  verdict / max_verdict                -> overall conclusion
  claim_status {(id, text): status}    -> evidence labels
  requirement_status {(id, req): st}   -> requirement-check labels
  critique_removal_refused [ids]       -> critique tried, code refused
  dropped_type_mismatch, removed_requirement_mismatch
  language_flagged, unaddressed_terms, dimensions_include/exclude, top_first, has_products
"""

from evals.fakes import listing


def answer(product_type: str, materials: list[dict], **extra) -> dict:
    base = {
        "summary": "Found a few options worth considering.",
        "product_type": product_type,
        "user_requirements": [],
        "functional_requirements": [],
        "safety_requirements": [],
        "environmental_dimensions": [],
        "focus_note": None,
        "conclusion": {"verdict": "potential_advantage", "explanation": "Seller statements only."},
        "user_specified_material": None,
        "request_sustainability": "none",
        "caution_note": None,
        "materials": materials,
    }
    base.update(extra)
    return base


def group(name: str, products: list[dict], **extra) -> dict:
    return {"type": name, "matches_request": False, "about": f"{name} option",
            "why_different": f"differs by {name}", "impact": {}, "products": products, **extra}


def product(pid: str, claims=(), checks=()) -> dict:
    return {"id": pid, "why_suggested": "Worth considering.", "trade_off": "Limited information.",
            "claims": list(claims), "requirement_checks": list(checks)}


def claim(text: str, snippet: str, kind: str = "environmental", counter: str | None = None) -> dict:
    return {"claim": text, "kind": kind, "evidence_snippet": snippet, "counter_snippet": counter}


def check(req: str, status: str, snippet: str | None = None) -> dict:
    return {"requirement": req, "status": status, "evidence_snippet": snippet}


CASES = [
    {
        "name": "eco_friendly_lip_balm",
        "query": "eco-friendly lip balm",
        "listings": [
            listing("Beeswax Lip Balm Tube 4.25g", snippet="Paper outer box, the tube is made of plastic"),
            listing("Lip Balm in Aluminium Tin 15g", snippet="Reusable aluminium tin, no plastic used"),
        ],
        "answer": answer(
            "lip balm",
            [group("tube", [product("p1", [claim("Plastic-free packaging", "Paper outer box",
                                                   counter="the tube is made of plastic")])]),
             group("tin", [product("p2", [claim("Comes in a reusable aluminium tin", "Reusable aluminium tin")])])],
            environmental_dimensions=[{"dimension": "container format", "why_relevant": "tin vs tube differ", "priority": "high"}],
            conclusion={"verdict": "clear_advantage", "explanation": "The tin is better."},
        ),
        "expect": {
            "product_type": "lip balm", "kept": ["p1", "p2"], "max_verdict": "potential_advantage",
            # Outer paper box is not plastic-free packaging.
            "claim_status": {("p1", "plastic-free"): "conflicting_evidence",
                             ("p2", "aluminium tin"): "stated_in_listing"},
        },
    },
    {
        "name": "organic_lip_balm",
        "query": "organic lip balm",
        "listings": [
            listing("Strawberry Lip Balm 10g", snippet="Fruity flavour, long lasting"),
            listing("Organic Beeswax Lip Balm", snippet="Made with 100% organic ingredients"),
        ],
        "answer": answer(
            "lip balm",
            [group("flavoured", [product("p1", checks=[check("organic", "unclear")])]),
             group("organic", [product("p2", [claim("Organic ingredients", "100% organic ingredients")],
                                       [check("organic", "met", "100% organic ingredients")])])],
            user_requirements=["organic"],
            summary="The organic balm is the most sustainable choice.",
        ),
        "expect": {
            "product_type": "lip balm", "user_requirements": ["organic"],
            "requirement_status": {("p2", "organic"): "stated_in_listing", ("p1", "organic"): "unverified"},
            "top_first": "p2",  # verified user requirement outranks the agent's own order
            "language_flagged": True, "verdict": "no_clear_winner",
        },
    },
    {
        "name": "eco_friendly_toilet_cleaner",
        "query": "eco-friendly toilet cleaner",
        "listings": [
            listing("Toilet Cleaner Concentrate Refill 1L", snippet="Concentrate makes 5 bottles, plastic pouch"),
            listing("Toilet Cleaner Ready To Use 500ml", snippet="Plastic bottle with angled neck"),
        ],
        "answer": answer(
            "toilet cleaner",
            [group("concentrate refill", [product("p1", [claim("Concentrate makes several bottles", "Concentrate makes 5 bottles")])]),
             group("ready to use", [product("p2")])],
            environmental_dimensions=[
                {"dimension": "refill / concentrate model", "why_relevant": "main difference found", "priority": "high"},
                {"dimension": "packaging", "why_relevant": "all suitable options use plastic, so it does not distinguish them", "priority": "low"},
            ],
        ),
        "critique": {"pass": False, "issues": [{"type": "functional_mismatch", "product_id": "p2",
                                                 "requirement": "plastic-free packaging", "reason": "plastic bottle",
                                                 "action": "remove"}]},
        "expect": {"product_type": "toilet cleaner", "kept": ["p1", "p2"], "critique_removal_refused": ["p2"],
                   "dimensions_include": ["refill"]},
    },
    {
        "name": "plant_based_cleaner",
        "query": "plant-based floor cleaner",
        "listings": [
            listing("Plant Based Floor Cleaner 1L", snippet="Cleans with plant-based surfactants"),
            listing("Lavender Floor Cleaner 1L", snippet="Contains synthetic surfactants and fragrance"),
        ],
        "answer": answer(
            "floor cleaner",
            [group("plant-based", [product("p1", checks=[check("plant-based", "met", "plant-based surfactants")])]),
             group("conventional", [product("p2", checks=[check("plant-based", "contradicted", "synthetic surfactants")])])],
            user_requirements=["plant-based"],
        ),
        "expect": {"product_type": "floor cleaner", "user_requirements": ["plant-based"], "kept": ["p1"],
                   "removed": ["p2"], "removed_requirement_mismatch": 1},
    },
    {
        "name": "sustainable_water_bottle_for_office",
        "query": "sustainable water bottle for office",
        "listings": [
            listing("Steel Water Bottle 1L Leak Proof", snippet="Food grade stainless steel, leak proof lid"),
            listing("Recycled Plastic Water Bottle 750ml", snippet="Made with recycled plastic"),
            # Passes the type filter by title; only the requirement check catches it.
            listing("Recycled Plastic Water Bottle Shaped Vase", snippet="For flowers and decor, not for drinking"),
        ],
        "answer": answer(
            "water bottle",
            [group("steel", [product("p1", checks=[check("safe for drinking water", "met", "Food grade stainless steel")])]),
             group("recycled plastic", [
                 product("p2", [claim("Made with recycled plastic", "Made with recycled plastic")],
                         [check("safe for drinking water", "unclear")]),
                 product("p3", checks=[check("safe for drinking water", "contradicted", "not for drinking")])])],
            functional_requirements=["safe for drinking water", "suits office use"],
        ),
        "expect": {"product_type": "water bottle", "kept": ["p1", "p2"], "removed": ["p3"],
                   "max_verdict": "potential_advantage",
                   "claim_status": {("p2", "recycled"): "stated_in_listing"}},
    },
    {
        "name": "recycled_plastic_water_bottle",
        "query": "recycled plastic water bottle",
        "listings": [
            listing("Aquaone rPET Water Bottle 750ml", snippet="Made from 100% recycled PET"),
            listing("Blendo Water Bottle Recycled Blend", snippet="Contains 30% recycled plastic"),
        ],
        "searches": [{"query": "recycled plastic water bottle", "engine": "shopping"},
                     {"query": "Aquaone rPET bottle recycled content", "engine": "web", "purpose": "evidence"}],
        "web": [{"title": "Aquaone rPET bottles", "snippet": "Aquaone bottles are made from 100% recycled PET, certified by GRS", "link": "https://example.org/aquaone"}],
        "answer": answer(
            "water bottle",
            [group("recycled plastic", [
                product("p1", [claim("Made from 100% recycled PET", "made from 100% recycled PET, certified by GRS")],
                        [check("recycled plastic", "met", "100% recycled PET")]),
                product("p2", [claim("100% recycled plastic", "Contains 30% recycled plastic")],
                        [check("recycled plastic", "met", "30% recycled plastic")])],
                matches_request=True)],
            user_requirements=["recycled plastic"], request_sustainability="eco_leaning",
            user_specified_material="recycled plastic",
            conclusion={"verdict": "clear_advantage", "explanation": "Verified recycled content."},
        ),
        "expect": {
            "product_type": "water bottle", "user_requirements": ["recycled plastic"],
            "claim_status": {("p1", "100% recycled pet"): "supported_by_search",
                             ("p2", "100% recycled"): "unverified"},  # "30%" never supports "100%"
            # One verified claim is not overall superiority.
            "max_verdict": "potential_advantage",
        },
    },
    {
        "name": "organic_shampoo_requirement_dropped",
        "query": "organic shampoo",
        "listings": [listing("Herbal Shampoo 200ml", snippet="With herbal extracts")],
        "answer": answer("shampoo", [group("herbal", [product("p1")])]),  # model forgot "organic"
        "expect": {"product_type": "shampoo", "unaddressed_terms": ["organic"]},
    },
    {
        "name": "recycled_plastic_lunch_box",
        "query": "recycled plastic lunch box",
        "listings": [
            listing("Recycled Plastic Lunch Box 2 Compartment", snippet="Made from recycled plastic, food grade, BPA free"),
            listing("Recycled Plastic Toy Storage Box", snippet="For toys and stationery"),
        ],
        "answer": answer(
            "lunch box",
            [group("recycled plastic", [
                product("p1", [claim("Food grade recycled plastic", "recycled plastic, food grade", kind="requirement")],
                        [check("recycled plastic", "met", "Made from recycled plastic"), check("food safe", "met", "food grade, BPA free")]),
                product("p2", checks=[check("recycled plastic", "met", "Recycled Plastic Toy Storage Box")])],
                matches_request=True)],
            user_requirements=["recycled plastic"], safety_requirements=["food safe"],
            request_sustainability="eco_leaning", user_specified_material="recycled plastic",
        ),
        "expect": {"product_type": "lunch box", "user_requirements": ["recycled plastic"],
                   "kept": ["p1"], "removed": ["p2"], "dropped_type_mismatch": 1,
                   "requirement_status": {("p1", "food safe"): "stated_in_listing"}},
    },
    {
        "name": "sustainable_chair",
        "query": "sustainable chair",
        "listings": [
            listing("Solid Sheesham Wood Chair", snippet="Solid wood frame, repairable joints"),
            listing("Bamboo Chair Natural Finish", snippet="Made of bamboo"),
            listing("Wooden Bench Seat 3ft", snippet="Solid wood bench"),
        ],
        "answer": answer(
            "chair",
            [group("solid wood", [product("p1"), product("p3")]), group("bamboo", [product("p2")])],
            environmental_dimensions=[{"dimension": "longevity / repairability", "why_relevant": "chairs last years", "priority": "high"}],
            conclusion={"verdict": "no_clear_winner", "explanation": "Both have trade-offs; not verified."},
        ),
        "expect": {"product_type": "chair", "kept": ["p1", "p2"], "removed": ["p3"],
                   "dropped_type_mismatch": 1, "verdict": "no_clear_winner"},
    },
    {
        "name": "every_suitable_product_has_plastic_packaging",
        "query": "liquid hand wash refill",
        "listings": [
            listing("Hand Wash Refill Pouch 1.5L", snippet="Refill pouch, plastic"),
            listing("Hand Wash Pump Bottle 250ml", snippet="Plastic pump bottle"),
        ],
        "answer": answer(
            "hand wash",
            [group("refill pouch", [product("p1")]), group("pump bottle", [product("p2")])],
            environmental_dimensions=[{"dimension": "refill format", "why_relevant": "less packaging per wash", "priority": "high"}],
        ),
        "critique": {"pass": False, "issues": [
            {"type": "functional_mismatch", "product_id": "p1", "requirement": "plastic packaging", "reason": "plastic", "action": "remove"},
            {"type": "functional_mismatch", "product_id": "p2", "requirement": "plastic packaging", "reason": "plastic", "action": "remove"}]},
        "expect": {"product_type": "hand wash", "kept": ["p1", "p2"], "critique_removal_refused": ["p1", "p2"]},
    },
    {
        "name": "packaging_is_not_the_differentiator",
        "query": "durable office shirt",
        "listings": [
            listing("Cotton Office Shirt Slim Fit", snippet="100% cotton, double stitched seams"),
            listing("Linen Blend Office Shirt", snippet="Linen cotton blend"),
        ],
        "answer": answer(
            "shirt",
            [group("cotton", [product("p1", [claim("100% cotton fabric", "100% cotton, double stitched", kind="functional")])]),
             group("linen blend", [product("p2")])],
            functional_requirements=["durable", "suits office wear"],
            environmental_dimensions=[{"dimension": "longevity", "why_relevant": "user wants fewer replacements", "priority": "high"}],
        ),
        "expect": {"product_type": "shirt", "kept": ["p1", "p2"], "dimensions_include": ["longevity"],
                   "dimensions_exclude": ["packaging"],
                   "claim_status": {("p1", "cotton"): "stated_in_listing"},
                   # Only functional claims: no environmental advantage established.
                   "verdict": "no_clear_winner"},
    },
    {
        "name": "environmental_claim_cannot_be_verified",
        "query": "biodegradable garbage bags",
        "listings": [listing("Garbage Bags Medium 30 pcs", snippet="Strong and leak proof")],
        "answer": answer(
            "garbage bags",
            [group("bags", [product("p1", [claim("Certified compostable", "certified compostable to EN13432")],
                                     [check("biodegradable", "met", "certified compostable to EN13432")])])],
            user_requirements=["biodegradable"],
            conclusion={"verdict": "clear_advantage", "explanation": "Compostable."},
        ),
        "expect": {"product_type": "garbage bag", "user_requirements": ["biodegradable"],
                   "claim_status": {("p1", "compostable"): "unverified"},
                   "requirement_status": {("p1", "biodegradable"): "unverified"},
                   "verdict": "no_clear_winner"},
    },
    {
        "name": "no_clear_winner_is_a_valid_answer",
        "query": "reusable coffee cup",
        "listings": [
            listing("Stainless Steel Coffee Cup 350ml", snippet="Double wall steel"),
            listing("Glass Coffee Cup with Sleeve 350ml", snippet="Borosilicate glass"),
        ],
        "answer": answer(
            "coffee cup",
            [group("steel", [product("p1")]), group("glass", [product("p2")])],
            conclusion={"verdict": "no_clear_winner", "explanation": "Both reusable; evidence does not separate them."},
        ),
        "expect": {"product_type": "coffee cup", "verdict": "no_clear_winner", "has_products": True,
                   "kept": ["p1", "p2"]},
    },
]
