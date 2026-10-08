"""
Structured research state and its enforcement. Pure code: no LLM, no SerpApi.

The model decides WHAT matters for a request: the user's explicit
requirements, the functional/safety needs, which characteristics could
distinguish suitable alternatives (environmental_dimensions), which
approaches to compare, and what overall conclusion the evidence allows.

This module only enforces the boundaries around those decisions. It
contains NO product-category knowledge and no list of dimensions:

  - normalize_research_state: keeps the model's structured metadata
    well-formed (types, sizes, allowed enum values) without changing meaning.
  - enforce_user_requirements: FUNCTION FIRST. A product the model itself
    marked as contradicting an explicit/functional/safety requirement is
    removed; a requirement it marked "met" must be backed by retrieved
    text, otherwise it is shown as "Not verified".
  - guard_conclusion: an overall environmental conclusion can only be as
    strong as the evidence. Without product-specific evidence there is no
    "clear advantage"; with nothing validated it is "no clear winner".
    The guard only ever makes a conclusion MORE conservative.
  - absolute_language: flags verdict phrases ("the most sustainable ...")
    that the evidence does not back.
"""

import re

from evidence import validate_claims, STATUS_LABELS

PRIORITIES = ("high", "medium", "low")
REQUIREMENT_KINDS = ("user", "functional", "safety")

VERDICTS = ("clear_advantage", "potential_advantage", "no_clear_winner")
VERDICT_LABELS = {
    "clear_advantage": "Evidence-backed advantage",
    "potential_advantage": "Potential advantage (not independently verified)",
    "no_clear_winner": "No clear winner based on available evidence",
}
# Lower index = more conservative.
_CONSERVATISM = {"no_clear_winner": 0, "potential_advantage": 1, "clear_advantage": 2}

MAX_DIMENSIONS = 5
MAX_REQUIREMENTS = 8


def _clean_str(value, limit: int = 200) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text[:limit] or None


def _str_list(values, limit: int = MAX_REQUIREMENTS) -> list[str]:
    out = []
    for v in values or []:
        text = _clean_str(v, 120)
        if text and text.lower() not in {o.lower() for o in out}:
            out.append(text)
    return out[:limit]


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

def normalize_research_state(parsed: dict) -> dict:
    """The model's research metadata, well-formed. Meaning is never changed."""
    dimensions = []
    for d in parsed.get("environmental_dimensions") or []:
        if not isinstance(d, dict):
            continue
        name = _clean_str(d.get("dimension"), 60)
        if not name:
            continue
        priority = str(d.get("priority") or "medium").lower()
        dimensions.append({
            "dimension": name,
            "why_relevant": _clean_str(d.get("why_relevant"), 200),
            "priority": priority if priority in PRIORITIES else "medium",
        })
    dimensions.sort(key=lambda d: PRIORITIES.index(d["priority"]))

    approaches = []
    for m in parsed.get("materials") or []:
        if isinstance(m, dict) and m.get("type"):
            approaches.append({
                "approach": _clean_str(m.get("type"), 60),
                "why_different": _clean_str(m.get("why_different") or m.get("about"), 200),
            })

    return {
        "product_type": _clean_str(parsed.get("product_type"), 40),
        "user_requirements": _str_list(parsed.get("user_requirements")),
        "functional_requirements": _str_list(parsed.get("functional_requirements")),
        "safety_requirements": _str_list(parsed.get("safety_requirements")),
        "environmental_dimensions": dimensions[:MAX_DIMENSIONS],
        "alternative_approaches": approaches,
        "focus_note": _clean_str(parsed.get("focus_note"), 300),
    }


def _all_requirements(state: dict) -> list[tuple[str, str]]:
    return (
        [("user", r) for r in state["user_requirements"]]
        + [("functional", r) for r in state["functional_requirements"]]
        + [("safety", r) for r in state["safety_requirements"]]
    )


def _key(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


# ---------------------------------------------------------------------
# Function first: requirement enforcement
# ---------------------------------------------------------------------

def enforce_user_requirements(
    materials: list[dict], state: dict, web_findings: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Attach a checked `requirement_checks` list to every product and remove
    products the model itself marked as contradicting a stated requirement.

    Model input per product (optional):
        "requirement_checks": [{"requirement", "status": met|contradicted|unclear,
                                "evidence_snippet"}]
    Output per check: requirement, kind, status, label, evidence_snippet.
      - met + snippet found in the listing      -> "Stated by seller"
      - met + product-specific web evidence     -> "Evidence found"
      - met + snippet not retrieved/too vague   -> downgraded to "Not verified"
      - unclear or not assessed                 -> "Not verified"
      - contradicted                            -> product removed
    Only requirements present in the research state count; the model cannot
    remove a product for a requirement nobody stated.

    Returns (materials, removed) where removed lists {id, name, requirement}.
    """
    requirements = _all_requirements(state)
    by_key = {_key(text): (kind, text) for kind, text in requirements}
    removed: list[dict] = []
    kept_materials = []

    for material in materials:
        kept = []
        for product in material.get("products", []):
            raw_checks = {
                _key(c.get("requirement")): c
                for c in (product.pop("raw_requirement_checks", None) or [])
                if isinstance(c, dict) and _key(c.get("requirement")) in by_key
            }

            contradicted = [
                by_key[k][1] for k, c in raw_checks.items()
                if str(c.get("status")).lower() == "contradicted"
            ]
            if contradicted:
                removed.append({"id": product.get("id"), "name": product.get("name"),
                                "requirement": contradicted[0]})
                continue

            checks = []
            for k, (kind, text) in by_key.items():
                c = raw_checks.get(k) or {}
                checks.append(_check(product, kind, text, c, web_findings))
            product["requirement_checks"] = checks
            kept.append(product)

        if kept:
            kept_materials.append({**material, "products": kept})
    return kept_materials, removed


def _check(product: dict, kind: str, requirement: str, raw: dict, web_findings: list[dict]) -> dict:
    status = str(raw.get("status") or "unclear").lower()
    snippet = _clean_str(raw.get("evidence_snippet"), 200) or ""
    result = {
        "requirement": requirement,
        "kind": kind,
        "status": "unverified",
        "label": STATUS_LABELS["unverified"],
        "evidence_snippet": snippet or None,
    }
    if status != "met" or not snippet:
        return result

    # Reuse the claim validator: the requirement is the claim, the snippet its evidence.
    probe = [{"products": [{**product, "claims": [{"claim": requirement, "evidence_snippet": snippet}]}]}]
    validated = validate_claims(probe, web_findings)[0]["products"][0]["claims"][0]
    if validated["status"] in ("stated_in_listing", "supported_by_search"):
        result.update(status=validated["status"], label=validated["label"])
    return result


def requirements_satisfied(product: dict) -> bool:
    """True when every explicit USER requirement is backed by retrieved text."""
    user_checks = [c for c in product.get("requirement_checks", []) if c["kind"] == "user"]
    return bool(user_checks) and all(c["status"] != "unverified" for c in user_checks)


# ---------------------------------------------------------------------
# Conclusion guard
# ---------------------------------------------------------------------

def _evidence_strength(materials: list[dict]) -> str:
    """
    The strongest verdict the validated environmental claims allow.

    One verified claim is not overall superiority: "clear_advantage" needs a
    product with at least two environmental claims backed by product-specific
    evidence and none in conflict. Seller statements or a single verified
    claim allow at most "potential_advantage".
    """
    strongest = "no_clear_winner"
    for m in materials:
        for p in m.get("products", []):
            env = [c for c in p.get("claims", []) if c.get("kind", "environmental") == "environmental"]
            statuses = [c.get("status") for c in env]
            if statuses.count("supported_by_search") >= 2 and "conflicting_evidence" not in statuses:
                return "clear_advantage"
            if {"supported_by_search", "stated_in_listing"} & set(statuses):
                strongest = "potential_advantage"
    return strongest


def guard_conclusion(conclusion, materials: list[dict], override: str | None = None) -> dict:
    """
    Cap the model's overall conclusion at what the validated evidence allows,
    and apply a critique override only when it is more conservative.
    """
    raw = conclusion if isinstance(conclusion, dict) else {}
    claimed = str(raw.get("verdict") or "no_clear_winner").lower()
    if claimed not in VERDICTS:
        claimed = "no_clear_winner"

    allowed = _evidence_strength(materials) if materials else "no_clear_winner"
    verdict = min(claimed, allowed, key=_CONSERVATISM.get)
    reason = None if verdict == claimed else "the validated evidence does not support a stronger conclusion"
    if override in VERDICTS and _CONSERVATISM[override] < _CONSERVATISM[verdict]:
        verdict, reason = override, "the self-critique judged the evidence insufficient"

    out = {
        "verdict": verdict,
        "label": VERDICT_LABELS[verdict],
        "explanation": _clean_str(raw.get("explanation"), 400),
    }
    if reason:
        out["downgraded_from"] = claimed
        out["downgrade_reason"] = reason
    return out


# ---------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------

_ABSOLUTE = re.compile(
    r"\b(?:the\s+)?(?:most|least)\s+(?:sustainable|eco[- ]?friendly|environmentally\s+friendly|green|ethical)\b"
    r"|\bgreenest\b"
    r"|\bbest\s+(?:choice\s+)?for\s+the\s+(?:environment|planet)\b"
    r"|\b(?:is|are)\s+(?:100%\s+|fully\s+|completely\s+)?(?:sustainable|eco[- ]?friendly|environmentally\s+friendly)\b"
    r"|\benvironmentally\s+superior\b"
    r"|\b(?:zero|no)\s+(?:environmental\s+)?impact\b",
    re.IGNORECASE,
)


def absolute_language(texts: list[str | None]) -> list[str]:
    """Verdict phrases GreenSwap should not state as fact."""
    found = []
    for text in texts:
        for match in _ABSOLUTE.finditer(text or ""):
            phrase = match.group(0).strip()
            if phrase.lower() not in {f.lower() for f in found}:
                found.append(phrase)
    return found


def answer_texts(summary: str | None, conclusion: dict, materials: list[dict]) -> list[str | None]:
    texts = [summary, conclusion.get("explanation")]
    for m in materials:
        texts.append(m.get("about"))
        for p in m.get("products", []):
            texts += [p.get("why_suggested"), p.get("trade_off")]
    return texts
