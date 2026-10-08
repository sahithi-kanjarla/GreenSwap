"""
Behavioural checks on a finished GreenSwap result. They test what the agent
DID (kept, removed, labelled, concluded), never how it phrased it.

invariant_failures(result, case) holds for every run, offline or live.
expectation_failures(result, case) checks case-specific offline expectations.
Both return a list of human-readable failures (empty = pass).
"""

from research import VERDICTS, absolute_language, answer_texts

KNOWN_CLAIM_STATUSES = {
    "stated_in_listing", "supported_by_search", "general_evidence",
    "conflicting_evidence", "unverified",
}
_RANK = {v: i for i, v in enumerate(("no_clear_winner", "potential_advantage", "clear_advantage"))}


def _products(result: dict) -> list[dict]:
    return [p for m in result.get("materials", []) for p in m.get("products", [])]


def _ids(result: dict) -> set[str]:
    return {p["id"] for p in _products(result)}


def invariant_failures(result: dict, case: dict) -> list[str]:
    fails = []
    expect = case.get("expect", {})
    conclusion = result.get("conclusion") or {}

    # Product type understood.
    if expect.get("product_type"):
        got = (result.get("product_type") or "").lower()
        if expect["product_type"].lower() not in got and got not in expect["product_type"].lower():
            fails.append(f"product_type {got!r} != {expect['product_type']!r}")

    # Explicit user requirements preserved.
    reqs = " | ".join(result.get("user_requirements") or []).lower()
    for word in expect.get("user_requirements", []):
        if word.lower() not in reqs:
            fails.append(f"user requirement {word!r} not preserved (got {result.get('user_requirements')})")

    # Every product carries a check for every explicit user requirement.
    if result.get("user_requirements"):
        for p in _products(result):
            kinds = [c["kind"] for c in p.get("requirement_checks", [])]
            if kinds.count("user") != len(result["user_requirements"]):
                fails.append(f"{p['id']} lacks requirement checks")

    # Evidence statuses well-formed; "Evidence found" always has a source.
    for p in _products(result):
        for c in p.get("claims", []):
            if c.get("status") not in KNOWN_CLAIM_STATUSES:
                fails.append(f"{p['id']} claim has unknown status {c.get('status')}")
            if c.get("status") == "supported_by_search" and not c.get("source_url"):
                fails.append(f"{p['id']} 'Evidence found' claim without a source")

    # No overall conclusion stronger than the evidence.
    verdict = conclusion.get("verdict")
    if verdict not in VERDICTS:
        fails.append(f"invalid verdict {verdict!r}")
    if verdict == "clear_advantage":
        backed = any(
            sum(c.get("status") == "supported_by_search" for c in p.get("claims", [])) >= 2
            for p in _products(result)
        )
        if not backed:
            fails.append("clear_advantage without two product-specific evidence-backed claims")

    # Verdict language never survives without an evidence-backed verdict.
    flags = absolute_language(answer_texts(result.get("summary"), conclusion, result.get("materials", [])))
    if flags and verdict != "clear_advantage" and verdict != "no_clear_winner":
        fails.append(f"absolute language {flags} with verdict {verdict}")
    return fails


def expectation_failures(result: dict, case: dict) -> list[str]:
    fails = []
    expect = case.get("expect", {})
    ids = _ids(result)
    meta = result.get("meta", {})

    for pid in expect.get("kept", []):
        if pid not in ids:
            fails.append(f"{pid} should have been kept")
    for pid in expect.get("removed", []):
        if pid in ids:
            fails.append(f"{pid} should have been removed")

    if "verdict" in expect and (result.get("conclusion") or {}).get("verdict") != expect["verdict"]:
        fails.append(f"verdict {(result.get('conclusion') or {}).get('verdict')} != {expect['verdict']}")
    if "max_verdict" in expect:
        verdict = (result.get("conclusion") or {}).get("verdict")
        if _RANK.get(verdict, 99) > _RANK[expect["max_verdict"]]:
            fails.append(f"verdict {verdict} stronger than allowed {expect['max_verdict']}")

    by_id = {p["id"]: p for p in [p for m in result.get("materials", []) for p in m["products"]]}
    for (pid, fragment), status in expect.get("claim_status", {}).items():
        claims = [c for c in by_id.get(pid, {}).get("claims", []) if fragment.lower() in c["claim"].lower()]
        if not claims:
            fails.append(f"{pid}: no claim containing {fragment!r}")
        elif claims[0]["status"] != status:
            fails.append(f"{pid} claim {fragment!r}: {claims[0]['status']} != {status}")

    for (pid, fragment), status in expect.get("requirement_status", {}).items():
        checks = [c for c in by_id.get(pid, {}).get("requirement_checks", []) if fragment.lower() in c["requirement"].lower()]
        if not checks:
            fails.append(f"{pid}: no requirement check for {fragment!r}")
        elif checks[0]["status"] != status:
            fails.append(f"{pid} requirement {fragment!r}: {checks[0]['status']} != {status}")

    refused = {i.get("product_id") for i in (meta.get("critique") or {}).get("refused_removals", [])}
    for pid in expect.get("critique_removal_refused", []):
        if pid not in refused:
            fails.append(f"critique removal of {pid} should have been refused")

    for key in ("dropped_type_mismatch", "removed_requirement_mismatch"):
        if key in expect and meta.get(key) != expect[key]:
            fails.append(f"{key} = {meta.get(key)} != {expect[key]}")

    if "language_flagged" in expect and bool(result.get("language_flags")) != expect["language_flagged"]:
        fails.append(f"language_flags = {result.get('language_flags')}")

    for term in expect.get("unaddressed_terms", []):
        if term not in (result.get("unaddressed_request_terms") or []):
            fails.append(f"dropped request term {term!r} was not surfaced")

    dims = [d["dimension"].lower() for d in result.get("environmental_dimensions") or []]
    for word in expect.get("dimensions_include", []):
        if not any(word in d for d in dims):
            fails.append(f"dimension containing {word!r} missing (got {dims})")
    for word in expect.get("dimensions_exclude", []):
        if any(word in d for d in dims):
            fails.append(f"dimension containing {word!r} should not be selected (got {dims})")

    if "top_first" in expect:
        top = result.get("top_picks") or []
        if not top or top[0]["id"] != expect["top_first"]:
            fails.append(f"top pick {top[0]['id'] if top else None} != {expect['top_first']}")

    if expect.get("has_products") and not ids:
        fails.append("expected products to be shown")
    return fails
