"""
Pure-code claim validation.

No LLM call.
No SerpApi call.

The agent is responsible for:
    1. deciding which claims matter,
    2. searching for evidence,
    3. attaching the evidence it found to each claim.

This module is the ENFORCEMENT layer.

It checks that:
    - the cited evidence actually exists in data we retrieved,
    - the snippet is specific enough to mean something (a lone word
      like "steel" cannot back "100% stainless steel, BPA-free"),
    - the snippet shares at least one content word with the claim,
    - listing evidence is distinguished from web evidence,
    - web evidence is tied to THIS product (brand/title words appear in
      the result); otherwise it is labelled general, not product-specific,
    - missing evidence is downgraded to "unverified",
    - the model cannot simply declare a claim "verified" without
      providing evidence that was actually retrieved.

Important:
Finding a sentence in a web result does NOT automatically mean
the claim is scientifically proven. This module only verifies that
the cited evidence was genuinely retrieved. The LLM/agent is
responsible for determining whether that evidence is relevant to
the claim.
"""

import re


# ---------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------

STATUS_LABELS = {
    "stated_in_listing": "Stated by seller",
    "supported_by_search": "Evidence found",
    "general_evidence": "General info (not product-specific)",
    "unverified": "Not verified",
}

# A snippet must be at least this specific to count as evidence
# (either condition is enough).
MIN_SNIPPET_WORDS = 3
MIN_SNIPPET_CHARS = 15

# Words that say nothing about a specific product; ignored when tying a
# web result to a product or a snippet to a claim.
_GENERIC_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "made", "pack",
    "set", "piece", "pcs", "new", "best", "premium", "quality", "online",
    "india", "buy", "price", "free", "eco", "friendly", "natural",
}


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _normalize(text: str | None) -> str:
    """
    Normalize text so small differences in whitespace/casing do not
    prevent evidence matching.
    """
    return " ".join((text or "").lower().split())


def _appears_in(
    snippet: str | None,
    texts: list[str | None],
) -> bool:
    """
    Return True when the complete normalized evidence snippet appears
    inside one of the retrieved texts.
    """
    normalized_snippet = _normalize(snippet)

    if not normalized_snippet:
        return False

    for text in texts:
        if normalized_snippet in _normalize(text):
            return True

    return False


def _content_words(text: str | None) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) >= 3 and w not in _GENERIC_WORDS
    }


def _snippet_problem(snippet: str, claim: str) -> str | None:
    """Return why a snippet cannot count as evidence, or None if it can."""
    if not snippet:
        return "no evidence snippet was cited"
    if len(snippet.split()) < MIN_SNIPPET_WORDS and len(snippet) < MIN_SNIPPET_CHARS:
        return "cited snippet is too short to be meaningful"
    # Compare on 4-char prefixes so "recycled"/"recyclable" style
    # variations still count as the same topic.
    snippet_roots = {w[:4] for w in _content_words(snippet)}
    claim_roots = {w[:4] for w in _content_words(claim)}
    if claim_roots and not (snippet_roots & claim_roots):
        return "cited snippet does not mention what the claim is about"
    return None


def _is_about_product(result: dict, product: dict) -> bool:
    """
    True when a web result plausibly talks about THIS product: its brand
    (first title word) or at least two distinctive title words appear.
    """
    name = product.get("name") or product.get("title") or ""
    text = f"{result.get('title') or ''} {result.get('snippet') or ''}"
    text_words = set(re.findall(r"[a-z0-9]+", text.lower()))

    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w.isalpha()]
    brand = words[0] if words else None
    if brand and len(brand) >= 3 and brand not in _GENERIC_WORDS and brand in text_words:
        return True

    return len(_content_words(name) & _content_words(text)) >= 2


# ---------------------------------------------------------------------
# Web-result helpers
# ---------------------------------------------------------------------

def _collect_web_results(
    web_findings: list[dict],
) -> list[dict]:
    """
    Flatten web findings into a simple list of retrieved search results.

    Expected structure:

        [
            {
                "query": "...",
                "results": [
                    {
                        "title": "...",
                        "snippet": "...",
                        "link": "..."
                    }
                ]
            }
        ]

    The function is intentionally tolerant of missing fields.
    """
    results = []

    for finding in web_findings or []:
        for result in finding.get("results", []) or []:
            results.append(
                {
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "link": (
                        result.get("link")
                        or result.get("url")
                        or result.get("source_url")
                    ),
                    "source": result.get("source"),
                }
            )

    return results


def _find_matching_web_result(
    evidence_snippet: str | None,
    web_results: list[dict],
    product: dict,
) -> dict | None:
    """
    Find the actual retrieved web result containing the cited snippet.

    Prefers a result that is about this product over a general one.
    Returns the complete matching result or None.
    """
    if not _normalize(evidence_snippet):
        return None

    matches = [
        result for result in web_results
        if _appears_in(evidence_snippet, [result.get("title"), result.get("snippet")])
    ]
    for result in matches:
        if _is_about_product(result, product):
            return result
    return matches[0] if matches else None


# ---------------------------------------------------------------------
# Source matching
# ---------------------------------------------------------------------

def _find_listing_evidence(
    evidence_snippet: str | None,
    product: dict,
) -> bool:
    """
    Check whether the evidence snippet appears in information that came
    directly from the shopping/product listing.

    The merchant name ("source") is deliberately NOT listing evidence:
    a snippet like "Amazon" says nothing about the product.
    """
    product_texts = [
        product.get("name"),
        product.get("title"),
        product.get("description"),
        product.get("listing_text"),
    ]

    return _appears_in(
        evidence_snippet,
        product_texts,
    )


# ---------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------

def validate_claims(
    materials: list[dict],
    web_findings: list[dict],
) -> list[dict]:
    """
    Validate claims attached to products.

    Parameters
    ----------
    materials:
        Current GreenSwap product/alternative structure.

        Expected shape:

        [
            {
                "material_type": "...",
                "products": [
                    {
                        "name": "...",
                        "listing_text": "...",
                        "claims": [
                            {
                                "claim": "...",
                                "evidence_snippet": "...",
                            }
                        ]
                    }
                ]
            }
        ]

    web_findings:
        Actual web search results retrieved by the agent.

    Returns
    -------
    list[dict]
        The same structure with claims replaced by validated claims.

    Validation rule
    ---------------
    The model's claimed status is NEVER trusted.

    The final status is determined from evidence that actually exists
    in retrieved listing/web data.
    """

    web_results = _collect_web_results(web_findings)

    for material in materials:

        for product in material.get("products", []) or []:

            validated_claims = []

            for claim_data in product.get("claims", []) or []:

                claim_text = (
                    claim_data.get("claim") or ""
                ).strip()

                evidence_snippet = (
                    claim_data.get("evidence_snippet") or ""
                ).strip()

                if not claim_text:
                    continue

                # -----------------------------------------------------
                # 0. Is the snippet specific enough to be evidence?
                # -----------------------------------------------------

                problem = _snippet_problem(evidence_snippet, claim_text)

                if problem:
                    validated_claims.append(
                        _unverified(claim_text, evidence_snippet, problem)
                    )
                    continue

                # -----------------------------------------------------
                # 1. Check the actual shopping/product listing
                # -----------------------------------------------------

                if _find_listing_evidence(evidence_snippet, product):
                    validated_claims.append(
                        {
                            "claim": _format_claim(claim_text),
                            "status": "stated_in_listing",
                            "label": STATUS_LABELS[
                                "stated_in_listing"
                            ],
                            "validated": True,
                            "evidence_snippet": evidence_snippet,
                            "evidence_source": "product_listing",
                            "source_title": product.get("name")
                            or product.get("title"),
                            "source_url": product.get("link"),
                        }
                    )

                    continue

                # -----------------------------------------------------
                # 2. Check actual web-search results
                # -----------------------------------------------------

                matching_result = _find_matching_web_result(
                    evidence_snippet,
                    web_results,
                    product,
                )

                if matching_result:

                    status = (
                        "supported_by_search"
                        if _is_about_product(matching_result, product)
                        else "general_evidence"
                    )

                    validated_claims.append(
                        {
                            "claim": _format_claim(claim_text),
                            "status": status,
                            "label": STATUS_LABELS[status],
                            # Only product-specific evidence counts as
                            # validated for ranking purposes.
                            "validated": status == "supported_by_search",
                            "evidence_snippet": evidence_snippet,
                            "evidence_source": "web_search",
                            "source_title": matching_result.get(
                                "title"
                            ),
                            "source_url": matching_result.get(
                                "link"
                            ),
                        }
                    )

                    continue

                # -----------------------------------------------------
                # 3. Evidence could not be found
                # -----------------------------------------------------

                validated_claims.append(
                    _unverified(
                        claim_text,
                        evidence_snippet,
                        "cited snippet was not found in any retrieved listing or web result",
                    )
                )

            product["claims"] = validated_claims

    return materials


def _unverified(claim_text: str, evidence_snippet: str, reason: str) -> dict:
    return {
        "claim": _format_claim(claim_text),
        "status": "unverified",
        "label": STATUS_LABELS["unverified"],
        "validated": False,
        "evidence_snippet": evidence_snippet,
        "evidence_source": None,
        "source_title": None,
        "source_url": None,
        "reason": reason,
    }


# ---------------------------------------------------------------------
# Claim formatting
# ---------------------------------------------------------------------

def _format_claim(claim_text: str) -> str:
    """
    Make claim presentation consistent without changing its meaning.
    """
    claim_text = claim_text.strip()

    if not claim_text:
        return claim_text

    return claim_text[0].upper() + claim_text[1:]
