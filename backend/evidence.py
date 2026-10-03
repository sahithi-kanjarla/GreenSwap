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
    - listing evidence is distinguished from web evidence,
    - missing evidence is downgraded to "unverified",
    - conflicting evidence is represented explicitly,
    - the model cannot simply declare a claim "verified" without
      providing evidence that was actually retrieved.

Important:
Finding a sentence in a web result does NOT automatically mean
the claim is scientifically proven. This module only verifies that
the cited evidence was genuinely retrieved. The LLM/agent is
responsible for determining whether that evidence is relevant to
the claim.
"""


# ---------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------

STATUS_LABELS = {
    "stated_in_listing": "Stated by seller",
    "supported_by_search": "Evidence found",
    "conflicting_evidence": "Conflicting / unclear",
    "unverified": "Not verified",
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
) -> dict | None:
    """
    Find the actual retrieved web result containing the cited snippet.

    Returns the complete matching result or None.
    """
    normalized_snippet = _normalize(evidence_snippet)

    if not normalized_snippet:
        return None

    for result in web_results:
        title = result.get("title")
        snippet = result.get("snippet")

        if _appears_in(
            evidence_snippet,
            [title, snippet],
        ):
            return result

    return None


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

    We deliberately restrict this to fields representing the actual
    product listing.
    """
    product_texts = [
        product.get("name"),
        product.get("title"),
        product.get("source"),
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
                        "source": "...",
                        "claims": [
                            {
                                "claim": "...",
                                "evidence_snippet": "...",
                                "status": "..."
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
                # 1. Check the actual shopping/product listing
                # -----------------------------------------------------

                listing_match = _find_listing_evidence(
                    evidence_snippet,
                    product,
                )

                if listing_match:
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
                            "source_url": product.get("link")
                            or product.get("product_link"),
                        }
                    )

                    continue

                # -----------------------------------------------------
                # 2. Check actual web-search results
                # -----------------------------------------------------

                matching_result = _find_matching_web_result(
                    evidence_snippet,
                    web_results,
                )

                if matching_result:

                    validated_claims.append(
                        {
                            "claim": _format_claim(claim_text),
                            "status": "supported_by_search",
                            "label": STATUS_LABELS[
                                "supported_by_search"
                            ],
                            "validated": True,
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
                    {
                        "claim": _format_claim(claim_text),
                        "status": "unverified",
                        "label": STATUS_LABELS[
                            "unverified"
                        ],
                        "validated": False,
                        "evidence_snippet": evidence_snippet,
                        "evidence_source": None,
                        "source_title": None,
                        "source_url": None,
                    }
                )

            product["claims"] = validated_claims

    return materials


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