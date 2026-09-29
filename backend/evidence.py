"""
Pure-code claim validation. No LLM call, no SerpApi call.

The agent may attach a "claims" list to any product, each claim paired
with an evidence_snippet the model says it saw in the retrieved data
(a listing title, or a web search result). This module is the
ENFORCEMENT layer: it checks that snippet actually appears in text we
genuinely retrieved, and downgrades anything it can't confirm to
"unverified" — regardless of what status the model claimed for itself.

This is what makes "never invent facts" a checked rule instead of a
prompt request.
"""


def _normalize(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _appears_in(snippet: str | None, texts: list[str | None]) -> bool:
    s = _normalize(snippet)
    if not s:
        return False
    return any(s in _normalize(t) for t in texts if t)


def validate_claims(materials: list[dict], web_findings: list[dict]) -> list[dict]:
    """
    Mutates and returns `materials`. Each product's "claims" list is
    replaced with validated versions:
      status: "stated_in_listing" | "supported_by_search" | "unverified"
      validated: True only for the first two
    """
    web_texts: list[str | None] = []
    for w in web_findings:
        for r in w.get("results", []):
            web_texts.append(r.get("title"))
            web_texts.append(r.get("snippet"))

    for m in materials:
        for p in m["products"]:
            product_texts = [p.get("name"), p.get("source")]
            validated = []
            for c in p.get("claims", []) or []:
                snippet = c.get("evidence_snippet")
                claim_text = c.get("claim")
                if not claim_text:
                    continue
                if _appears_in(snippet, product_texts):
                    status, ok = "stated_in_listing", True
                elif _appears_in(snippet, web_texts):
                    status, ok = "supported_by_search", True
                else:
                    status, ok = "unverified", False
                validated.append(
                    {
                        "claim": claim_text,
                        "status": status,
                        "validated": ok,
                        "evidence_snippet": snippet,
                    }
                )
            p["claims"] = validated

    return materials