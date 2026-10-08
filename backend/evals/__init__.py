"""
GreenSwap behaviour evaluations.

Each case is a user request plus BEHAVIOUR expectations (never exact wording).

    pytest                     runs every case OFFLINE: scripted model answers and
                               fixture listings go through the real agent pipeline
                               (ids, type filter, requirement enforcement, evidence
                               validation, critique guardrails, conclusion guard,
                               ranking). No API calls, no quota.
    pytest --run-live-evals    additionally runs every case against the real
                               LLMs and SerpApi and checks the generic invariants.
                               Uses quota: up to 5 SerpApi searches per case.
"""
