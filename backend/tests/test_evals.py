"""
Behaviour evaluation harness.

Offline (default): every case runs through the real agent pipeline with a
scripted model and fixture listings. Live (--run-live-evals): the same cases
run against the real LLMs and SerpApi; only the generic invariants plus the
expected product type / explicit requirements are checked, because a live
model's exact choices legitimately vary.
"""

import pytest

from evals.cases import CASES
from evals.checks import expectation_failures, invariant_failures
from evals.fakes import run_offline


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_offline(case, monkeypatch):
    result = run_offline(case, monkeypatch)
    failures = invariant_failures(result, case) + expectation_failures(result, case)
    assert not failures, "\n".join(failures)


@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_live(case):
    from dotenv import load_dotenv
    load_dotenv()
    from agent import run_agent
    from ranking import finalize

    result = finalize(run_agent(case["query"]))
    assert "error" not in result.get("meta", {}), result["meta"].get("error")
    live_case = {"expect": {k: v for k, v in case.get("expect", {}).items()
                            if k in ("product_type", "user_requirements")}}
    failures = invariant_failures(result, live_case)
    assert not failures, "\n".join(failures)
