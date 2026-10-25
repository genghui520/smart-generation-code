from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts import run_chatdev1_baseline, run_gpt_engineer_baseline, run_metagpt_baseline
from scripts.rq1_scenario_tasks import (
    RQ1_BASELINE_ENGINEERING_FRAMEWORK,
    RQ1_SCENARIO_TASKS,
    baseline_prompt_for_scenario,
    baseline_task_for_scenario,
)
from scripts.run_rq1_smpagent import SCENARIOS as SMPAGENT_SCENARIOS


BASELINE_RUNNERS = (
    run_gpt_engineer_baseline,
    run_metagpt_baseline,
    run_chatdev1_baseline,
)
LEAKED_API = re.compile(r"\b(?:cnc|pmc)_[A-Za-z0-9_]+\b", re.IGNORECASE)
LEAKED_ABI_TERMS = (
    "POSELM",
    "ODBACT",
    "ODBPOS",
    "ODBST",
    "IODBTLMNG",
    "WINAPI",
)
REQUIRED_FRAMEWORK_TERMS = ("C++17", "MSVC", "Fwlib32.h", "Fwlib32.lib", "127.0.0.1:8193")


@pytest.mark.parametrize("scenario", sorted(RQ1_SCENARIO_TASKS))
def test_all_rq1_methods_resolve_the_identical_external_prompt(scenario: str) -> None:
    canonical = RQ1_SCENARIO_TASKS[scenario]
    assert SMPAGENT_SCENARIOS[scenario] == canonical
    prompts = []
    for runner in BASELINE_RUNNERS:
        prompt = runner.baseline_prompt_for_scenario(scenario)
        prompts.append(prompt)
        assert prompt == baseline_prompt_for_scenario(scenario)
    assert prompts[0] == prompts[1] == prompts[2]
    assert prompts[0].split("[Common Engineering Framework]", 1)[0] == (
        f"[Scenario Task]\n\n{canonical}\n\n"
    )
    assert prompts[0].split("[Common Engineering Framework]\n\n", 1)[1] == RQ1_BASELINE_ENGINEERING_FRAMEWORK


@pytest.mark.parametrize("scenario", sorted(RQ1_SCENARIO_TASKS))
def test_rq1_external_prompts_do_not_leak_api_or_abi_details(scenario: str) -> None:
    prompt = baseline_prompt_for_scenario(scenario)
    assert LEAKED_API.search(prompt) is None
    assert all(term not in prompt for term in LEAKED_ABI_TERMS)
    assert all(term in prompt for term in REQUIRED_FRAMEWORK_TERMS)


@pytest.mark.parametrize("scenario", sorted(RQ1_SCENARIO_TASKS))
def test_baseline_prompt_has_canonical_scenario_and_shared_framework(scenario: str) -> None:
    prompt = baseline_prompt_for_scenario(scenario)
    scenario_part, framework_part = prompt.split("[Common Engineering Framework]\n\n", 1)
    assert scenario_part == f"[Scenario Task]\n\n{RQ1_SCENARIO_TASKS[scenario]}\n\n"
    assert framework_part == RQ1_BASELINE_ENGINEERING_FRAMEWORK


def test_smpagent_prompt_does_not_receive_baseline_framework() -> None:
    assert all(RQ1_BASELINE_ENGINEERING_FRAMEWORK not in task for task in SMPAGENT_SCENARIOS.values())


def test_baseline_prompt_override_must_be_byte_for_byte_canonical() -> None:
    # Keep this test independent of pytest's OS temp directory, which may be
    # inaccessible in managed Windows environments.
    test_dir = Path(__file__).with_name(".rq1_prompt_test_inputs")
    test_dir.mkdir(exist_ok=True)
    matching = test_dir / "matching.txt"
    divergent = test_dir / "divergent.txt"
    try:
        matching.write_text(RQ1_SCENARIO_TASKS["coordinate_motion"], encoding="utf-8")
        assert baseline_task_for_scenario("coordinate_motion", matching) == RQ1_SCENARIO_TASKS["coordinate_motion"]

        divergent.write_text("cnc_rdposition", encoding="utf-8")
        with pytest.raises(ValueError, match="canonical SMPAgent RQ1 task text"):
            baseline_task_for_scenario("coordinate_motion", divergent)
    finally:
        matching.unlink(missing_ok=True)
        divergent.unlink(missing_ok=True)
        test_dir.rmdir()
