"""Run the pinned GPT-Engineer checkout for the code-only benchmark."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.responses_transport import messages_to_responses_input, responses_request
except ModuleNotFoundError:
    from responses_transport import messages_to_responses_input, responses_request

try:
    from scripts.rq1_scenario_tasks import RQ1_SCENARIO_TASKS, baseline_prompt_for_scenario
except ModuleNotFoundError:  # Supports ``python scripts/run_gpt_engineer_baseline.py``.
    from rq1_scenario_tasks import RQ1_SCENARIO_TASKS, baseline_prompt_for_scenario
try:
    from scripts.baseline_sdk import stage_sdk
except ModuleNotFoundError:
    from baseline_sdk import stage_sdk

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scenario", choices=RQ1_SCENARIO_TASKS, required=True)
    parser.add_argument("--prompt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://fast.smartaipro.cn/v1")
    parser.add_argument("--api-key-env", default="MY_LLM_API_KEY2")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    generated = output / "generated"
    output.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    sdk_manifest = stage_sdk(output / "sdk")
    prompt_text = baseline_prompt_for_scenario(args.scenario, args.prompt)
    (output / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    # Load the pinned checkout bundled with this project, rather than relying
    # on a separately installed package or the project root itself.
    framework_repo = repo / "baselines" / "gpt-engineer"
    if not framework_repo.is_dir():
        raise SystemExit(f"Pinned GPT-Engineer checkout not found: {framework_repo}")
    sys.path.insert(0, str(framework_repo))
    _load_dotenv(Path.cwd() / ".env")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"{args.api_key_env} is required")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ["OPENAI_BASE_URL"] = args.base_url

    from gpt_engineer.core.ai import AI
    from gpt_engineer.core.default.disk_memory import DiskMemory
    from gpt_engineer.core.default.paths import PREPROMPTS_PATH
    from gpt_engineer.core.default.steps import gen_code
    from gpt_engineer.core.preprompts_holder import PrepromptsHolder
    from gpt_engineer.core.prompt import Prompt

    usage_rows: list[dict] = []

    class MeasuredAI(AI):
        def backoff_inference(self, messages):
            from langchain_core.messages import AIMessage

            instructions, input_text = messages_to_responses_input(messages)
            result = responses_request(
                api_key=key,
                base_url=args.base_url,
                model=args.model,
                instructions=instructions,
                input_text=input_text,
                reasoning_effort="low",
            )
            usage_rows.append(
                {
                    "input_tokens": result.usage.get("input_tokens", 0),
                    "output_tokens": result.usage.get("output_tokens", 0),
                    "total_tokens": result.usage.get("total_tokens", 0),
                }
            )
            return AIMessage(content=result.text, response_metadata={"usage": result.usage})

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    error = None
    files = {}
    try:
        ai = MeasuredAI(
            model_name=args.model,
            temperature=args.temperature,
            streaming=False,
        )
        memory = DiskMemory(output / "gpt_engineer_memory")
        files = dict(
            gen_code(
                ai,
                Prompt(prompt_text),
                memory,
                PrepromptsHolder(PREPROMPTS_PATH),
            )
        )
        for name, content in files.items():
            target = generated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        # Do not use GPT-Engineer's local tiktoken estimate for this fair
        # comparison.  Token counts must come only from provider responses.
        estimated_usage = None
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"
        estimated_usage = None

    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    duration = time.perf_counter() - started
    actual_usage = {
        "prompt_tokens": sum(int(x.get("input_tokens", 0)) for x in usage_rows),
        "completion_tokens": sum(int(x.get("output_tokens", 0)) for x in usage_rows),
        "total_tokens": sum(int(x.get("total_tokens", 0)) for x in usage_rows),
    }
    metadata = {
        "framework": "GPT-Engineer",
        "repository": "https://github.com/AntonOsika/gpt-engineer.git",
        "commit": commit,
        "python": sys.version,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "sdk": sdk_manifest,
        "temperature": args.temperature,
        "code_only": True,
        "simulator_started": False,
        "started_at": started_at,
        "duration_seconds": duration,
        "generated_files": sorted(files),
        "usage": actual_usage if actual_usage["total_tokens"] else None,
        "usage_source": "provider_response" if actual_usage["total_tokens"] else "unavailable",
        "success": error is None and bool(files),
        "error": error,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
