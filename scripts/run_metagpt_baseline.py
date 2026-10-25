"""Run pinned MetaGPT in code-generation-only mode for the common task."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
except ModuleNotFoundError:  # Supports ``python scripts/run_metagpt_baseline.py``.
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
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Maximum MetaGPT collaboration rounds; the workflow may finish earlier.",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompt_text = baseline_prompt_for_scenario(args.scenario, args.prompt)
    (output / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    sdk_manifest = stage_sdk(output / "workspace" / "sdk")
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    _load_dotenv(Path.cwd() / ".env")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"{args.api_key_env} is required")
    # MetaGPT validates its repository config before the caller can override it.
    # Supply an isolated home config so the benchmark never edits the checkout
    # and the API key is not persisted in the repository.
    isolated_home = output / "metagpt_home"
    isolated_config = isolated_home / ".metagpt" / "config2.yaml"
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        "llm:\n"
        "  api_type: openai\n"
        f"  model: {args.model}\n"
        f"  base_url: {args.base_url}\n"
        # This file only satisfies MetaGPT's early config validation.  The
        # real key is loaded from .env and injected into the in-memory config
        # below; never persist it in an experiment artifact.
        "  api_key: benchmark-placeholder-not-used\n"
        "  temperature: 0.1\n"
        "  stream: false\n",
        encoding="utf-8",
    )
    os.environ["HOME"] = str(isolated_home)
    os.environ["USERPROFILE"] = str(isolated_home)
    # MetaGPT derives the default working directories of Editor, Terminal,
    # repositories, and storage from this supported environment variable at
    # import time.  Set it before importing MetaGPT modules so every native
    # tool stays inside this run rather than using the pinned checkout's
    # ``workspace`` directory.
    os.environ["METAGPT_PROJECT_ROOT"] = str(output)
    sys.path.insert(0, str(repo))
    from metagpt.config2 import config
    from metagpt.configs.llm_config import LLMConfig, LLMType
    config.llm = LLMConfig(
        api_key=key,
        api_type=LLMType.OPENAI,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        stream=False,
        calc_usage=True,
        max_token=4096,
    )
    config.workspace.path = output / "workspace"
    config.workspace.path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    error = None
    project_path = None
    reply_contents: list[str] = []
    usage_rows: list[dict[str, int]] = []
    framework_workspace = output / "workspace"
    for stale_artifact in (framework_workspace / "api_script.cpp", framework_workspace / "program.nc"):
        stale_artifact.unlink(missing_ok=True)
    try:
        # MetaGPT's current software-company workflow may return a complete
        # source artifact through RoleZero.reply_to_human instead of writing a
        # project directory. Capture that official framework output without
        # changing any role, prompt, planning, or generation behavior.
        from metagpt.roles.di.role_zero import RoleZero

        from metagpt.provider.openai_api import OpenAILLM

        async def responses_acompletion_text(self, messages, stream=False, timeout=None):
            instructions, input_text = messages_to_responses_input(messages)
            result = responses_request(
                api_key=key,
                base_url=args.base_url,
                model=args.model,
                instructions=instructions,
                input_text=input_text,
                reasoning_effort="low",
                timeout=float(timeout or 300),
            )
            row = {
                "prompt_tokens": result.usage.get("input_tokens", 0),
                "completion_tokens": result.usage.get("output_tokens", 0),
                "total_tokens": result.usage.get("total_tokens", 0),
            }
            usage_rows.append(row)
            self._update_costs(row)
            return result.text

        OpenAILLM.acompletion_text = responses_acompletion_text

        original_reply_to_human = RoleZero.reply_to_human

        async def measured_reply_to_human(self, content: str) -> str:
            reply_contents.append(content)
            return await original_reply_to_human(self, content)

        RoleZero.reply_to_human = measured_reply_to_human
        from metagpt.software_company import generate_repo

        # RoleZero.editor is a Pydantic instance field in the pinned MetaGPT
        # checkout, so it cannot be configured through the RoleZero class.
        # Engineer2 initializes its editor from the process working directory;
        # enter the isolated run workspace only for the framework invocation.
        previous_cwd = Path.cwd()
        try:
            os.chdir(config.workspace.path)
            project_path = generate_repo(
                prompt_text,
                investment=3.0,
                n_round=args.rounds,
                code_review=False,
                run_tests=False,
                implement=True,
                project_name="metagpt_focas_benchmark",
            )
        finally:
            os.chdir(previous_cwd)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - started
    usage = {
        "prompt_tokens": sum(x["prompt_tokens"] for x in usage_rows),
        "completion_tokens": sum(x["completion_tokens"] for x in usage_rows),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    usage = usage if usage["total_tokens"] else None
    generated = output / "generated"
    if project_path:
        source_root = Path(project_path)
        if source_root.exists():
            if generated.exists():
                shutil.rmtree(generated)
            shutil.copytree(source_root, generated)
    else:
        cpp_blocks: list[str] = []
        for content in reply_contents:
            cpp_blocks.extend(
                match.group(1).strip()
                for match in re.finditer(r"```(?:cpp|c\+\+|cc|cxx)\s*\n(.*?)```", content, re.IGNORECASE | re.DOTALL)
            )
        if cpp_blocks:
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "focas_coordinate_motion.cpp").write_text(cpp_blocks[-1] + "\n", encoding="utf-8")
    # Engineer2 may use its own Editor tool to write directly into MetaGPT's
    # framework workspace even when generate_repo returns no project_path, and
    # it is free to choose artifact names. Preserve every C/C++ and NC artifact
    # byte-for-byte instead of assuming api_script.cpp/program.nc.
    workspace_suffixes = {".c", ".cc", ".cpp", ".cxx", ".nc"}
    for fallback_workspace_source in framework_workspace.rglob("*"):
        if not fallback_workspace_source.is_file() or fallback_workspace_source.suffix.lower() not in workspace_suffixes:
            continue
        relative = fallback_workspace_source.relative_to(framework_workspace)
        if relative.parts and relative.parts[0].lower() == "sdk":
            continue
        target = generated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback_workspace_source, target)
    generated_sources = []
    if generated.exists():
        generated_sources = [
            str(path.relative_to(generated))
            for path in generated.rglob("*")
            if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}
        ]
    commit = __import__("subprocess").check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "framework": "MetaGPT",
        "repository": "https://github.com/FoundationAgents/MetaGPT.git",
        "commit": commit,
        "python": sys.version,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "sdk": sdk_manifest,
        "temperature": args.temperature,
        "max_rounds": args.rounds,
        "code_only": True,
        "simulator_started": False,
        "started_at": started_at,
        "duration_seconds": duration,
        "generated_root": str(project_path) if project_path else None,
        "usage": usage,
        # Counts originate from the Responses API payload; MetaGPT's cost
        # manager is only used as an internal accumulator.
        "usage_source": "provider_response" if usage is not None else "unavailable",
        "success": error is None and bool(generated_sources),
        "reply_count": len(reply_contents),
        "error": error,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
