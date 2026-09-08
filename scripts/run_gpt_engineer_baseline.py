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
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://fast.smartaipro.cn/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    generated = output / "generated"
    output.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt.read_text(encoding="utf-8")
    (output / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    sys.path.insert(0, str(repo))
    _load_dotenv(Path.cwd() / ".env")
    os.environ["OPENAI_API_KEY"] = os.environ["SMARTAIPRO_API_KEY"]
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
            response = super().backoff_inference(messages)
            usage = getattr(response, "usage_metadata", None) or {}
            usage_rows.append(dict(usage))
            return response

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
        estimated = ai.token_usage_log.log()
        estimated_usage = {
            "prompt_tokens": sum(x.in_step_prompt_tokens for x in estimated),
            "completion_tokens": sum(x.in_step_completion_tokens for x in estimated),
            "total_tokens": sum(x.in_step_total_tokens for x in estimated),
        }
    except Exception as exc:
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
        "temperature": args.temperature,
        "code_only": True,
        "simulator_started": False,
        "started_at": started_at,
        "duration_seconds": duration,
        "generated_files": sorted(files),
        "usage": actual_usage if actual_usage["total_tokens"] else estimated_usage,
        "usage_source": "provider_response" if actual_usage["total_tokens"] else "gpt_engineer_tiktoken_estimate",
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
