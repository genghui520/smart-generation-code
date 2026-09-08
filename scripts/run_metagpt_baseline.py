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
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(args.prompt.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    _load_dotenv(Path.cwd() / ".env")
    key = os.environ.get("SMARTAIPRO_API_KEY")
    if not key:
        raise SystemExit("SMARTAIPRO_API_KEY is required")
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
        f"  api_key: {key}\n"
        "  temperature: 0.1\n"
        "  stream: false\n",
        encoding="utf-8",
    )
    os.environ["HOME"] = str(isolated_home)
    os.environ["USERPROFILE"] = str(isolated_home)
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
    fallback_workspace_source = repo / "workspace" / "focas_coordinate_motion.cpp"
    fallback_workspace_source.unlink(missing_ok=True)
    try:
        # MetaGPT's current software-company workflow may return a complete
        # source artifact through RoleZero.reply_to_human instead of writing a
        # project directory. Capture that official framework output without
        # changing any role, prompt, planning, or generation behavior.
        from metagpt.roles.di.role_zero import RoleZero

        original_reply_to_human = RoleZero.reply_to_human

        async def measured_reply_to_human(self, content: str) -> str:
            reply_contents.append(content)
            return await original_reply_to_human(self, content)

        RoleZero.reply_to_human = measured_reply_to_human
        from metagpt.software_company import generate_repo
        project_path = generate_repo(
            args.prompt.read_text(encoding="utf-8"),
            investment=3.0,
            n_round=5,
            code_review=False,
            run_tests=False,
            implement=True,
            project_name="metagpt_focas_benchmark",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - started
    usage = {
        "prompt_tokens": int(getattr(config, "cost_manager", None).total_prompt_tokens) if getattr(config, "cost_manager", None) else 0,
        "completion_tokens": int(getattr(config, "cost_manager", None).total_completion_tokens) if getattr(config, "cost_manager", None) else 0,
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
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
    # framework workspace even when generate_repo returns no project_path.
    # Copy that byte-for-byte framework artifact into the run directory.
    if fallback_workspace_source.exists():
        generated.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback_workspace_source, generated / fallback_workspace_source.name)
    commit = __import__("subprocess").check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "framework": "MetaGPT",
        "repository": "https://github.com/FoundationAgents/MetaGPT.git",
        "commit": commit,
        "python": sys.version,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "code_only": True,
        "simulator_started": False,
        "started_at": started_at,
        "duration_seconds": duration,
        "generated_root": str(project_path) if project_path else None,
        "usage": usage,
        "usage_source": "metagpt_cost_manager",
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
