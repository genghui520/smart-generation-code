"""Run pinned ChatDev 1.0 for the shared code-generation-only benchmark.

The upstream release only exposes historical OpenAI model enum names.  This
runner preserves ChatDev's roles, phases, and chat chain while adapting its
OpenAI-compatible transport to the same model used by the other methods.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
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
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _usage_from_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    prompt = sum(map(int, re.findall(r"^prompt_tokens:\s*(\d+)", text, re.MULTILINE)))
    completion = sum(map(int, re.findall(r"^completion_tokens:\s*(\d+)", text, re.MULTILINE)))
    total_rows = list(map(int, re.findall(r"^total_tokens:\s*(\d+)", text, re.MULTILINE)))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": sum(total_rows) if total_rows else prompt + completion,
    }


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
    prompt_text = args.prompt.read_text(encoding="utf-8")
    (output / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    _load_dotenv(Path.cwd() / ".env")
    key = os.environ.get("SMARTAIPRO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("SMARTAIPRO_API_KEY or OPENAI_API_KEY is required")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["BASE_URL"] = args.base_url
    sys.path.insert(0, str(repo))
    # ChatDev 1.0's optional ECL package uses historical top-level imports.
    # Register those two modules without changing the pinned checkout.
    for legacy_name in ("utils", "embedding"):
        spec = importlib.util.spec_from_file_location(legacy_name, repo / "ecl" / f"{legacy_name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load ChatDev ECL module: {legacy_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[legacy_name] = module
        spec.loader.exec_module(module)

    # Import the original framework, then replace only its provider transport.
    import openai
    import camel.model_backend as model_backend
    from camel.typing import ModelType

    usage_rows: list[dict[str, int]] = []

    def compatible_run(self, *call_args, **kwargs):
        config = dict(self.model_config_dict)
        config["temperature"] = args.temperature
        config.pop("max_tokens", None)
        response = openai.OpenAI(api_key=key, base_url=args.base_url).chat.completions.create(
            *call_args, **kwargs, model=args.model, **config
        )
        usage = response.usage
        row = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        usage_rows.append(row)
        from chatdev.utils import log_visualize
        log_visualize(
            "**[OpenAI_Usage_Info Receive]**\n"
            f"prompt_tokens: {row['prompt_tokens']}\n"
            f"completion_tokens: {row['completion_tokens']}\n"
            f"total_tokens: {row['total_tokens']}\n"
        )
        # openai>=1.35 adds ``refusal`` to ChatCompletionMessage.  ChatDev
        # 1.0 expands the message into its older dataclass, which does not
        # accept that field, so expose the exact legacy four-field mapping.
        for choice in response.choices:
            message = choice.message
            choice.message = {
                "role": message.role,
                "content": message.content or "",
                "function_call": message.function_call,
                "tool_calls": message.tool_calls,
            }
        return response

    model_backend.OpenAIModel.run = compatible_run
    # The old wrapper retries all exceptions, including deterministic schema
    # errors, hiding their cause behind RetryError.  Keep one attempt per
    # message for reproducibility and preserve the concrete traceback.
    import camel.agents.chat_agent as chat_agent_module
    if hasattr(chat_agent_module.ChatAgent.step, "__wrapped__"):
        chat_agent_module.ChatAgent.step = chat_agent_module.ChatAgent.step.__wrapped__
    from chatdev.chat_chain import ChatChain

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    error = None
    postprocess_warning = None
    project_path: Path | None = None
    chain = None
    try:
        chain = ChatChain(
            config_path=str(repo / "CompanyConfig" / "Default" / "ChatChainConfig.json"),
            config_phase_path=str(repo / "CompanyConfig" / "Default" / "PhaseConfig.json"),
            config_role_path=str(repo / "CompanyConfig" / "Default" / "RoleConfig.json"),
            task_prompt=prompt_text,
            project_name="chatdev1_focas_benchmark",
            org_name="SMPAgent",
            model_type=ModelType.GPT_4O,
            code_path="",
        )
        logging.basicConfig(
            filename=chain.log_filepath,
            level=logging.INFO,
            format="[%(asctime)s %(levelname)s] %(message)s",
            datefmt="%Y-%d-%m %H:%M:%S",
            encoding="utf-8",
        )
        chain.pre_processing()
        project_path = Path(chain.chat_env.env_dict["directory"])
        chain.make_recruitment()
        chain.execute_chain()
        try:
            chain.post_processing()
        except Exception as exc:
            postprocess_warning = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    generated = output / "generated"
    if project_path and project_path.exists():
        if generated.exists():
            shutil.rmtree(generated)
        shutil.copytree(project_path, generated)
    usage = {
        "prompt_tokens": sum(x["prompt_tokens"] for x in usage_rows),
        "completion_tokens": sum(x["completion_tokens"] for x in usage_rows),
        "total_tokens": sum(x["total_tokens"] for x in usage_rows),
    }
    if not usage["total_tokens"] and chain is not None:
        usage = _usage_from_log(Path(chain.log_filepath))
    generated_sources = []
    if generated.exists():
        generated_sources = [
            str(path.relative_to(generated))
            for path in generated.rglob("*")
            if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}
        ]
    metadata = {
        "framework": "ChatDev 1.0",
        "repository": "https://github.com/OpenBMB/ChatDev.git",
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "model": args.model,
        "transport_adapter": "OpenAI-compatible runtime adapter; ChatDev roles/phases unchanged",
        "base_url": args.base_url,
        "temperature": args.temperature,
        "code_only": True,
        "simulator_started": False,
        "started_at": started_at,
        "duration_seconds": time.perf_counter() - started,
        "generated_root": str(project_path) if project_path else None,
        "usage": usage,
        "usage_source": "provider_response",
        "success": error is None and bool(generated_sources),
        "generated_files": generated_sources,
        "error": error,
        "postprocess_warning": postprocess_warning,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
