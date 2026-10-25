"""Import/startup smoke tests for the pinned baseline frameworks.

Each framework is initialized only through the point immediately before its
first model call. No inference or generation method is invoked.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def smoke_gpt_engineer() -> str:
    repo = ROOT / "baselines" / "gpt-engineer"
    sys.path.insert(0, str(repo))
    os.environ.setdefault("OPENAI_API_KEY", "smoke-no-call")
    os.environ.setdefault("OPENAI_API_BASE", "https://openrelayx.cc/v1")
    from gpt_engineer.core.ai import AI
    from gpt_engineer.core.default.disk_memory import DiskMemory
    from gpt_engineer.core.default.paths import PREPROMPTS_PATH
    from gpt_engineer.core.default.steps import gen_code
    from gpt_engineer.core.preprompts_holder import PrepromptsHolder
    from gpt_engineer.core.prompt import Prompt

    AI(model_name="gpt-5.5", temperature=0.1, streaming=False)
    DiskMemory(ROOT / "tmp" / "smoke_gpt_engineer")
    PrepromptsHolder(PREPROMPTS_PATH)
    Prompt("smoke only")
    assert callable(gen_code)
    return "GPT-Engineer: import and pre-call initialization OK"


def smoke_metagpt() -> str:
    repo = ROOT / "baselines" / "MetaGPT"
    sys.path.insert(0, str(repo))
    os.environ.setdefault("OPENAI_API_KEY", "smoke-no-call")
    isolated_home = ROOT / "tmp" / "smoke_metagpt_home"
    isolated_config = isolated_home / ".metagpt" / "config2.yaml"
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        "llm:\n"
        "  api_type: openai\n"
        "  model: gpt-5.5\n"
        "  base_url: https://openrelayx.cc/v1\n"
        "  api_key: smoke-no-call\n"
        "  temperature: 0.1\n"
        "  stream: false\n",
        encoding="utf-8",
    )
    os.environ["HOME"] = str(isolated_home)
    os.environ["USERPROFILE"] = str(isolated_home)
    from metagpt.config2 import config
    from metagpt.configs.llm_config import LLMConfig, LLMType
    from metagpt.roles.di.role_zero import RoleZero
    from metagpt.software_company import generate_repo

    config.llm = LLMConfig(
        api_key="smoke-no-call",
        api_type=LLMType.OPENAI,
        base_url="https://openrelayx.cc/v1",
        model="gpt-5.5",
        temperature=0.1,
        stream=False,
        calc_usage=True,
        max_token=4096,
    )
    assert callable(RoleZero.reply_to_human)
    assert callable(generate_repo)
    return "MetaGPT: import and pre-call initialization OK"


def smoke_chatdev() -> str:
    repo = ROOT / "baselines" / "ChatDev1"
    sys.path.insert(0, str(repo))
    os.environ.setdefault("OPENAI_API_KEY", "smoke-no-call")
    os.environ.setdefault("BASE_URL", "https://openrelayx.cc/v1")
    # ChatDev 1.0's ECL modules use historical top-level imports.
    for legacy_name in ("utils", "embedding"):
        spec = importlib.util.spec_from_file_location(legacy_name, repo / "ecl" / f"{legacy_name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {legacy_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[legacy_name] = module
        spec.loader.exec_module(module)
    import camel.model_backend as model_backend
    from camel.typing import ModelType
    from chatdev.chat_chain import ChatChain

    assert hasattr(model_backend, "OpenAIModel")
    assert ModelType.GPT_4O is not None
    chain = ChatChain(
        config_path=str(repo / "CompanyConfig" / "Default" / "ChatChainConfig.json"),
        config_phase_path=str(repo / "CompanyConfig" / "Default" / "PhaseConfig.json"),
        config_role_path=str(repo / "CompanyConfig" / "Default" / "RoleConfig.json"),
        task_prompt="smoke only",
        project_name="chatdev1_smoke",
        org_name="SMPAgent",
        model_type=ModelType.GPT_4O,
        code_path="",
    )
    assert chain is not None
    return "ChatDev 1.0: import and pre-call initialization OK"


def main() -> None:
    for function in (smoke_gpt_engineer, smoke_metagpt, smoke_chatdev):
        print(function())


if __name__ == "__main__":
    main()
