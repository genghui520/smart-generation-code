"""Launch a full SMPAgent coordinate-motion run against real NCGuide."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--task",
        "生成坐标运动流量：执行包含 G00/G01 的 NC 程序，执行前读取运行状态，执行中重复采集坐标位置、进给速度和运行状态，执行后读取最终状态并评估输出变化",
        "--target",
        "ncguide-generated-cpp",
        "--out",
        str(PROJECT_ROOT / "runs" / "iiegpt_ncguide_coordinate_motion_full"),
        "--task-id",
        "iiegpt_ncguide_coordinate_motion_full",
        "--quality-gate",
        "--llm-provider",
        "openai_compatible",
        "--llm-model",
        "gpt-5.5",
        "--llm-base-url",
        "https://iiegpt.gentelmen.cn/v1",
        "--llm-api-key-env",
        "MY_LLM_API_KEY",
        "--llm-transport",
        "http",
        "--focas-port",
        "8193",
        "--trigger-ncguide-ui",
    ]

    print("Starting full SMPAgent NCGuide coordinate-motion run...", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print("The API key is read from .env via MY_LLM_API_KEY.", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    print(f"\nSMPAgent exited with code {completed.returncode}.", flush=True)
    input("Press Enter to close this window...")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
