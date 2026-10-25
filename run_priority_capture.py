#!/usr/bin/env python3
"""Start a desktop full-API capture for the read-only, no-parameter-change set."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "runs" / "desktop" / "20261023_152654_fd3f43" / "settings.json"
PRIORITY = ROOT / "inventcom_output" / "api_capture_priority.json"
LOG_DIR = ROOT / "runs" / "desktop"


def env_file_values() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_config(mode: str = "auto") -> dict:
    config = json.loads(SETTINGS.read_text(encoding="utf-8"))
    config.pop("selected", None)
    config["api_key"] = env_file_values().get("DEEPSEEK_API_KEY", "")
    from smart_traffic_agent.desktop_backend import (
        auto_no_parameter_functions, read_only_priority_functions,
    )
    if mode == "auto":
        config["capture_mode"] = "auto_no_parameter"
        targets = auto_no_parameter_functions(config)
        config["requirements"] = (
            "只采集 AUTO/MEM 模式可用、且无 CNC 参数前置的 API；"
            "写/控制类必须先读取原值，执行后校验并恢复；"
            "不为了覆盖而破坏数据。"
        )
    else:
        config["capture_mode"] = "read_only_priority"
        targets = read_only_priority_functions(config)
        config["requirements"] = (
            "只采集无需修改 CNC 参数、无需切换 CNC 模式的只读 API；"
            "测试数据不存在时记录限制，不臆造编号，不为了覆盖而写入。"
        )
    if not targets:
        raise RuntimeError(f"no eligible targets for mode={mode}")
    config["selected_apis"] = targets
    config["action"] = "capture"
    config["ui_start"] = mode != "read_only"
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "read_only"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    config = build_config(args.mode)
    from smart_traffic_agent.desktop_backend import validate, API_CAPTURE_BATCH_SIZE
    errors = validate(config, capture=True)
    if errors:
        print("validation_failed:", errors)
        return 2

    count = len(config["selected_apis"])
    batches = max(1, -(-count // API_CAPTURE_BATCH_SIZE))
    print(json.dumps({
        "targets": count,
        "batch_size": API_CAPTURE_BATCH_SIZE,
        "expected_batches": batches,
        "output": config["output"],
        "first_targets": config["selected_apis"][:8],
    }, ensure_ascii=False))
    if args.dry_run:
        return 0

    env = dict(os.environ)
    if args.max_batches > 0:
        env["FOCAS_MAX_CAPTURE_BATCHES"] = str(args.max_batches)
    else:
        env["FOCAS_MAX_CAPTURE_BATCHES"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = LOG_DIR / f"priority_ready_{datetime.now():%Y%m%d_%H%M%S}.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", "smart_traffic_agent.desktop_backend"],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    print("log:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith("@@SMP@@"):
                try:
                    event = json.loads(line[len("@@SMP@@"):]).get("event")
                except json.JSONDecodeError:
                    event = "?"
                if event in {"coverage_batch", "coverage_progress", "coverage_batch_failed",
                             "coverage_batch_failed", "status", "success", "failure", "output"}:
                    print(line.rstrip(), flush=True)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
