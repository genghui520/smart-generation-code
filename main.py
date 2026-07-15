from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_TASK = (
    "Generate high-quality FOCAS traffic around an NC coordinate motion program. "
    "The NC program must contain observable G00/G01 axis movement. "
    "Before execution collect CNC run status. During execution repeatedly collect "
    "coordinate position, run status, and feed speed. After execution collect final "
    "status and evaluate whether the collected traffic shows useful variation."
)
DEFAULT_OUT = Path("runs/focas_nc_position_main")
DEFAULT_VECTOR_DB = Path("rag_indexes/focas/vector_db")
DEFAULT_FOCAS_HEADER_DIR = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD")
VERBOSE_CONSOLE = False
RUN_STARTED_AT: float | None = None
AGENT_STARTED_AT: dict[str, float] = {}
AGENT_DURATIONS: dict[str, list[float]] = {}


def parse_args() -> argparse.Namespace:
    # 这里是论文/实验入口的统一配置区：默认任务、输出路径、RAG库、大模型和仿真器目标都在这里固定。
    parser = argparse.ArgumentParser(description="Run SMPAgent for NC-centered FOCAS traffic generation.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    parser.add_argument("--task-id", default="focas_nc_position_main")
    parser.add_argument("--protocol", default="focas")
    parser.add_argument(
        "--target",
        default="ncguide-generated-cpp",
        help=(
            "Use ncguide-generated-cpp to compile and run the generated C++ API script. "
            "Use ncguide-bridge-readonly for the older fixed read-only bridge path."
        ),
    )
    parser.add_argument("--llm-provider", default="openai_compatible", choices=["disabled", "tokenhub", "openai_compatible"])
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--llm-base-url", default="https://fast.smartaipro.cn/v1")
    parser.add_argument("--llm-api-key-env", default="SMARTAIPRO_API_KEY")
    parser.add_argument(
        "--focas-header-dir",
        type=Path,
        default=Path(os.environ.get("FOCAS_HEADER_DIR", str(DEFAULT_FOCAS_HEADER_DIR))),
        help="Directory containing the controller-specific official FANUC Fwlib32.h header.",
    )
    parser.add_argument(
        "--allow-delete-all-programs",
        action="store_true",
        help="Authorize PlannerAgent to consider cnc_delall for this run. Disabled by default.",
    )
    parser.add_argument(
        "--trigger-ncguide-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NCGuide UI cycle-start triggering inside the generated C++ script. Enabled by default.",
    )
    parser.add_argument("--ncguide-window-title", default="FANUC CNC GUIDE")
    parser.add_argument("--ncguide-start-button-text", default="")
    parser.add_argument("--ncguide-mode-x", type=int, default=0)
    parser.add_argument("--ncguide-mode-y", type=int, default=0)
    parser.add_argument(
        "--ncguide-click-mode",
        default="screen",
        choices=["client", "screen"],
        help="Use screen coordinates for NCGuide floating panels, or client coordinates for the main window.",
    )
    parser.add_argument("--ncguide-cycle-start-x", type=int, default=989)
    parser.add_argument("--ncguide-cycle-start-y", type=int, default=914)
    parser.add_argument(
        "--manual-cycle-start-wait",
        type=int,
        default=0,
        help="Wait this many seconds after program selection so the operator can press Cycle Start manually.",
    )
    parser.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        default=True,
        help="Only run planning and code generation. By default, ExecutionAgent is connected and executed.",
    )
    parser.add_argument(
        "--verbose-console",
        action="store_true",
        help="Print full plan/artifact/execution details to the console. By default details are saved to files only.",
    )
    return parser.parse_args()


def main() -> int:
    global RUN_STARTED_AT, VERBOSE_CONSOLE
    args = parse_args()
    AGENT_STARTED_AT.clear()
    AGENT_DURATIONS.clear()
    RUN_STARTED_AT = time.perf_counter()
    VERBOSE_CONSOLE = args.verbose_console
    os.environ["FOCAS_HEADER_DIR"] = str(args.focas_header_dir)
    configure_ncguide_ui_env(args)

    print("Starting SMPAgent...", flush=True)

    from smart_traffic_agent.knowledge import KnowledgeBase
    from smart_traffic_agent.llm import LlmClient, LlmConfig
    from smart_traffic_agent.memory import LongTermMemoryStore
    from smart_traffic_agent.models import TaskRequest, WorkflowState
    from smart_traffic_agent.utils import ensure_dir, json_default, write_json
    from smart_traffic_agent.workflow import TrafficGenerationWorkflow, workflow_summary

    out_dir = ensure_dir(next_run_output_dir(args.out))
    args.out = out_dir

    # 1. 打印实验头信息，保证每次生成流量时能看到任务、协议、目标环境和输出目录。
    print_header(args)

    # 2. 加载已经构建好的 FOCAS RAG 向量知识库，PlannerAgent 会从这里检索规则和API上下文。
    print("Loading RAG knowledge base...", flush=True)
    knowledge_base = KnowledgeBase(vector_db_dir=args.vector_db)

    # 3. 接入大模型。默认使用项目本地 .env 中的 SMARTAIPRO_API_KEY，不需要写到系统全局环境变量。
    print("Initializing LLM client...", flush=True)
    llm_client = LlmClient.from_config(
        LlmConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )
    )

    # 4. 初始化 LangGraph 多Agent工作流。这里复用系统里的 Router/Planner/CodeGenerator/Executor/Annotator。
    print("Initializing workflow...", flush=True)
    memory_store = LongTermMemoryStore()
    workflow = TrafficGenerationWorkflow(
        knowledge_base,
        llm_client=llm_client,
        memory_store=memory_store,
        progress_callback=print_progress_event,
    )

    # 5. 构造任务状态。target_environment 决定 ExecutionAgent 连接仿真客户端还是 FANUC NCGuide bridge。
    state = WorkflowState(
        request=TaskRequest(
            description=args.task,
            task_id=args.task_id,
            protocol=args.protocol,
            target_environment=args.target,
            permissions={"allow_delete_all_programs": args.allow_delete_all_programs},
        )
    )
    state.long_term_memories = memory_store.search(args.task)
    if state.long_term_memories:
        print(f"    long_term_memories: {len(state.long_term_memories)}", flush=True)

    if args.execute:
        print("\n========== Runtime ==========", flush=True)
        state = workflow.run(state.request, out_dir)
        summary = workflow_summary(state)
        write_json(out_dir / "summary.json", summary)
        metrics = build_run_metrics(state, summary)
        write_json(out_dir / "run_metrics.json", metrics)
        if VERBOSE_CONSOLE:
            print_detailed_outputs(state)
            print("\n========== Summary ==========")
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
            print("=============================")
        else:
            print_run_files(out_dir, summary)
        return 0

    # 6. 路由Agent：根据当前状态判断下一步应该进入规划、代码生成、执行还是标注。
    print("\n[1] RouterAgent")
    state.stage = workflow.router.route(state)
    print(f"    next_stage: {state.stage}")

    # 7. 规划Agent：围绕用户任务选择场景模板、检索RAG规则，并生成NC程序需求和API采集步骤。
    print("\n[2] PlanningAgent")
    state = workflow.planner.run(state)
    write_json(out_dir / "plan.json", state.plan)
    if VERBOSE_CONSOLE:
        print_plan(state)
    else:
        print(f"    plan: {out_dir / 'plan.json'}")

    # 8. 再次路由：规划完成后应进入代码生成阶段。
    print("\n[3] RouterAgent")
    state.stage = workflow.router.route(state)
    print(f"    next_stage: {state.stage}")

    # 9. 代码生成Agent：生成NC程序、Python API脚本和FOCAS C++测试框架。
    print("\n[4] CodeGenerationAgent")
    state = workflow.generator.run(state, out_dir)
    if VERBOSE_CONSOLE:
        print_artifacts(state)
    else:
        assert state.artifacts is not None
        print(f"    api_script: {state.artifacts.api_script_path}")
        print(f"    nc_program: {state.artifacts.nc_program_path}")

    # 10. 再次路由：代码生成完成后应进入执行阶段。
    print("\n[5] RouterAgent")
    state.stage = workflow.router.route(state)
    print(f"    next_stage: {state.stage}")

    if args.execute:
        # 11. 执行Agent：连接当前 target 指定的执行环境。
        # 当前默认 ncguide-generated-cpp 会编译并运行 CodeGenerationAgent 生成的 api_script.cpp。
        print("\n[6] ExecutionAgent")
        state = workflow.executor.run(state, out_dir)
        if VERBOSE_CONSOLE:
            print_execution(state)
        else:
            assert state.result is not None
            print(
                f"    success={state.result.success} api_logs={len(state.result.api_logs)} "
                f"capture_events={len(state.result.capture_events)}"
            )

        # 12. 再次路由：执行后由 Router 判断是否完成或进入修复。
        print("\n[7] RouterAgent")
        state.stage = workflow.router.route(state)
        print(f"    next_stage: {state.stage}")
    else:
        print("\n[6] ExecutionAgent")
        print("    skipped: --no-execute was set, so only planning and code generation were performed.")

    # 14. 写出本次运行摘要，便于论文实验记录和后续批量对比。
    summary = workflow_summary(state)
    memory_store.remember_workflow(state)
    write_json(out_dir / "summary.json", summary)
    metrics = build_run_metrics(state, summary)
    write_json(out_dir / "run_metrics.json", metrics)
    if VERBOSE_CONSOLE:
        print("\n========== Summary ==========")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
        print("=============================")
    else:
        print_run_files(out_dir, summary)
    return 0


def configure_ncguide_ui_env(args: argparse.Namespace) -> None:
    os.environ["NCGUIDE_ENABLE_UI_START"] = "1" if args.trigger_ncguide_ui else "0"
    os.environ["NCGUIDE_WINDOW_TITLE"] = args.ncguide_window_title
    os.environ["NCGUIDE_START_BUTTON_TEXT"] = args.ncguide_start_button_text
    os.environ["NCGUIDE_MODE_X"] = str(args.ncguide_mode_x)
    os.environ["NCGUIDE_MODE_Y"] = str(args.ncguide_mode_y)
    os.environ["NCGUIDE_CLICK_MODE"] = args.ncguide_click_mode
    os.environ["NCGUIDE_CYCLE_START_X"] = str(args.ncguide_cycle_start_x)
    os.environ["NCGUIDE_CYCLE_START_Y"] = str(args.ncguide_cycle_start_y)
    os.environ["NCGUIDE_MANUAL_START_WAIT_SECONDS"] = str(args.manual_cycle_start_wait)


def next_run_output_dir(base_out: Path) -> Path:
    if not base_out.exists():
        return base_out / "run_001"
    for index in range(1, 10000):
        candidate = base_out / f"run_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a new run directory under {base_out}")


def print_progress_event(event: str, payload: dict) -> None:
    if event == "workflow_start":
        print(f"[Workflow] start task={payload['task_id']} out={payload['output_dir']}", flush=True)
    elif event == "memory_loaded":
        if payload.get("count"):
            print(f"[Memory] loaded {payload['count']} memories", flush=True)
    elif event == "memory_reloaded_for_error":
        print(f"[Memory] reloaded {payload['count']} memories for error", flush=True)
        if VERBOSE_CONSOLE:
            for error in payload.get("errors", []):
                print(f"    error_context: {error}", flush=True)
    elif event == "router_decision":
        errors = payload.get("errors", [])
        suffix = f" errors={len(errors)}" if errors else ""
        print(f"[RouterAgent] -> {payload['next_stage']}{suffix}", flush=True)
        if VERBOSE_CONSOLE:
            print(
                "    state: "
                f"plan={payload['has_plan']} artifacts={payload['has_artifacts']} "
                f"result={payload['has_result']} mapping={payload['mapping_count']}",
                flush=True,
            )
            for error in errors:
                print(f"    error: {error}", flush=True)
    elif event == "agent_start":
        agent = payload["agent"]
        AGENT_STARTED_AT[agent] = time.perf_counter()
        print(f"[{agent}] start", flush=True)
        if VERBOSE_CONSOLE and payload.get("task"):
            print(f"    task: {payload['task']}", flush=True)
    elif event == "agent_complete":
        print_agent_complete(payload)
    elif event == "agent_error":
        agent = payload["agent"]
        duration = pop_elapsed_text(AGENT_STARTED_AT, agent)
        suffix = f" elapsed={duration}" if duration else ""
        print(f"[{agent}] failed{suffix}: {short_text(str(payload['error']), 180)}", flush=True)
    elif event == "tool_start":
        print(f"    [Tool] {payload['tool']} start", flush=True)
    elif event == "tool_complete":
        status = "ok" if payload.get("success") else "failed"
        print(
            f"    [Tool] {payload['tool_name']} {status} elapsed={payload.get('duration_ms', 0)}ms",
            flush=True,
        )
    elif event == "repair_start":
        print(
            f"[Repair] attempt={payload['attempt']} stage={payload['repair_stage']}",
            flush=True,
        )
        if payload.get("router_reason"):
            print(f"    reason: {short_text(payload['router_reason'], 180)}", flush=True)
        if VERBOSE_CONSOLE:
            print(f"    action: {payload['action']}", flush=True)
            if payload.get("repair_instruction"):
                print(f"    repair_instruction: {payload['repair_instruction']}", flush=True)
            for error in payload.get("errors", []):
                print(f"    reason: {error}", flush=True)
    elif event == "repair_complete":
        print(f"[Repair] next -> {payload['next_stage']}", flush=True)
    elif event == "repair_stopped":
        print(f"[Repair] stopped after {payload['repair_attempts']} attempts", flush=True)
    elif event == "workflow_complete":
        print(
            f"[Workflow] complete stage={payload['stage']} success={payload['success']} "
            f"repair_attempts={payload['repair_attempts']} elapsed={total_elapsed_text()}",
            flush=True,
        )
        print(f"    summary: {payload['summary_path']}", flush=True)
    else:
        if VERBOSE_CONSOLE:
            print(f"[{event}] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def print_agent_complete(payload: dict) -> None:
    agent = payload["agent"]
    duration = pop_elapsed_text(AGENT_STARTED_AT, agent)
    suffix = f" elapsed={duration}" if duration else ""
    print(f"[{agent}] complete{suffix}", flush=True)
    if agent == "PlanningAgent":
        print(
            f"    scenario={payload['scenario_type']} "
            f"steps={payload['plan_steps']} retrieved_chunks={payload['retrieved_chunks']}",
            flush=True,
        )
        print(f"    plan: {payload['plan_path']}", flush=True)
        if VERBOSE_CONSOLE:
            nc_program_spec = payload.get("nc_program_spec", {})
            if nc_program_spec:
                print("    planner_nc_spec:", flush=True)
                print(f"      program_name: {nc_program_spec.get('program_name')}", flush=True)
                print(f"      purpose: {nc_program_spec.get('purpose')}", flush=True)
                print("      block_goals:", flush=True)
                for goal in payload.get("nc_program_spec", {}).get("block_goals", []):
                    print(f"        - {goal}", flush=True)
            print("    planned_steps:", flush=True)
            for step in payload.get("steps", []):
                print(
                    "      - "
                    f"{step['step_id']} [{step['phase']}] "
                    f"{step['interface_name']}/{step['protocol_function']} "
                    f"repeat={step['repeat']} interval={step['interval_seconds']} "
                    f"{step['action']}",
                    flush=True,
                )
    elif agent == "CodeGenerationAgent":
        print(f"    api_script: {payload['api_script_path']}", flush=True)
        print(f"    nc_program: {payload['nc_program_path']}", flush=True)
        diagnostics = payload.get("diagnostics", [])
        if diagnostics:
            print(f"    diagnostics: {len(diagnostics)} items (see summary/artifacts)", flush=True)
        if VERBOSE_CONSOLE:
            for item in diagnostics:
                print(f"    diagnostic: {item}", flush=True)
    elif agent == "ExecutionAgent":
        print(
            f"    success={payload['success']} api_logs={payload['api_logs']} "
            f"capture_events={payload['capture_events']} tool_calls={payload.get('tool_calls', 0)}",
            flush=True,
        )
        print(f"    output_dir: {payload['output_dir']}", flush=True)
        quality = payload.get("quality_assessment")
        if quality is not None:
            metrics = getattr(quality, "metrics", {}) or {}
            print(
                "    quality_metrics: "
                f"feed_samples={metrics.get('feed_sample_count')} "
                f"feed_unique={metrics.get('feed_unique_count')} "
                f"position_samples={metrics.get('position_sample_count')} "
                f"position_unique={metrics.get('position_unique_count')} "
                f"run_active={metrics.get('run_active_count')} "
                f"motion_active={metrics.get('motion_active_count')}",
                flush=True,
            )
        errors = payload.get("errors", [])
        if errors:
            print(f"    errors: {len(errors)} (see execution logs/summary)", flush=True)
        if VERBOSE_CONSOLE:
            for error in errors:
                print(f"    error: {error}", flush=True)


def print_detailed_outputs(state: WorkflowState) -> None:
    print("\n========== Final Artifacts ==========")
    if state.plan is not None:
        print("\n[Plan]")
        print_plan(state)
    if state.artifacts is not None:
        print("\n[Generated Files]")
        print_artifacts(state)
    if state.result is not None:
        print("\n[Execution]")
        print_execution(state)
    if state.repair_history:
        print("\n[Repair History]")
        for item in state.repair_history:
            print(
                f"    attempt={item.get('attempt')} stage={item.get('repair_stage')} "
                f"action={item.get('action')}"
            )
            for error in item.get("errors", []):
                print(f"      - {error}")
    print("===================================")


def print_header(args: argparse.Namespace) -> None:
    print("========== SMPAgent Run ==========")
    print(f"task_id={args.task_id} target={args.target} execute={args.execute}")
    print(f"out={args.out}")
    print("==================================")


def print_run_files(out_dir: Path, summary: dict) -> None:
    print("\n========== Files ==========")
    print(f"elapsed: {total_elapsed_text()}")
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"metrics: {out_dir / 'run_metrics.json'}")
    print(f"plan: {out_dir / 'plan.json'}")
    artifacts = summary.get("artifacts") or {}
    if artifacts.get("api_script_path"):
        print(f"api_script: {artifacts['api_script_path']}")
    if artifacts.get("nc_program_path"):
        print(f"nc_program: {artifacts['nc_program_path']}")
    if summary.get("api_log_count"):
        print(f"execution: {out_dir / 'execution'}")
    print("=========================")


def short_text(value: str, max_chars: int = 160) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def pop_elapsed_text(starts: dict[str, float], key: str) -> str:
    started_at = starts.pop(key, None)
    if started_at is None:
        return ""
    elapsed = time.perf_counter() - started_at
    AGENT_DURATIONS.setdefault(key, []).append(elapsed)
    return format_elapsed(elapsed)


def total_elapsed_text() -> str:
    if RUN_STARTED_AT is None:
        return "n/a"
    return format_elapsed(time.perf_counter() - RUN_STARTED_AT)


def total_elapsed_seconds() -> float | None:
    if RUN_STARTED_AT is None:
        return None
    return round(time.perf_counter() - RUN_STARTED_AT, 3)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m{remainder:04.1f}s"


def build_run_metrics(state: WorkflowState, summary: dict) -> dict:
    quality_metrics = {}
    if state.quality_assessment is not None:
        quality_metrics = state.quality_assessment.metrics
    result_errors = state.result.errors if state.result is not None else []
    artifact_diagnostics = state.artifacts.diagnostics if state.artifacts is not None else []
    return {
        "task_id": state.request.task_id,
        "target_environment": state.request.target_environment,
        "stage": state.stage,
        "success": summary.get("success"),
        "total_elapsed_seconds": total_elapsed_seconds(),
        "agent_durations_seconds": {
            agent: [round(value, 3) for value in values]
            for agent, values in AGENT_DURATIONS.items()
        },
        "agent_total_durations_seconds": {
            agent: round(sum(values), 3)
            for agent, values in AGENT_DURATIONS.items()
        },
        "repair_attempts": state.repair_attempts,
        "repair_history": [
            {
                "attempt": item.get("attempt"),
                "repair_stage": item.get("repair_stage"),
                "router_reason": item.get("router_reason"),
                "repair_instruction": item.get("repair_instruction"),
                "errors": item.get("errors", []),
            }
            for item in state.repair_history
        ],
        "api_log_count": len(state.result.api_logs) if state.result is not None else 0,
        "capture_event_count": len(state.result.capture_events) if state.result is not None else 0,
        "result_success": state.result.success if state.result is not None else None,
        "result_errors": result_errors,
        "workflow_errors": state.errors,
        "quality_metrics": quality_metrics,
        "changed_output_parameter_count": quality_metrics.get("changed_output_parameter_count"),
        "feed_sample_count": quality_metrics.get("feed_sample_count"),
        "feed_unique_count": quality_metrics.get("feed_unique_count"),
        "position_sample_count": quality_metrics.get("position_sample_count"),
        "position_unique_count": quality_metrics.get("position_unique_count"),
        "run_active_count": quality_metrics.get("run_active_count"),
        "motion_active_count": quality_metrics.get("motion_active_count"),
        "plan_steps": len(state.plan.steps) if state.plan is not None else 0,
        "scenario_type": state.plan.scenario_type if state.plan is not None else None,
        "retrieved_chunks": len(state.retrieved_chunks),
        "long_term_memory_count": len(state.long_term_memories),
        "artifact_diagnostics_count": len(artifact_diagnostics),
        "artifact_diagnostics": artifact_diagnostics,
        "summary_path": "summary.json",
        "plan_path": "plan.json",
    }


def print_plan(state: WorkflowState) -> None:
    # 打印 PlannerAgent 的核心结果：场景、NC需求、RAG检索数量和API操作步骤。
    assert state.plan is not None
    plan = state.plan
    print(f"    scenario_type: {plan.scenario_type}")
    print(f"    scenario_goal: {plan.scenario_goal}")
    print(f"    nc_program_type: {plan.nc_program_type}")
    print("    planner_nc_spec:")
    print(f"      program_name: {plan.nc_program_spec.program_name}")
    print(f"      purpose: {plan.nc_program_spec.purpose}")
    print("      block_goals:")
    for goal in plan.nc_program_spec.block_goals:
        print(f"        - {goal}")
    print("      constraints:")
    for constraint in plan.nc_program_spec.constraints:
        print(f"        - {constraint}")
    for note in plan.nc_program_spec.generation_notes:
        print(f"      note: {note}")
    print(f"    plan_steps: {len(plan.steps)}")
    print(f"    retrieved_chunks: {len(state.retrieved_chunks)}")
    print("    rag_context_counts:")
    for key, rows in plan.rag_context.items():
        print(f"      - {key}: {len(rows)}")
    print("    nc_program_requirements:")
    for item in plan.nc_program_requirements[:10]:
        print(f"      - {item}")
    print("    operation_steps:")
    for step in plan.steps:
        print(
            "      - "
            f"{step.step_id} [{step.phase}] {step.interface_name}/{step.protocol_function} "
            f"repeat={step.repeat} interval={step.interval_seconds}: {step.action}"
        )
    if plan.llm_notes:
        print("    llm_notes:")
        for note in plan.llm_notes:
            print(f"      - {note}")


def print_artifacts(state: WorkflowState) -> None:
    # 打印 CodeGenerationAgent 的输出文件，尤其关注生成的NC程序是否能产生可观察运动。
    assert state.artifacts is not None
    artifacts = state.artifacts
    print(f"    c++ api_script: {artifacts.api_script_path}")
    print(f"    nc_program: {artifacts.nc_program_path}")
    print("    nc_program_preview:")
    for line in artifacts.nc_program.splitlines():
        print(f"      {line}")
    print("    diagnostics:")
    for item in artifacts.diagnostics:
        print(f"      - {item}")


def print_execution(state: WorkflowState) -> None:
    # 打印 ExecutionAgent 的采集摘要：API日志数量、采集事件数量、返回状态和错误。
    assert state.result is not None
    result = state.result
    print(f"    success: {result.success}")
    print(f"    api_logs: {len(result.api_logs)}")
    print(f"    capture_events: {len(result.capture_events)}")
    print(f"    tool_calls: {len(result.tool_calls)}")
    if result.errors:
        print("    errors:")
        for error in result.errors:
            print(f"      - {error}")
    if state.quality_assessment is not None:
        qa = state.quality_assessment
        print(f"    quality_passed: {qa.passed}")
        print(f"    quality_metrics: {json.dumps(qa.metrics, ensure_ascii=False)}")
        if qa.issues:
            print("    quality_issues:")
            for issue in qa.issues:
                print(f"      - {issue}")
        if qa.recommendations:
            print("    quality_recommendations:")
            for item in qa.recommendations:
                print(f"      - {item}")
    print("    api_log_preview:")
    for log in result.api_logs[:20]:
        response = json.dumps(log.response, ensure_ascii=False)
        print(
            "      - "
            f"{log.step_id} {log.interface_name}/{log.protocol_function} "
            f"status={log.status_code} label={log.semantic_label} response={response[:360]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
