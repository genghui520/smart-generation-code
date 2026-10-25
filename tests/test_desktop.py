import io
import json
from unittest.mock import patch, MagicMock

import pytest

from smart_traffic_agent import desktop_backend as backend


def config():
    result = backend.defaults()
    result["api_key"] = "test-secret-not-real"
    result["selected"] = [backend.scenarios()[0]["scenario_id"]]
    return result


def test_desktop_uses_deepseek_api_key_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-secret")
    assert backend.defaults()["api_key"] == "deepseek-test-secret"
    cfg = config()
    cfg["api_key"] = "ui-entered-secret"
    backend.configure(cfg)
    assert backend.os.environ["DEEPSEEK_API_KEY"] == "ui-entered-secret"
    client = backend.make_llm(cfg)
    assert client.http_config.api_key_env == "DEEPSEEK_API_KEY"


def test_prompt_tracks_selected_scene_and_endpoint():
    cfg = config()
    cfg.update(host="192.0.2.5", port="9000", requirements="仅采集只读 API")
    first, second = backend.scenarios()[:2]
    prompt = backend.render_prompt(cfg, first)
    assert first["scenario_id"] in prompt
    assert second["scenario_id"] not in prompt
    assert "192.0.2.5:9000" in prompt
    assert "仅采集只读 API" in prompt
    assert "parameters=" in prompt
    assert first["main_apis"][0] in prompt
    assert backend.render_prompt(cfg, second) != prompt


def test_select_all_switches_to_global_api_and_parameter_coverage():
    cfg = config()
    cfg["selected"] = [scene["scenario_id"] for scene in backend.scenarios()]
    assert backend.is_full_api_mode(cfg)
    prompts = backend.render_selected_prompts(cfg)
    assert len(prompts) == 1
    assert str(len(backend.selected_api_names(cfg))) in prompts[0]
    assert "每一个 API" in prompts[0]
    assert "枚举参数" in prompts[0]
    assert "边界值" in prompts[0]
    assert "不得在只完成一个批次时宣称全覆盖" in prompts[0]


def test_selected_api_prompt_and_validation_do_not_expand_to_all_functions():
    cfg = config()
    cfg["selected"] = []
    cfg["selected_apis"] = ["cnc_statinfo", "cnc_actf"]
    prompt = backend.render_selected_prompts(cfg)[0]
    assert "本次目标共 2 个唯一 API" in prompt
    assert "cnc_statinfo" in prompt and "cnc_actf" in prompt
    assert not backend.is_full_api_mode(cfg)
    assert backend.selected_api_names(cfg) == [
        row["function"] for row in backend.function_manifest()
        if row["function"] in {"cnc_statinfo", "cnc_actf"}
    ]
    cfg["selected_apis"] = []
    assert any("至少选择一个 API" in error for error in backend.validate(cfg, capture=True))
    cfg["selected_apis"] = ["cnc_not_real"]
    assert any("未知 API" in error for error in backend.validate(cfg, capture=True))


def test_batch_prompt_mentions_only_current_targets():
    cfg = config()
    cfg["selected_apis"] = ["cnc_statinfo", "cnc_actf", "cnc_absolute"]
    prompt = backend.render_full_api_prompt(cfg, batch_targets=["cnc_statinfo", "cnc_actf"])
    assert "本次目标共 2 个唯一 API" in prompt
    assert "cnc_statinfo" in prompt and "cnc_actf" in prompt
    assert "cnc_absolute" not in prompt


def test_single_api_failure_does_not_dispatch_another_batch(tmp_path):
    from smart_traffic_agent.utils import write_json

    header_dir = tmp_path / "sdk"
    header_dir.mkdir()
    (header_dir / "Fwlib32.h").write_text("short cnc_renameprog(void);", encoding="ascii")
    cfg = config()
    cfg.update(capture_mode="single_api", selected_apis=["cnc_renameprog"], header=str(header_dir))
    prompt = backend.render_full_api_prompt(cfg, batch_targets=["cnc_renameprog"])
    assert "必须分批覆盖" not in prompt
    assert "持续执行后续批次" not in prompt
    write_json(tmp_path / "full_api_coverage_state.json", {
        "schema_version": 1,
        "functions": {
            "cnc_renameprog": {
                "function": "cnc_renameprog", "status": "covered_success",
                "traffic_matched": True, "coverage_role": "target",
            },
        },
        "runs": [],
    })

    events = []
    failed_result = MagicMock(plan=None, result=None, errors=["CODEGEN_REPAIR_EXHAUSTED"])
    with patch.object(backend, "function_manifest", return_value=[{"function": "cnc_renameprog"}]), \
         patch.object(backend, "selected_api_names", return_value=["cnc_renameprog"]), \
         patch.object(backend, "make_llm", return_value=None), \
         patch("smart_traffic_agent.workflow.TrafficGenerationWorkflow") as workflow, \
         patch("smart_traffic_agent.workflow.workflow_success", return_value=False), \
         patch("smart_traffic_agent.function_coverage.build_function_coverage_metrics", return_value={}):
        workflow.return_value.run.return_value = failed_result
        with pytest.raises(RuntimeError, match="单 API 采集失败"):
            backend.run_full_api_capture(cfg, tmp_path, None, lambda name, data: events.append((name, data)))
        assert workflow.return_value.run.call_count == 1

    failures = json.loads((tmp_path / "coverage_batch_failures.json").read_text(encoding="utf-8"))
    assert len(failures["failures"]) == 1
    assert failures["failures"][0]["targets"] == ["cnc_renameprog"]
    assert not (tmp_path / "batch_002").exists()
    assert any(name == "coverage_batch_failed" and data["single_api"] for name, data in events)


def test_selected_api_unavailable_does_not_mark_full_coverage(tmp_path):
    header = tmp_path / "sdk"
    header.mkdir()
    (header / "Fwlib32.h").write_text("short cnc_other(void);", encoding="ascii")
    cfg = config()
    cfg.update(selected=[], selected_apis=["cnc_missing"], header=str(header))
    with patch.object(backend, "function_manifest", return_value=[
        {"function": "cnc_missing"}, {"function": "cnc_other"},
    ]):
        assert backend.run_full_api_capture(cfg, tmp_path, None, lambda *_: None) == 0
    state = json.loads((tmp_path / "full_api_coverage_state.json").read_text(encoding="utf-8"))
    assert state["functions"]["cnc_missing"]["status"] == "unsupported_by_header"
    progress = json.loads((tmp_path / "coverage_progress.json").read_text(encoding="utf-8"))
    assert progress["last_batch_status"] == "partial"
    assert progress["captured_function_count"] == 0
    assert progress["selected_api_functions"] == ["cnc_missing"]


def test_validation_rejects_invalid_settings_and_template():
    cfg = config()
    cfg.update(port="70000", interval="nan", selected=[], template="$unknown", base_url="broken")
    errors = backend.validate(cfg, capture=True)
    assert any("port" in e for e in errors)
    assert any("interval" in e for e in errors)
    assert any("场景" in e for e in errors)
    cfg["selected"] = [backend.scenarios()[0]["scenario_id"]]
    assert any("模板" in e for e in backend.validate(cfg, capture=True))


def test_capture_requires_cycle_start_calibration_when_ui_start_enabled():
    cfg = config()
    cfg.update(start_x="", start_y="", ui_start=True)
    errors = backend.validate(cfg, capture=True)
    assert any("开始定位 Cycle Start" in error for error in errors)


def test_cycle_start_calibration_accepts_native_negative_monitor_coordinates():
    cfg = config()
    cfg.update(start_x="-150", start_y="0", ui_start=True)
    assert not any("开始定位 Cycle Start" in error for error in backend.validate(cfg, capture=True))


def test_scene_api_names_split_atomic_functions():
    scene = {"main_apis": ["cnc_statinfo/cnc_actf", "cnc_absolute;cnc_actf", "not_an_api"]}
    assert backend.atomic_scene_api_names(scene) == ["cnc_absolute", "cnc_actf", "cnc_statinfo"]


def test_source_clusters_are_grouped_into_operation_scenes_without_losing_apis():
    from smart_traffic_agent.agents.planner import load_scenario_templates
    from smart_traffic_agent.rag.scenario_templates import atomic_template_apis

    source = json.loads((backend.ROOT / "rag_indexes/focas/final_scenario_templates.json").read_text(encoding="utf-8"))["templates"]
    scenes = backend.scenarios()
    assert len(scenes) < len(source)
    assert len({scene["scenario_name"] for scene in scenes}) == len(scenes)
    assert {api for row in source for api in atomic_template_apis(row["main_apis"])} == {
        api for scene in scenes for api in scene["target_apis"]
    }
    assert all(scene["device_states"] and scene["source_cluster_ids"] for scene in scenes)
    program = next(scene for scene in scenes if scene["scenario_name"] == "program_lifecycle")
    assert any(cluster.startswith("cluster_12_") for cluster in program["source_cluster_ids"])
    assert all(not cluster.startswith("cluster_12_") for scene in scenes
               if scene["scenario_name"] == "pmc_signal_monitoring"
               for cluster in scene["source_cluster_ids"])
    planner_scenes = load_scenario_templates(backend.ROOT / "rag_indexes/focas/final_scenario_templates.json")
    assert {scene["scenario_id"] for scene in planner_scenes} == {scene["scenario_id"] for scene in scenes}


def test_resume_requires_matching_settings_and_skips_only_verified_success(tmp_path):
    from smart_traffic_agent.utils import write_json

    cfg = config()
    cfg["output"] = str(tmp_path)
    run = tmp_path / "old_run"
    run.mkdir()
    write_json(run / "settings.json", {key: value for key, value in cfg.items() if key != "api_key"})
    cfg["resume_from"] = str(run)
    assert backend.capture_output_dir(cfg) == run
    scene = backend.scenarios()[0]
    folder = backend.next_scene_folder(run, 1, scene)
    folder.mkdir()
    (folder / "capture.pcap").write_bytes(b"pcap" * 20)
    backend.write_capture_progress(run / "capture_progress.json", {
        "scenarios": {scene["scenario_id"]: {"status": "success", "folder": str(folder)}}
    })
    assert backend.scene_has_verified_success(run, scene["scenario_id"])
    assert backend.next_scene_folder(run, 1, scene).name.endswith("retry_001")
    (folder / "capture.pcap").write_bytes(b"short")
    assert not backend.scene_has_verified_success(run, scene["scenario_id"])
    cfg["port"] = "8194"
    with pytest.raises(ValueError, match="port"):
        backend.capture_output_dir(cfg)


def test_failed_scene_continues_only_when_controller_is_verified_idle():
    cfg = config()
    with patch("smart_traffic_agent.integrations.ncguide.FocasCppBridgeClient") as bridge:
        bridge.return_value.call.return_value = {"status_code": 0, "data": {"run": 0, "motion": 0, "alarm": 0, "emergency": 0}}
        assert backend.simulator_is_idle_after_failure(cfg)
        bridge.return_value.call.return_value = {"status_code": 0, "data": {"run": 1, "motion": 0, "alarm": 0, "emergency": 0}}
        assert backend.simulator_is_idle_after_failure(cfg)
        bridge.return_value.call.return_value = {"status_code": 0, "data": {"run": 0, "motion": 0, "alarm": 0, "emergency": 1}}
        assert not backend.simulator_is_idle_after_failure(cfg)
        bridge.return_value.call.return_value = {"status_code": 0, "data": {"run": 0}}
        assert not backend.simulator_is_idle_after_failure(cfg)


def test_required_target_api_requires_invocation_but_not_online_traffic_match():
    from smart_traffic_agent.agents.executor import required_traffic_errors
    from smart_traffic_agent.models import ApiCallLog, TaskRequest, WorkflowState

    state = WorkflowState(request=TaskRequest(
        description="capture", task_id="target",
        permissions={"required_api_functions": ["cnc_actf", "cnc_statinfo"]},
    ))
    log = ApiCallLog(
        timestamp="2026-01-01T00:00:00", task_id="target", step_id="S001",
        phase="during", interface_name="ReadFeedSpeed", input_parameters={},
        status_code=0, response={}, protocol_function="cnc_actf",
        traffic_status="no_corresponding_traffic",
    )
    assert required_traffic_errors(state, [log]) == ["TARGET_API_NOT_CALLED: cnc_statinfo"]
    log.traffic_status = "request_only_segment"
    assert required_traffic_errors(state, [log]) == ["TARGET_API_NOT_CALLED: cnc_statinfo"]
    second = ApiCallLog(
        timestamp="2026-01-01T00:00:01", task_id="target", step_id="S002",
        phase="during", interface_name="ReadStatus", input_parameters={},
        status_code=0, response={}, protocol_function="cnc_statinfo",
        traffic_status="no_corresponding_traffic",
    )
    assert required_traffic_errors(state, [log, second]) == []


def test_fixed_motion_runner_rejects_unplanned_scene_or_api():
    from smart_traffic_agent.agents.code_generator import render_fixed_ncguide_cpp_script, use_fixed_ncguide_runner
    from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep, TaskRequest, WorkflowState

    request = TaskRequest(description="capture", task_id="motion", permissions={
        "required_api_functions": ["cnc_actf"],
    })
    state = WorkflowState(request=request)
    state.plan = ExecutionPlan(
        plan_id="motion", task_id="motion", scenario_type="coordinate_motion",
        scenario_goal="motion", target_environment="ncguide-generated-cpp",
        nc_program_type="test", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name="O1234"),
        steps=[], expected_outputs=[],
    )
    steps = [PlanStep("S001", "during", "start", "StartProgram", {}, protocol_function="cnc_statinfo")]
    assert use_fixed_ncguide_runner(state, steps)
    cpp = render_fixed_ncguide_cpp_script(state, steps, "O1234\nG90 G54\nM30\n")
    assert "CYCLE_START_EFFECT_NOT_OBSERVED_ABORT" in cpp
    assert "NCGUIDE_ENABLE_UI_START" in cpp
    assert "if (!SetCursorPos(x, y)) return false" in cpp
    state.plan.scenario_type = "tool_management"
    assert not use_fixed_ncguide_runner(state, steps)
    state.plan.scenario_type = "coordinate_motion"
    request.permissions["required_api_functions"] = ["cnc_wrparam"]
    assert not use_fixed_ncguide_runner(state, steps)


@pytest.mark.skipif(backend.os.name != "nt", reason="MSVC/FOCAS SDK compile check is Windows-only")
def test_fixed_motion_runner_compiles_without_starting_simulator(tmp_path):
    from smart_traffic_agent.agents.code_generator import preflight_compile_generated_cpp, render_fixed_ncguide_cpp_script
    from smart_traffic_agent.agent_tools.cpp_execution import find_msvc_x86_toolchain
    from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep, TaskRequest, WorkflowState

    if find_msvc_x86_toolchain() is None or not (backend.Path(backend.defaults()["header"]) / "Fwlib32.h").is_file():
        pytest.skip("local MSVC x86 or FOCAS header unavailable")
    state = WorkflowState(request=TaskRequest(description="capture", task_id="compile-motion"))
    state.plan = ExecutionPlan(
        plan_id="compile-motion", task_id="compile-motion", scenario_type="coordinate_motion",
        scenario_goal="motion", target_environment="ncguide-generated-cpp",
        nc_program_type="test", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name="O1234"), steps=[], expected_outputs=[],
    )
    steps = [PlanStep("S001", "during", "start", "StartProgram", {}, protocol_function="cnc_statinfo")]
    source = tmp_path / "api_script.cpp"
    source.write_text(render_fixed_ncguide_cpp_script(state, steps, "O1234\nG90 G54\nM30\n"), encoding="utf-8")
    assert preflight_compile_generated_cpp(source, tmp_path) == ""


def test_simulator_requires_protocol_success_not_just_tcp():
    with patch("socket.create_connection"), patch("smart_traffic_agent.integrations.ncguide.FocasCppBridgeClient") as bridge:
        bridge.return_value.call.return_value = {"status_code": -15}
        with pytest.raises(RuntimeError, match="FOCAS"):
            backend.test_simulator(config())
        bridge.return_value.call.return_value = {"status_code": 0}
        assert "成功" in backend.test_simulator(config())


def test_model_test_rejects_empty_response():
    with patch.object(backend, "make_llm") as factory:
        factory.return_value.invoke_text.return_value = " "
        with pytest.raises(RuntimeError, match="空响应"):
            backend.test_llm(config())


def test_worker_redacts_failure_and_returns_nonzero(monkeypatch, capsys):
    cfg = dict(config(), action="llm")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(cfg) + "\n"))
    with patch.object(backend, "configure"), patch.object(backend, "test_llm", side_effect=RuntimeError(cfg["api_key"])):
        original_out, original_err = backend.sys.stdout, backend.sys.stderr
        try:
            assert backend.worker() == 1
        finally:
            backend.sys.stdout, backend.sys.stderr = original_out, original_err
    output = capsys.readouterr().out
    assert cfg["api_key"] not in output
    assert "[REDACTED]" in output
    assert '"failure"' in output


def test_explicit_scene_rejects_unknown_id_before_retrieval():
    from smart_traffic_agent.agents.planner import PlanningAgent
    from smart_traffic_agent.models import TaskRequest, WorkflowState
    planner = PlanningAgent(MagicMock())
    state = WorkflowState(request=TaskRequest(description="coordinate motion", task_id="test",
                                             permissions={"selected_scenario_ids": ["not-a-scene"]}))
    with pytest.raises(ValueError, match="Unknown explicitly"):
        planner._run_with_timeout(state)


def test_desktop_build_preview_and_connection_invalidation():
    import tkinter as tk
    from tkinter import ttk
    from smart_traffic_agent.desktop import DesktopApp
    root = tk.Tk()
    root.withdraw()
    try:
        app = DesktopApp(root)
        root.update_idletasks()
        assert len(app.config()["selected_apis"]) == 1
        app.select_all(True)
        assert len(app.config()["selected_apis"]) == app.available_api_count()
        assert app.template_mode == "full"
        assert len(app.start_buttons) >= 2
        assert "开始采集" in app.start_buttons[0].cget("text")
        assert "全部 API" in app.preview_label.cget("text")
        assert "每一个 API" in app.preview_text.get("1.0", "end")
        app.select_all(False)
        app.api_search.set("cnc_statinfo")
        assert "cnc_statinfo" in app.visible_api_names
        app.select_filtered()
        assert "cnc_statinfo" in app.config()["selected_apis"]
        app.api_search.set("cnc_actf")
        assert "cnc_statinfo" in app.config()["selected_apis"]
        app.api_search.set("")
        actf_category = next(row["category"] for row in app.catalog if row["function"] == "cnc_actf")
        app.api_category.set(actf_category)
        assert "cnc_actf" in app.visible_api_names
        assert all((row.get("category") or "未分类") == actf_category
                   for row in app.catalog if row["function"] in app.visible_api_names)
        app.api_category.set("全部分类")
        assert "cnc_statinfo" in app.config()["selected_apis"]
        style = ttk.Style(root)
        assert style.lookup("Alert.TLabel", "foreground") == "#d62828"
        with patch("smart_traffic_agent.desktop.window_title_at_point", return_value="FANUC CNC GUIDE"):
            app.confirm_cycle_start_point(812, 744)
        assert app.values["start_x"].get() == "812"
        assert app.values["start_y"].get() == "744"
        assert app.values["window_title"].get() == "FANUC CNC GUIDE"
        assert "已确认" in app.calibration_status.get()
        app.verified["simulator"] = app.signature(app.config(), "simulator")
        app.values["port"].set("8194")
        assert app.signature(app.config(), "simulator") != app.verified["simulator"]
        assert "测试" in app.sim_status.get()
        app.select_all(False)
        assert not app.config()["selected_apis"]
    finally:
        root.destroy()


def test_api_panel_scroll_reaches_cycle_start_controls_at_minimum_window_height():
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp

    root = tk.Tk()
    root.geometry("920x680")
    try:
        app = DesktopApp(root)
        app.tabs.select(app.scene_tab)
        root.update()
        canvas = app.api_canvas
        button = app.locate_cycle_start_button
        assert button.winfo_rooty() >= canvas.winfo_rooty()
        assert button.winfo_rooty() + button.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
        canvas.yview_moveto(1.0)
        root.update()
        assert canvas.winfo_height() > 0
        assert button.winfo_rooty() >= canvas.winfo_rooty()
        assert button.winfo_rooty() + button.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
    finally:
        root.destroy()


@pytest.mark.skipif(backend.os.name != "nt", reason="Windows native overlay check")
def test_cycle_start_overlay_covers_native_virtual_screen():
    import ctypes
    import tkinter as tk
    from ctypes import wintypes
    from smart_traffic_agent.desktop import cover_virtual_screen, virtual_screen_geometry

    root = tk.Tk()
    root.withdraw()
    overlay = tk.Toplevel(root)
    overlay.withdraw()
    overlay.overrideredirect(True)
    try:
        overlay.deiconify()
        overlay.update_idletasks()
        cover_virtual_screen(overlay)
        overlay.update()
        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        hwnd = user32.GetAncestor(overlay.winfo_id(), 2) or overlay.winfo_id()
        rect = wintypes.RECT()
        assert user32.GetWindowRect(hwnd, ctypes.byref(rect))
        x, y, width, height = virtual_screen_geometry()
        assert (rect.left, rect.top, rect.right, rect.bottom) == (
            x, y, x + width, y + height,
        )
        # The bottom-right desktop point (where Main Panel can sit) is covered.
        assert rect.left <= x + width - 1 < rect.right
        assert rect.top <= y + height - 1 < rect.bottom
    finally:
        overlay.destroy()
        root.destroy()


def test_explicit_scene_controls_retrieval_even_with_conflicting_prompt():
    from smart_traffic_agent.agents.planner import PlanningAgent
    from smart_traffic_agent.models import TaskRequest, WorkflowState
    planner = PlanningAgent(MagicMock())
    scene = next(s for s in planner.scenario_templates if s["scenario_name"] == "alarm_diagnosis")
    state = WorkflowState(request=TaskRequest(description="coordinate position 坐标",
        task_id="test", permissions={"selected_scenario_ids": [scene["scenario_id"]]}))
    with patch("smart_traffic_agent.agents.planner.retrieve_planning_context", side_effect=RuntimeError("stop before LLM")) as retrieve:
        with pytest.raises(RuntimeError, match="stop before LLM"):
            planner._run_with_timeout(state)
        assert retrieve.call_args.args[2] == "alarm_query"


def test_explicit_capture_modes_keep_scene_and_api_scopes_separate():
    cfg = config()
    scene = backend.scenarios()[0]
    cfg.update(capture_mode="single_scene", selected_apis=["cnc_statinfo"])
    prompt = backend.render_selected_prompts(cfg)[0]
    assert scene["scenario_id"] in prompt
    assert not backend.is_full_api_mode(cfg)
    cfg.update(capture_mode="all_apis", selected_apis=[])
    assert backend.is_full_api_mode(cfg)
    assert backend.selected_api_names(cfg) == [row["function"] for row in backend.function_manifest() if not backend.unavailable_reason(row["function"], cfg)]
    assert "每一个 API" in backend.render_selected_prompts(cfg)[0]
    cfg.update(capture_mode="single_api", selected_apis=["cnc_actf", "cnc_statinfo"])
    assert any("仅选择一个 API" in error for error in backend.validate(cfg, capture=True))
    cfg.update(capture_mode="single_scene", selected=[])
    assert any("仅选择一个场景" in error for error in backend.validate(cfg, capture=True))
    readable = backend.read_only_priority_functions(cfg)
    cfg.update(capture_mode="read_only_priority", selected_apis=readable)
    assert readable
    assert set(backend.selected_api_names(cfg)) == set(readable)
    assert not backend.is_full_api_mode(cfg)
    assert not backend.validate(cfg, capture=True)
    assert readable[0] in backend.render_selected_prompts(cfg)[0]
    auto_names = backend.auto_no_parameter_functions(cfg)
    cfg.update(capture_mode="auto_no_parameter", selected_apis=auto_names)
    assert len(auto_names) > len(readable)
    assert {"cnc_wrparam", "cnc_download3", "cnc_modal", "cnc_rdgcode",
            "cnc_rdposition"} <= set(auto_names)
    assert set(backend.selected_api_names(cfg)) == set(auto_names)
    cfg.update(ui_start=True, start_x="975", start_y="972", window_title="Main Panel")
    assert not backend.validate(cfg, capture=True)


def test_desktop_three_modes_preserve_templates_and_single_selection():
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp
    root = tk.Tk()
    root.withdraw()
    try:
        app = DesktopApp(root)
        app.template.insert("end", "\nAPI 自定义要求")
        app.capture_mode.set("single_scene")
        app.change_capture_mode()
        cfg = app.config()
        assert "selected_apis" not in cfg
        assert len(cfg["selected"]) == 1
        assert cfg["selected"][0] in app.preview_text.get("1.0", "end")
        assert set(row["scenario_id"] for row in app.scene_choices.values()) == set(
            row["scenario_id"] for row in backend.scenarios())
        app.template.insert("end", "\n场景自定义要求")
        app.selected_scene.set(list(app.scene_choices)[1])
        app.update_scene_selection()
        assert app.config()["selected"] == [list(app.scene_choices.values())[1]["scenario_id"]]
        app.capture_mode.set("all_apis")
        app.change_capture_mode()
        assert len(app.config()["selected_apis"]) == app.available_api_count()
        assert "API 自定义要求" in app.template.get("1.0", "end")
        app.capture_mode.set("single_api")
        app.change_capture_mode()
        app.api_search.set("cnc_actf")
        app.listbox.selection_clear(0, "end")
        app.listbox.selection_set(app.visible_api_names.index("cnc_actf"))
        app.on_api_selection()
        assert app.config()["selected_apis"] == ["cnc_actf"]
        app.capture_mode.set("single_scene")
        app.change_capture_mode()
        assert "场景自定义要求" in app.template.get("1.0", "end")
    finally:
        root.destroy()


def test_detailed_log_keeps_history_position_when_new_lines_arrive():
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp

    root = tk.Tk()
    try:
        app = DesktopApp(root)
        root.geometry("1000x760")
        root.update()
        for index in range(160):
            app.append_log(f"history line {index}")
        app.log.yview_moveto(0.25)
        root.update()
        before = app.log.yview()[0]

        app.append_log("new live event")
        root.update()
        after = app.log.yview()[0]

        assert after == pytest.approx(before, abs=0.02)
        app.log.yview_moveto(1.0)
        root.update()
        app.append_log("follow tail event")
        root.update()
        assert app.log.yview()[1] >= 0.99
    finally:
        root.destroy()


def test_prompt_panel_keeps_actions_visible_and_text_scrollable():
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp
    root = tk.Tk()
    try:
        app = DesktopApp(root)
        app.tabs.select(app.scene_tab)
        for geometry in ("920x680", "1120x820"):
            root.geometry(geometry)
            for mode in ("single_api", "single_scene", "all_apis"):
                app.capture_mode.set(mode)
                app.change_capture_mode()
                root.update()
                bottom = app.scene_tab.winfo_rooty() + app.scene_tab.winfo_height()
                button = app.start_buttons[-1]
                assert button.winfo_height() >= button.winfo_reqheight()
                assert button.winfo_rooty() + button.winfo_height() <= bottom
                assert button.winfo_rootx() + button.winfo_width() <= app.scene_tab.winfo_rootx() + app.scene_tab.winfo_width()
                for widget in (app.template, app.preview_text):
                    assert widget.winfo_height() > 30
                    widget.yview_moveto(1.0)
                    root.update()
                    assert widget.yview()[1] >= 0.99
                assert app.template_scrollbar.winfo_ismapped()
                assert app.preview_scrollbar.winfo_ismapped()
    finally:
        root.destroy()


def test_unsupported_protocol_never_uses_focas_adapter():
    cfg = config()
    for protocol in ("ezsocket", "s7comm", "lsv2", "unknown"):
        cfg["protocol"] = protocol
        assert any("尚未接入" in error for error in backend.validate(cfg, capture=True))
        assert "FOCAS" in backend.render_selected_prompts(cfg)[0]
        with patch.object(backend.socket, "create_connection") as connect:
            with pytest.raises(ValueError, match="尚未接入"):
                backend.test_simulator(cfg)
            connect.assert_not_called()


def test_protocol_switch_inherits_focas_content_and_keeps_edits_independent():
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp
    from smart_traffic_agent.protocol_profiles import PROTOCOL_PROFILES
    root = tk.Tk()
    try:
        app = DesktopApp(root)
        app.tabs.select(app.scene_tab)
        app.values["host"].set("192.0.2.10")
        app.values["port"].set("8192")
        app.template.insert("end", "\n保留 FOCAS 要求")
        original_api = app.config()["selected_apis"]
        for key in ("ezsocket", "s7comm", "lsv2"):
            app.protocol_choice.set(PROTOCOL_PROFILES[key]["label"])
            app.change_protocol()
            root.update()
            assert app.config()["protocol"] == key
            assert app.config()["selected_apis"] == original_api
            assert app.config()["template"]
            assert app.values["host"].get() == "192.0.2.10"
            assert app.values["port"].get() == "8192"
            assert "保留 FOCAS 要求" in app.config()["full_template"]
            assert app.capture_pane.winfo_ismapped()
            assert not app.protocol_pending.winfo_ismapped()
            assert app.protocol_environment.winfo_manager() == "pack"
            assert app.start_buttons[0].instate(["disabled"])
            app.values["host"].set("192.0.2.20")
        app.protocol_choice.set(PROTOCOL_PROFILES["focas"]["label"])
        app.change_protocol()
        root.update()
        assert app.values["host"].get() == "192.0.2.10"
        assert app.values["port"].get() == "8192"
        assert app.config()["selected_apis"] == original_api
        assert "保留 FOCAS 要求" in app.config()["full_template"]
        assert app.capture_pane.winfo_ismapped()
        assert app.start_buttons[0].instate(["!disabled"])
        app.protocol_choice.set(PROTOCOL_PROFILES["ezsocket"]["label"])
        app.change_protocol()
        assert app.values["host"].get() == "192.0.2.20"
        app.capture_mode.set("single_scene")
        app.change_capture_mode()
        assert app.config()["selected"]
        assert "FOCAS" in app.preview_text.get("1.0", "end")
        app.capture_mode.set("all_apis")
        app.change_capture_mode()
        assert len(app.config()["selected_apis"]) == app.available_api_count()

    finally:
        root.destroy()


def test_coverage_progress_updates_after_failed_batch_with_captured_targets(tmp_path):
    from smart_traffic_agent.utils import write_json

    state_path = tmp_path / "full_api_coverage_state.json"
    state = {
        "schema_version": 1,
        "functions": {
            "cnc_acts": {
                "function": "cnc_acts", "status": "covered_success",
                "traffic_matched": True, "parameter_coverage_passed": True,
                "coverage_role": "target",
            },
            "cnc_rddynamic": {
                "function": "cnc_rddynamic", "status": "unplanned",
                "traffic_matched": False,
                "coverage_role": "target",
            },
        },
        "runs": [],
    }
    write_json(state_path, state)
    selected = {"cnc_acts", "cnc_rddynamic"}
    batch_dir = tmp_path / "batch_004"
    execution_dir = batch_dir / "execution"
    execution_dir.mkdir(parents=True)
    write_json(batch_dir / "summary.json", {
        "success": False, "api_log_count": 3, "capture_event_count": 6,
        "errors": ["API_PREREQUISITE_BLOCKED"],
        "function_coverage": {"parameter_coverage": {"functions": {
            "cnc_acts": {}, "cnc_rddynamic": {},
        }, "passed": True}},
    })
    (execution_dir / "api_logs.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"protocol_function": "cnc_acts", "status_code": 0, "traffic_status": "matched"},
        {"protocol_function": "cnc_statinfo", "status_code": 0, "traffic_status": "matched"},
        {"protocol_function": "cnc_download3", "status_code": 0, "traffic_status": "no_corresponding_traffic"},
    ]), encoding="utf-8")
    backend.write_selected_api_coverage_progress(
        tmp_path, state_path, selected, 4, "running", targets=selected,
    )
    running = json.loads((tmp_path / "coverage_progress.json").read_text(encoding="utf-8"))
    assert running["captured_function_count"] == 1
    assert running["missing_target_functions_this_batch"] == ["cnc_rddynamic"]
    assert running["overall_progress"]["observed_api_function_count"] == 3
    assert running["overall_progress"]["matched_api_function_count"] == 2
    assert running["overall_progress"]["api_log_count"] == 3
    assert running["support_api_progress"]["cnc_statinfo"]["successful_matched_call_count"] == 1
    assert running["support_api_progress"]["cnc_download3"]["matched_call_count"] == 0
    assert running["overall_progress"]["support_observed_function_count"] == 2
    assert len(running["batches"]) == 1

    state["functions"]["cnc_rddynamic"].update(
        status="covered_success", traffic_matched=True, parameter_coverage_passed=True,
    )
    write_json(state_path, state)
    backend.write_selected_api_coverage_progress(
        tmp_path, state_path, selected, 4, "failed", targets=selected,
        errors=["API_PREREQUISITE_BLOCKED"],
    )
    failed = json.loads((tmp_path / "coverage_progress.json").read_text(encoding="utf-8"))
    assert failed["captured_function_count"] == failed["resolved_function_count"] == 2
    assert failed["last_batch_status"] == "failed"
    assert failed["last_batch_errors"] == ["API_PREREQUISITE_BLOCKED"]
    assert failed["missing_target_functions_this_batch"] == []


def test_refresh_overall_collection_csv_includes_support_traffic(tmp_path):
    import csv
    from smart_traffic_agent.utils import write_json

    csv_path = tmp_path / "采集进度.csv"
    fields = ["category", "function", "中文功能说明", "collected", "status", "evidence_runs", "latest_status"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in ("cnc_acts", "cnc_rdseqnum", "cnc_unused"):
            writer.writerow({"category": "test", "function": name, "中文功能说明": "",
                             "collected": "", "status": "未采集", "evidence_runs": "", "latest_status": "unplanned"})
    run = tmp_path / "run_001"
    execution = run / "batch_001" / "execution"
    execution.mkdir(parents=True)
    (execution / "focas_capture.pcap").write_bytes(b"0" * 25)
    (execution / "api_logs.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"protocol_function": "cnc_acts", "status_code": 0, "traffic_status": "matched"},
        {"protocol_function": "cnc_rdseqnum", "status_code": 0, "traffic_status": "matched"},
        {"protocol_function": "cnc_unused", "status_code": 0, "traffic_status": "no_corresponding_traffic"},
    ]), encoding="utf-8")
    write_json(run / "full_api_coverage_state.json", {"functions": {
        "cnc_acts": {"status": "covered_success", "traffic_matched": True,
                     "parameter_coverage_passed": True},
    }})
    summary = backend.refresh_overall_collection_csv(csv_path)
    assert summary == {"function_count": 3, "collected_count": 2,
                       "fully_covered_count": 1, "traffic_observed_count": 1,
                       "local_success_count": 0}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["function"]: row for row in csv.DictReader(stream)}
    assert rows["cnc_acts"]["latest_status"] == "covered_success"
    assert rows["cnc_rdseqnum"]["latest_status"] == "traffic_observed"
    assert rows["cnc_unused"]["collected"] == ""
    assert rows["cnc_rdseqnum"]["evidence_runs"] == "run_001"
