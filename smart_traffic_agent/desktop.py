"""Native Tk desktop application for SMPAgent."""
from __future__ import annotations

import json
import csv
import math
import os
import re
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import ctypes
import webbrowser
import urllib.error
import urllib.request
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from ctypes import wintypes
from typing import Any

from .desktop_backend import (
    ROOT, EVENT, PROMPT, FULL_API_PROMPT, defaults, render_selected_prompts,
    function_manifest, scenarios, validate, atomic_scene_api_names, cycle_start_configuration_error, chinese_api_description,
    read_only_priority_functions, auto_no_parameter_functions,
)


from .protocol_profiles import PROTOCOL_PROFILES, protocol_profile
from .api_compatibility import detect_sdk_model, unavailable_reason
from .display_text import agent_label, format_event, localize_technical_text, stage_label
from .api_log_templates import api_payload
from .traffic_graph import build_default_relationship_service
from .knowledge import KnowledgeBase
from .heidenhain_catalog import heidenhain_tnc640_catalog
from .integrations.ezsocket import load_ezsocket_manifest


class DesktopApp:
    def __init__(self, root):
        self.root = root
        root.title("SMPAgent · 智能流量采集工作台")
        root.geometry("1120x820")
        root.minsize(920, 680)
        root.configure(bg="#eef2f6")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#26364b")
        style.configure("TLabelframe", background="#eef2f6")
        style.configure("TLabelframe.Label", background="#eef2f6", foreground="#176a83")
        style.configure("TButton", padding=(12, 7))
        style.configure("Alert.TLabel", background="#eef2f6", foreground="#d62828",
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#176a83")
        style.map("Accent.TButton", background=[("active", "#2188a4")])
        self.values = {k: tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=v)
                       for k, v in defaults().items() if k != "template"}
        self.protocol_choice = tk.StringVar(value=PROTOCOL_PROFILES["focas"]["label"])
        self.protocol_status = tk.StringVar(value="已接入 · 支持真实采集")
        self._active_protocol = "focas"
        self._protocol_connections = {}
        self.catalog = function_manifest()
        self.selected_api_set = {"cnc_statinfo"} if any(
            row["function"] == "cnc_statinfo" for row in self.catalog
        ) else ({self.catalog[0]["function"]} if self.catalog else set())
        self.capture_mode = tk.StringVar(value="single_api")
        self.scene_catalog = scenarios()
        scene_labels = {
            "coordinate_feed_motion": "坐标与进给运动", "spindle_control": "主轴控制",
            "program_lifecycle": "程序生命周期", "alarm_diagnosis": "报警与诊断",
            "parameter_macro_access": "参数与宏变量访问", "pmc_signal_monitoring": "PMC 信号监控",
            "tool_offset_management": "刀具与刀补管理", "work_coordinate_setting": "工件坐标设置",
            "ethernet_connection_exception": "以太网连接与异常", "general_status_collection": "基础状态采集",
        }
        self.scene_choices = {scene_labels.get(row["scenario_name"], row["scenario_name"]): row
                              for row in self.scene_catalog}
        self.selected_scene = tk.StringVar(value=next(iter(self.scene_choices), ""))
        self.scene_detail = tk.StringVar()
        self.visible_api_names = []
        self.api_search = tk.StringVar()
        self.api_category = tk.StringVar(value="全部分类")
        self.api_selected_count = tk.StringVar()
        self.api_detail = tk.StringVar(value="选中函数后可查看清单说明")
        self._header_cache_key = None
        self._header_cache_text = ""
        self._refreshing_api_list = False
        self.scene_template_text = PROMPT
        self.full_template_text = FULL_API_PROMPT
        self.template_mode = "full"
        self.events = queue.Queue()
        self.process = None
        self.busy = False
        self.cancelled = False
        self._ignore_active_capture_events = False
        self._terminal_capture_event = False
        self.verified = {}
        self.output = None
        self.started = 0
        self.action = ""
        self.snapshot = {}
        self.status = tk.StringVar(value="请测试仿真器与模型连接，然后选择采集模式与内容")
        self.sim_status = tk.StringVar(value="尚未测试")
        self.llm_status = tk.StringVar(value="尚未测试")
        self.metrics = tk.StringVar(value="耗时 00:00   |   尚无采集文件")
        self.scene_status = tk.StringVar(value="未开始")
        self.final_capture_status = tk.StringVar(value="")
        self.plan_target = tk.StringVar(value="尚未生成")
        self.plan_steps_count = tk.StringVar(value="—")
        self.plan_chunks_count = tk.StringVar(value="—")
        self.plan_nc_required = tk.StringVar(value="—")
        self.plan_badge_text = tk.StringVar(value="等待规划")
        self.plan_badge_color = "#64748b"
        self.current_plan_target = ""
        self.graph_search = tk.StringVar()
        self.graph_status = tk.StringVar(value="正在准备关系图")
        self.graph_service = build_default_relationship_service()
        self.knowledge_status = tk.StringVar(value="正在检查知识库连接…")
        self.ezsocket_test_mode = tk.StringVar(value="tcp")
        self.ezsocket_com_machine = tk.StringVar(value=PROTOCOL_PROFILES["ezsocket"].get("host", ""))
        self._knowledge_warning_shown = False
        self.graph_querying = False
        self.graph_zoom = 1.0
        self.graph_data = None
        self.graph_node_items = {}
        self.graph_pan_origin = None
        self.graph_positions = {}
        self.graph_drag_node = None
        self.graph_drag_last_position = None
        self.graph_click_origin = None
        self.graph_detail_tree = None
        self.graph_detail_values = {}
        self.calibration_status = tk.StringVar(value="尚未定位 Cycle Start")
        self.calibration_overlay = None
        self.operator_prompt = None
        self.operator_prompt_label = None
        self.operator_prompt_timer = None
        header = tk.Frame(root, bg="#142d44", padx=26, pady=17)
        header.pack(fill="x")
        tk.Label(header, text="SMPAgent", font=("Segoe UI", 23, "bold"), bg="#142d44", fg="white").pack(anchor="w")
        tk.Label(header, text="选择协议  /  连接设备  /  配置采集  /  监控结果", bg="#142d44", fg="#a8c7d9", font=("Microsoft YaHei UI", 10)).pack(anchor="w")
        # Persistent action bar: it stays visible whichever tab is selected and
        # at every supported window size.
        action_bar = tk.Frame(root, bg="#dce7ee", padx=18, pady=9)
        action_bar.pack(fill="x")
        tk.Label(action_bar, textvariable=self.status, bg="#dce7ee", fg="#d62828",
                 font=("Microsoft YaHei UI", 10, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        self.start_buttons = []
        self.stop_buttons = []
        top_stop = ttk.Button(action_bar, text="停止采集", command=self.stop, state="disabled")
        top_stop.pack(side="right", padx=(8, 0))
        self.stop_buttons.append(top_stop)
        top_start = ttk.Button(action_bar, text="▶ 开始采集", style="Accent.TButton", command=self.start)
        top_start.pack(side="right")
        self.start_buttons.append(top_start)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=18, pady=12)
        self.connection = ttk.Frame(self.tabs)
        # The connection page contains more controls than a small laptop
        # viewport can display. Keep the page itself scrollable so the
        # preflight controls are never clipped by the window height.
        self.connection_canvas = tk.Canvas(self.connection, highlightthickness=0,
                                           bg="#eef2f6")
        self.connection_scrollbar = ttk.Scrollbar(
            self.connection, orient="vertical",
            command=self.connection_canvas.yview,
        )
        self.connection_canvas.configure(
            yscrollcommand=self.connection_scrollbar.set,
        )
        self.connection_canvas.grid(row=0, column=0, sticky="nsew")
        self.connection_scrollbar.grid(row=0, column=1, sticky="ns")
        self.connection.rowconfigure(0, weight=1)
        self.connection.columnconfigure(0, weight=1)
        self.connection_body = ttk.Frame(self.connection_canvas, padding=16)
        self.connection_window = self.connection_canvas.create_window(
            (0, 0), window=self.connection_body, anchor="nw",
        )
        self.connection_body.bind(
            "<Configure>",
            lambda _event: self.connection_canvas.configure(
                scrollregion=self.connection_canvas.bbox("all")
            ),
        )
        self.connection_canvas.bind(
            "<Configure>",
            lambda event: self.connection_canvas.itemconfigure(
                self.connection_window, width=event.width
            ),
        )
        self.scene_tab = ttk.Frame(self.tabs, padding=16)
        self.monitor = ttk.Frame(self.tabs, padding=16)
        self.graph_tab = ttk.Frame(self.tabs, padding=16)
        for panel, name in ((self.connection, "01  连接配置"), (self.scene_tab, "02  采集模式与提示词"), (self.monitor, "03  采集监控"), (self.graph_tab, "04  流量关系图")):
            self.tabs.add(panel, text=name)
        self.build_connections()
        self.build_scenes()
        self.build_monitor()
        self.build_graph()
        for value in self.values.values():
            value.trace_add("write", self.changed)
        self.values["header"].trace_add("write", self.change_controller_model)
        # Restore the most recent evidence as soon as the monitor exists.  A
        # fresh desktop process otherwise has no event stream to populate the
        # two monitor tabs until the user edits a setting.
        self.load_history_for_config(self.config())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(150, self.poll)
        root.after(350, self.refresh_graph)
        root.after(450, self.refresh_knowledge_status)
        self.preview()

    def field(self, parent, row, label, key, choices=None, secret=False, browse=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        if choices:
            widget = ttk.Combobox(parent, textvariable=self.values[key], values=choices)
        else:
            widget = ttk.Entry(parent, textvariable=self.values[key], show="•" if secret else "")
        widget.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        parent.columnconfigure(1, weight=1)
        if browse:
            ttk.Button(parent, text="选择…", command=lambda: self.choose_path(key)).grid(row=row, column=2, padx=6)
        return widget

    def choose_path(self, key):
        path = filedialog.askopenfilename() if key == "bridge" else filedialog.askdirectory()
        if path:
            self.values[key].set(path)

    def refresh_models(self):
        if self.busy:
            return
        base_url = self.values["base_url"].get().strip().rstrip("/")
        api_key = self.values["api_key"].get().strip()
        if not base_url or not api_key:
            messagebox.showerror("无法获取模型", "请先填写 Base URL 和 API Key")
            return
        self.model_refresh_button.configure(state="disabled")
        self.llm_status.set("正在获取可用模型列表…")
        try:
            timeout = max(1.0, min(float(self.values["timeout"].get()), 60.0))
        except ValueError:
            timeout = 30.0

        def fetch():
            request = urllib.request.Request(
                base_url + "/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                models = sorted({
                    str(row.get("id", "")).strip()
                    for row in rows if isinstance(row, dict) and row.get("id")
                }, key=str.casefold)
                if not models:
                    raise ValueError("服务端返回成功，但响应中没有模型 ID")
                self.events.put(("models", {"models": models}))
            except urllib.error.HTTPError as exc:
                self.events.put(("models", {"error": f"HTTP {exc.code}：{exc.reason}"}))
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                self.events.put(("models", {"error": str(exc)}))
            except Exception as exc:
                self.events.put(("models", {"error": f"{type(exc).__name__}: {exc}"}))

        threading.Thread(target=fetch, daemon=True).start()

    def build_connections(self):
        parent = self.connection_body
        protocol_row = ttk.LabelFrame(parent, text="采集协议", padding=8)
        protocol_row.pack(fill="x", pady=(0, 10))
        self.protocol_combo = ttk.Combobox(protocol_row, textvariable=self.protocol_choice,
            values=[p["label"] for p in PROTOCOL_PROFILES.values()], state="readonly", width=20)
        self.protocol_combo.pack(side="left", padx=6)
        self.protocol_combo.bind("<<ComboboxSelected>>", self.change_protocol)
        ttk.Label(protocol_row, textvariable=self.protocol_status, wraplength=530).pack(side="left", padx=12)
        columns = ttk.Frame(parent)
        columns.pack(fill="x")
        columns.columnconfigure((0, 1), weight=1)
        self.device_frame = sim = ttk.LabelFrame(columns, text="目标设备 · FANUC CNC / NCGuide", padding=10)
        sim.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        llm = ttk.LabelFrame(columns, text="大模型 · 本地配置已填入", padding=10)
        llm.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.field(sim, 0, "地址 / IP", "host", ["127.0.0.1", "localhost"])
        self.field(sim, 1, "服务端口", "port")
        self.device_label = ttk.Label(sim, text="抓包网卡")
        self.device_label.grid(row=2, column=0, sticky="w", padx=8, pady=5)
        self.device_entry = ttk.Combobox(
            sim, textvariable=self.values["device"], values=[r"\Device\NPF_Loopback"]
        )
        self.device_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=5)
        sim.columnconfigure(1, weight=1)
        self.sdk_model_label = ttk.Label(sim, text="SDK 自动识别型号")
        self.sdk_model_label.grid(row=5, column=0, sticky="w", padx=8)
        self.sdk_model_value = ttk.Label(sim, textvariable=self.values["controller_model"], wraplength=260)
        self.sdk_model_value.grid(row=5, column=1, sticky="w", padx=8)
        self.sdk_help_label = ttk.Label(sim, text="更换 SDK 目录后自动更新型号和可选 API。", wraplength=360)
        self.sdk_help_label.grid(row=6, column=0, columnspan=3, sticky="w", padx=8)
        self.ezsocket_hint = ttk.Label(
            sim,
            text="EZSocket 连接测试为只读 TCP 探测，不会向仿真器写入指令。",
            wraplength=360,
        )
        self.device_test_button = ttk.Button(sim, text="测试设备连接", command=self.test_device_connection)
        self.device_test_button.grid(row=3, column=1, sticky="w", padx=8, pady=12)
        ttk.Label(sim, textvariable=self.sim_status, wraplength=400, style="Alert.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", padx=8)
        self.field(llm, 0, "Base URL", "base_url", ["https://api.deepseek.com"])
        self.model_combo = self.field(llm, 1, "模型", "model", [self.values["model"].get()])
        self.model_refresh_button = ttk.Button(
            llm, text="获取模型列表", command=self.refresh_models,
        )
        self.model_refresh_button.grid(row=1, column=2, padx=(0, 8), pady=5)
        self.field(llm, 2, "API Key（DEEPSEEK_API_KEY）", "api_key", secret=True)
        self.field(llm, 3, "接口", "wire_api", ["chat_completions", "responses"])
        self.field(llm, 4, "超时 / 秒", "timeout")
        ttk.Button(llm, text="测试模型调用", command=lambda: self.launch("llm")).grid(row=5, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(llm, textvariable=self.llm_status, wraplength=400, style="Alert.TLabel").grid(row=6, column=0, columnspan=3, sticky="w", padx=8)
        self.protocol_environment = advanced = ttk.LabelFrame(parent, text="FOCAS 专用环境", padding=8)
        advanced.pack(fill="x", pady=12)
        for row, (label, key) in enumerate((("FOCAS DLL 目录", "dll"), ("SDK 头文件目录", "header"),
                                           ("连接测试桥接程序", "bridge"), ("RAG 向量库", "vector"))):
            self.field(advanced, row, label, key, browse=True)
        ttk.Label(advanced, textvariable=self.knowledge_status, wraplength=760,
                  style="Alert.TLabel").grid(row=4, column=0, columnspan=3,
                                             sticky="w", padx=8, pady=(6, 2))
        self.ezsocket_environment = ttk.LabelFrame(parent, text="EZSocket 连接环境", padding=8)
        ttk.Label(
            self.ezsocket_environment,
            text="连接测试使用 TCP 建连探测，不需要 FOCAS SDK、Npcap 或抓包网卡。",
            wraplength=760,
        ).pack(anchor="w", padx=8, pady=4)
        ezsocket_machine = ttk.Frame(self.ezsocket_environment)
        ezsocket_machine.pack(fill="x", padx=8, pady=(2, 4))
        ttk.Label(ezsocket_machine, text="COM 远程主机（可空）").pack(side="left")
        ttk.Entry(ezsocket_machine, textvariable=self.ezsocket_com_machine, width=24).pack(side="left", padx=8)
        ttk.Label(ezsocket_machine, text="默认填写仿真器虚拟机 IP；留空表示本机 COM。", foreground="#64748b").pack(side="left")
        ezsocket_actions = ttk.Frame(self.ezsocket_environment)
        ezsocket_actions.pack(fill="x", padx=8, pady=(4, 8))
        self.ezsocket_api_test_button = ttk.Button(
            ezsocket_actions,
            text="测试 Status_GetRunStatus（COM）",
            command=self.test_ezsocket_com,
        )
        self.ezsocket_api_test_button.pack(side="left")
        ttk.Label(
            ezsocket_actions,
            text="需要本机或虚拟机已注册 EZSocket COM；仅调用只读 API。",
            foreground="#64748b",
            wraplength=520,
        ).pack(side="left", padx=10)
        self.output_environment = common = ttk.LabelFrame(parent, text="采集输出", padding=8)
        common.pack(fill="x", pady=(0, 12))
        self.field(common, 0, "采集输出目录", "output", browse=True)
        self.field(common, 1, "续采运行目录（可选）", "resume_from", browse=True)

    def change_protocol(self, _event=None):
        if self.busy:
            self.protocol_choice.set(PROTOCOL_PROFILES[self._active_protocol]["label"])
            return
        key = next(k for k, profile in PROTOCOL_PROFILES.items()
                   if profile["label"] == self.protocol_choice.get())
        self.config()  # Save the currently edited template before switching.
        self._protocol_connections[self._active_protocol] = {
            "values": {name: value.get() for name, value in self.values.items() if name != "protocol"},
            "selected_apis": set(self.selected_api_set), "mode": self.capture_mode.get(),
            "scene": self.selected_scene.get(), "scene_template": self.scene_template_text,
            "full_template": self.full_template_text,
        }
        # A new protocol starts as a full copy of the current FOCAS workspace.
        saved = self._protocol_connections.get(key, self._protocol_connections["focas"])
        has_protocol_snapshot = key in self._protocol_connections
        self._active_protocol = key
        profile = PROTOCOL_PROFILES[key]
        self._switching_protocol = True
        try:
            self.values["protocol"].set(key)
            for name, value in saved["values"].items():
                self.values[name].set(value)
            # Apply a protocol's known endpoint only when the source still
            # has the untouched FOCAS defaults. Explicit user edits remain
            # intact when switching protocols.
            source_defaults = defaults()
            source_is_default_endpoint = (
                saved["values"].get("host") == source_defaults.get("host")
                and saved["values"].get("port") == source_defaults.get("port")
            )
            if not has_protocol_snapshot and profile.get("port") and source_is_default_endpoint:
                self.values["port"].set(profile["port"])
            if not has_protocol_snapshot and profile.get("host") and (
                key == "lsv2" or source_is_default_endpoint
            ):
                self.values["host"].set(profile["host"])
            if not has_protocol_snapshot and key == "lsv2":
                # Keep the LSV-2 store separate from the FOCAS index.  The
                # path is absolute so launching the desktop from another
                # working directory cannot silently reopen FOCAS knowledge.
                self.values["vector"].set(str(ROOT / "rag_indexes" / "heidenhain" / "vector_db"))
            if not has_protocol_snapshot and key == "ezsocket":
                # Mitsubishi API/manual chunks are kept in a protocol-local
                # index; do not accidentally query the FOCAS collection.
                self.values["vector"].set(str(ROOT / "rag_indexes" / "ezsocket" / "vector_db"))
            if not has_protocol_snapshot:
                for name in ("samples", "interval"):
                    if profile.get(name) and name in self.values:
                        self.values[name].set(profile[name])
            if not has_protocol_snapshot:
                self.selected_api_set = (
                    {"directory_info"} if key == "lsv2" else
                    {"Status_GetRunStatus"} if key == "ezsocket" else
                    set(saved["selected_apis"])
                )
                self.api_category.set("全部分类")
            elif key != "lsv2":
                self.selected_api_set = set(saved["selected_apis"])
            self.capture_mode.set("all_apis" if saved["mode"] == "single_scene" else saved["mode"])
            self.selected_scene.set(saved["scene"])
            self.scene_template_text = saved["scene_template"]
            self.full_template_text = saved["full_template"]
            self.template_mode = "scene" if saved["mode"] == "single_scene" else "full"
            self.template.delete("1.0", "end")
            self.template.insert("1.0", self.scene_template_text if self.template_mode == "scene" else self.full_template_text)
        finally:
            self._switching_protocol = False
        available = profile["available"]
        connection_testable = profile.get("connection_testable", available)
        capture_available = profile.get("capture_available", available)
        self.device_frame.configure(text="目标设备 · " + profile["target"])
        if capture_available:
            status = "已接入 · 支持真实采集"
        elif connection_testable:
            status = "已接入连接测试 · 只读 TCP 探测可用 · API/报文采集适配器待接入"
        else:
            status = "已继承 FOCAS 全部配置与模板 · 协议适配器待接入"
        self.protocol_status.set(status)
        self.protocol_environment.configure(
            text=("FOCAS 专用环境" if key == "focas" else
                  ("LSV2 只读采集环境" if key == "lsv2" else "协议运行环境"))
        )
        if key == "ezsocket":
            self.device_label.configure(text="抓包网卡（连接测试不需要）")
            self.sdk_model_label.grid_remove()
            self.sdk_model_value.grid_remove()
            self.sdk_help_label.grid_remove()
            self.ezsocket_hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=6)
            self.protocol_environment.pack_forget()
            self.ezsocket_environment.pack(fill="x", pady=12, before=self.output_environment)
            self.api_picker_title.set("EZSocket API · 选择函数查看知识库说明")
        else:
            self.device_label.configure(text="抓包网卡")
            self.sdk_model_label.grid()
            self.sdk_model_value.grid()
            self.sdk_help_label.grid()
            self.ezsocket_hint.grid_remove()
            self.ezsocket_environment.pack_forget()
            self.protocol_environment.pack(fill="x", pady=12, before=self.output_environment)
            self.api_picker_title.set(
                "LSV/2 · 先选只读操作" if key == "lsv2" else "FOCAS API · 先选分类，再选择一个函数"
            )
        self.device_test_button.configure(state="normal" if connection_testable else "disabled")
        simulator_verified = self.verified.get("simulator") == self.signature(self.config(), "simulator")
        capture_enabled = capture_available and (key == "focas" or simulator_verified)
        for button in self.start_buttons:
            button.configure(state="normal" if capture_enabled else "disabled")
        show_api_catalog = key == "ezsocket"
        # Keep the shared API/template pane mounted while switching protocols.
        # This preserves the operator's working context and lets unsupported
        # protocols inherit the FOCAS editor until their adapter is added;
        # connection and capture buttons remain disabled according to the
        # profile above.  The old replacement pane caused Tk to discard the
        # layout state and made switching back lose the editor geometry.
        self.protocol_pending.pack_forget()
        self.capture_pane.pack(fill="both", expand=True)
        self.refresh_api_list()
        self.change_capture_mode()
        self.sim_status.set("尚未测试" if connection_testable else "配置继承自 FOCAS；协议连接测试待接入")
        self.status.set(f"当前协议：{profile['label']} · " + (
            "请选择采集内容" if capture_available else
            ("可查看 API 并测试只读 COM 调用；报文采集适配器待接入" if show_api_catalog else "可测试连接；真实采集适配器待接入")
        ))
        if hasattr(self, "ezsocket_api_test_button"):
            self.ezsocket_api_test_button.configure(state="normal" if key == "ezsocket" and connection_testable else "disabled")
        self.verified.pop("simulator", None)
        if self.verified.get("llm") != self.signature(self.config(), "llm"):
            self.llm_status.set("配置已变化，请测试连接")
        self.preview()

    def build_scenes(self):
        self.protocol_pending_text = tk.StringVar()
        self.protocol_pending = ttk.Frame(self.scene_tab)
        ttk.Label(self.protocol_pending, textvariable=self.protocol_pending_text, wraplength=760,
                  font=("Microsoft YaHei UI", 12)).pack(anchor="nw")
        self.capture_pane = pane = ttk.Panedwindow(self.scene_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left, right = ttk.Frame(pane), ttk.Frame(pane)
        pane.add(left, weight=2)
        pane.add(right, weight=3)
        # The API list, details and Cycle Start controls can exceed the height
        # of a laptop window. Scroll the whole left column, not just the list.
        self.api_canvas = tk.Canvas(left, highlightthickness=0, bg="#eef2f6")
        self.api_scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.api_canvas.yview)
        self.api_canvas.configure(yscrollcommand=self.api_scrollbar.set)
        self.api_canvas.pack(side="left", fill="both", expand=True)
        self.api_scrollbar.pack(side="right", fill="y")
        left_body = ttk.Frame(self.api_canvas)
        self.api_canvas_window = self.api_canvas.create_window((0, 0), window=left_body, anchor="nw")
        left_body.bind("<Configure>", lambda _event: self.api_canvas.configure(
            scrollregion=self.api_canvas.bbox("all")
        ))
        self.api_canvas.bind("<Configure>", lambda event: self.api_canvas.itemconfigure(
            self.api_canvas_window, width=event.width
        ))
        self.api_canvas.bind("<MouseWheel>", lambda event: self.api_canvas.yview_scroll(
            -int(event.delta / 120), "units"
        ))
        left = left_body
        mode_row = ttk.LabelFrame(left, text="采集模式")
        mode_row.pack(fill="x", padx=(0, 12), pady=(0, 10))
        for label, mode in (("单个 API", "single_api"), ("全部 API", "all_apis"),
                            ("AUTO 无参数", "auto_no_parameter"),
                            ("只读优先", "read_only_priority")):
            ttk.Radiobutton(mode_row, text=label, value=mode, variable=self.capture_mode,
                            command=self.change_capture_mode).pack(side="left", padx=6, pady=8)
        self.scene_picker = ttk.Frame(left)
        ttk.Label(self.scene_picker, text="批次场景分类").pack(anchor="w")
        self.scene_combo = scene_combo = ttk.Combobox(self.scene_picker, textvariable=self.selected_scene,
                                   values=list(self.scene_choices), state="readonly")
        scene_combo.pack(fill="x", pady=6)
        scene_combo.bind("<<ComboboxSelected>>", lambda _: self.update_scene_selection())
        ttk.Label(self.scene_picker, textvariable=self.scene_detail, wraplength=390).pack(anchor="w", pady=6)
        self.all_api_info = ttk.Label(left, text=f"将按函数清单分批采集全部 {len(self.catalog)} 个 API。", wraplength=390)
        self.readonly_info = ttk.Label(
            left,
            text=(f"将按知识库前置条件分批采集无需修改 CNC 参数、无需切换模式的只读 API"
                  f"（当前 {len(read_only_priority_functions())} 个）。"),
            wraplength=390,
        )
        self.auto_info = ttk.Label(
            left,
            text=(f"将采集 AUTO/MEM 模式可用、且无 CNC 参数前置的 API"
                  f"（含写/控制类和需要运行 NC 程序的 API，当前 {len(auto_no_parameter_functions())} 个）。"),
            wraplength=390,
        )
        main_left = left
        self.api_picker = ttk.Frame(left)
        self.api_picker.pack(fill="x")
        left = self.api_picker
        self.api_picker_title = tk.StringVar(value="FOCAS API · 先选分类，再选择一个函数")
        ttk.Label(left, textvariable=self.api_picker_title).pack(anchor="w", pady=(0, 8))
        categories = ["全部分类", *sorted({
            str(row.get("category", "")).strip() or "未分类" for row in self.catalog
        }, key=str.casefold)]
        category_row = ttk.Frame(left)
        category_row.pack(fill="x", padx=(0, 12), pady=(0, 5))
        ttk.Label(category_row, text="API 分类").pack(side="left")
        self.category_combo = ttk.Combobox(
            category_row, textvariable=self.api_category, values=categories, state="readonly",
        )
        self.category_combo.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.api_category.trace_add("write", lambda *_: self.refresh_api_list())
        search = ttk.Frame(left)
        search.pack(fill="x", padx=(0, 12))
        ttk.Label(search, text="搜索函数 / 说明").pack(side="left")
        ttk.Entry(search, textvariable=self.api_search).pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.api_search.trace_add("write", lambda *_: self.refresh_api_list())
        selection_row = ttk.Frame(left)
        selection_row.pack(fill="x", padx=(0, 12), pady=(5, 0))
        ttk.Label(selection_row, textvariable=self.api_selected_count).pack(side="left")
        self.clear_api_button = ttk.Button(selection_row, text="取消选择", command=self.clear_api_selection)
        self.clear_api_button.pack(side="right")
        api_list = ttk.Frame(left)
        api_list.pack(fill="x", padx=(0, 12))
        self.listbox = tk.Listbox(api_list, selectmode="browse", exportselection=False,
                                 font=("Microsoft YaHei UI", 10), bg="white",
                                 selectbackground="#176a83", relief="flat", height=6)
        self.listbox.pack(side="left", fill="x", expand=True, pady=7)
        api_scroll = ttk.Scrollbar(api_list, orient="vertical", command=self.listbox.yview)
        api_scroll.pack(side="right", fill="y", pady=7)
        self.listbox.configure(yscrollcommand=api_scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self.on_api_selection)
        self.listbox.bind("<Button-1>", self.on_api_click)
        self.refresh_api_list()
        ttk.Label(left, textvariable=self.api_detail, wraplength=410,
                  foreground="#64748b").pack(anchor="w", padx=(0, 12), pady=(0, 6))
        left = main_left
        self.capture_settings = settings = ttk.Frame(left)
        settings.pack(fill="x", padx=(0, 12), before=self.api_picker)
        self.field(settings, 0, "采样次数参考值", "samples")
        self.field(settings, 1, "间隔参考值 / 秒", "interval")
        self.field(settings, 2, "补充要求", "requirements")
        ttk.Checkbutton(settings, text="采集需要执行 NC 程序时启用 Cycle Start", variable=self.values["ui_start"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        self.field(settings, 4, "Cycle Start X", "start_x")
        self.field(settings, 5, "Cycle Start Y", "start_y")
        self.field(settings, 6, "窗口标题", "window_title")
        locate = ttk.Button(settings, text="开始定位 Cycle Start", command=self.begin_cycle_start_calibration)
        locate.grid(row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        self.locate_cycle_start_button = locate
        ttk.Label(settings, textvariable=self.calibration_status, style="Alert.TLabel", wraplength=360).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6)
        )
        self.template_title = tk.StringVar(value="API 采集提示词模板")
        self.template_help = tk.StringVar(value="$manifest / $function_count / $selected_functions / $coverage_scope\n$host / $port / $device / $samples / $interval / $requirements")
        # Only the two text rows shrink; actions always retain their requested height.
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=3, uniform="prompt_text")
        right.rowconfigure(5, weight=2, uniform="prompt_text")
        ttk.Label(right, textvariable=self.template_title).grid(row=0, column=0, sticky="w")
        ttk.Label(right, textvariable=self.template_help, foreground="#64748b", wraplength=440).grid(
            row=1, column=0, sticky="w", pady=6)
        editor = ttk.Frame(right)
        editor.grid(row=2, column=0, sticky="nsew")
        editor.rowconfigure(0, weight=1)
        editor.columnconfigure(0, weight=1)
        self.template = tk.Text(editor, height=1, width=1, wrap="word", font=("Microsoft YaHei UI", 10), relief="flat", undo=True)
        self.template.insert("1.0", FULL_API_PROMPT)
        self.template.grid(row=0, column=0, sticky="nsew")
        self.template_scrollbar = ttk.Scrollbar(editor, command=self.template.yview)
        self.template_scrollbar.grid(row=0, column=1, sticky="ns")
        self.template.configure(yscrollcommand=self.template_scrollbar.set)
        self.template.bind("<KeyRelease>", lambda _: self.preview())
        ttk.Button(right, text="恢复默认模板", command=self.reset_template).grid(row=3, column=0, sticky="e", pady=5)
        self.preview_label = ttk.Label(right, text="")
        self.preview_label.grid(row=4, column=0, sticky="w")
        preview = ttk.Frame(right)
        preview.grid(row=5, column=0, sticky="nsew", pady=6)
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.preview_text = tk.Text(preview, height=1, width=1, wrap="word", font=("Microsoft YaHei UI", 9), bg="#e2eaf0", relief="flat", state="disabled")
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        self.preview_scrollbar = ttk.Scrollbar(preview, command=self.preview_text.yview)
        self.preview_scrollbar.grid(row=0, column=1, sticky="ns")
        self.preview_text.configure(yscrollcommand=self.preview_scrollbar.set)
        scene_actions = ttk.Frame(right)
        scene_actions.grid(row=6, column=0, sticky="ew", pady=(5, 0))
        scene_actions.columnconfigure(0, weight=1)
        ttk.Label(scene_actions, text="连接测试通过后可启动。", foreground="#64748b", wraplength=180).grid(row=0, column=0, sticky="w")
        scene_start = ttk.Button(scene_actions, text="▶ 开始采集所选内容", style="Accent.TButton", command=self.start)
        scene_start.grid(row=0, column=1, sticky="e")
        self.start_buttons.append(scene_start)
        self.bind_api_panel_wheel(left)

    def bind_api_panel_wheel(self, widget):
        if widget not in (self.listbox, self.category_combo, self.scene_combo):
            widget.bind("<MouseWheel>", self.scroll_api_panel, add="+")
        for child in widget.winfo_children():
            self.bind_api_panel_wheel(child)

    def scroll_api_panel(self, event):
        if event.delta:
            self.api_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def build_monitor(self):
        # The monitor contains several vertically stacked sections. Keep the
        # whole view scrollable so the plan card remains reachable on short
        # windows, while the plan and log panes retain their own scrollbars.
        monitor_scroll_host = ttk.Frame(self.monitor)
        monitor_scroll_host.pack(fill="both", expand=True)
        self.monitor_canvas = tk.Canvas(monitor_scroll_host, highlightthickness=0,
                                        borderwidth=0, background="#eef2f6")
        self.monitor_scrollbar = ttk.Scrollbar(
            monitor_scroll_host, orient="vertical", command=self.monitor_canvas.yview
        )
        self.monitor_canvas.configure(yscrollcommand=self.monitor_scrollbar.set)
        self.monitor_canvas.pack(side="left", fill="both", expand=True)
        self.monitor_scrollbar.pack(side="right", fill="y")
        monitor_body = ttk.Frame(self.monitor_canvas, padding=0)
        self.monitor_body = monitor_body
        self.monitor_canvas_window = self.monitor_canvas.create_window(
            (0, 0), window=monitor_body, anchor="nw"
        )
        monitor_body.bind(
            "<Configure>",
            lambda _event: self.monitor_canvas.configure(
                scrollregion=self.monitor_canvas.bbox("all")
            ),
        )
        self.monitor_canvas.bind(
            "<Configure>",
            lambda event: self.monitor_canvas.itemconfigure(
                self.monitor_canvas_window, width=event.width
            ),
        )
        self.monitor_canvas.bind(
            "<MouseWheel>", lambda event: self._scroll_monitor(event), add="+"
        )
        self.monitor_canvas.bind(
            "<Button-4>", lambda event: self._scroll_monitor(event), add="+"
        )
        self.monitor_canvas.bind(
            "<Button-5>", lambda event: self._scroll_monitor(event), add="+"
        )

        monitor_head = ttk.Frame(monitor_body)
        monitor_head.pack(fill="x")
        ttk.Label(monitor_head, textvariable=self.scene_status, font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        monitor_stop = ttk.Button(monitor_head, text="停止采集", command=self.stop, state="disabled")
        monitor_stop.pack(side="right")
        self.stop_buttons.append(monitor_stop)
        self.progress = ttk.Progressbar(monitor_body, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=12)
        ttk.Label(monitor_body, textvariable=self.metrics).pack(anchor="w", pady=(0, 10))
        self.monitor_stage = tk.StringVar(value="等待开始采集")
        ttk.Label(monitor_body, textvariable=self.monitor_stage, wraplength=850).pack(anchor="w", pady=(0, 10))
        self.final_capture_banner = tk.Label(
            monitor_body,
            textvariable=self.final_capture_status,
            anchor="w",
            justify="left",
            padx=12,
            pady=8,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.final_capture_banner.pack(fill="x", pady=(0, 8))
        self.failure_reason = tk.StringVar(value="失败原因将在采集失败后显示。")
        self.failure_banner = tk.Label(
            monitor_body,
            textvariable=self.failure_reason,
            anchor="w",
            justify="left",
            wraplength=1040,
            bg="#fff1f2",
            fg="#9f1239",
            padx=12,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.failure_banner.pack(fill="x", pady=(0, 8))
        # Reserve a real footer before the expanding notebook.  When the
        # notebook was packed first, small windows compressed this button to
        # a thin blank strip at the lower-right corner.
        monitor_footer = ttk.Frame(monitor_body)
        monitor_footer.pack(fill="x", side="bottom", pady=(8, 0))
        self.open_output_button = ttk.Button(
            monitor_footer, text="打开本次输出目录", command=self.open_output
        )
        self.open_output_button.pack(side="right", ipadx=8, ipady=3)
        # Keep the plan summary in the primary monitor view.  The detailed
        # log remains the expanding area below it, so a short window cannot
        # push the plan steps below the visible region.
        self.monitor_views = None
        self.build_plan_summary()
        details = ttk.Frame(monitor_body, padding=6, height=360)
        details.pack(fill="both", expand=False)
        details.pack_propagate(False)
        ttk.Label(
            details,
            text="详细日志（完整显示 Agent、工具、API 参数、返回码、流量和错误证据）",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="top", anchor="w", padx=8, pady=(2, 4))
        self.monitor_events = None
        self.monitor_flow = None
        self.log = tk.Text(
            details,
            bg="#142333",
            fg="#d6e5ee",
            font=("Consolas", 10),
            wrap="none",
            relief="flat",
            state="disabled",
            undo=False,
        )
        self.log.tag_configure("alert", foreground="#ff5c5c", font=("Consolas", 10, "bold"))
        self.details_hint = tk.StringVar(value="尚未加载运行记录；开始采集后这里会显示完整原始日志。")
        ttk.Label(details, textvariable=self.details_hint, foreground="#64748b").pack(
            side="top", anchor="w", padx=8, pady=(6, 4)
        )
        scrollbar = ttk.Scrollbar(details, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        horizontal_scrollbar = ttk.Scrollbar(details, orient="horizontal", command=self.log.xview)
        horizontal_scrollbar.pack(side="bottom", fill="x")
        self.log.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal_scrollbar.set)
        self.log.pack(fill="both", expand=True)
        # Bind the wheel explicitly because the surrounding notebook/panes
        # can otherwise consume the event before it reaches the Text widget.
        self.log.bind("<MouseWheel>", self._scroll_detail_log, add="+")
        self.log.bind("<Button-4>", self._scroll_detail_log, add="+")
        self.log.bind("<Button-5>", self._scroll_detail_log, add="+")
        self.log.bind("<Prior>", self._scroll_detail_log, add="+")
        self.log.bind("<Next>", self._scroll_detail_log, add="+")
        self.log.bind("<Home>", self._scroll_detail_log, add="+")
        self.log.bind("<End>", self._scroll_detail_log, add="+")
        self.log_placeholder = "详细日志将在采集开始后显示。"
        self.log.configure(state="normal")
        self.log.insert("1.0", self.log_placeholder)
        self.log.configure(state="disabled")

        # Retain the status variable for existing event handling and reports;
        # it is no longer rendered as a separate tab.
        self.flow_hint = tk.StringVar(value="调用记录会直接写入详细日志。")

    def _scroll_monitor(self, event):
        """Scroll the monitor page without stealing events from inner panes."""
        num = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if num == 4 or delta > 0:
            self.monitor_canvas.yview_scroll(-3, "units")
        elif num == 5 or delta < 0:
            self.monitor_canvas.yview_scroll(3, "units")
        return "break"

    def on_monitor_tab_changed(self, _event=None):
        """Refresh the detailed log for compatibility with older callers."""
        if self.monitor_views is None:
            if self.output and Path(self.output).is_dir():
                try:
                    self._load_history(Path(self.output))
                except OSError:
                    pass
            return
        try:
            tab_text = self.monitor_views.tab(self.monitor_views.select(), "text")
        except tk.TclError:
            return
        # The worker may have appended api_logs.jsonl after the last UI event.
        # Reload the same run when the user opens a detail tab so the click is
        # immediately reflected without requiring a new configuration change.
        if self.output and Path(self.output).is_dir() and tab_text in {"详细日志", "调用流程"}:
            try:
                self._load_history(Path(self.output))
            except OSError:
                pass
        self.status.set(f"当前查看：{tab_text}")

    def _scroll_detail_log(self, event):
        """Handle wheel and paging directly on the always-visible log pane."""
        keysym = str(getattr(event, "keysym", ""))
        if keysym in {"Prior", "Next", "Home", "End"}:
            if keysym == "Prior":
                self.log.yview_scroll(-1, "pages")
            elif keysym == "Next":
                self.log.yview_scroll(1, "pages")
            elif keysym == "Home":
                self.log.yview_moveto(0)
            else:
                self.log.yview_moveto(1)
        else:
            num = getattr(event, "num", None)
            delta = getattr(event, "delta", 0)
            if num == 4 or delta > 0:
                self.log.yview_scroll(-3, "units")
            elif num == 5 or delta < 0:
                self.log.yview_scroll(3, "units")
        return "break"

    def _append_flow_record(self, batch_name, index, record):
        """Add one raw API log record to the chronological call-flow table."""
        response = record.get("response") or {}
        input_data = record.get("input_parameters") or {}
        input_raw = str(input_data.get("raw", "")) if isinstance(input_data, dict) else str(input_data)
        input_parts = [
            part for part in input_raw.split(";")
            if part and part.split("=", 1)[0].strip().lower() not in {"flibhndl", "handle"}
        ]
        api_name = str(record.get("protocol_function") or record.get("interface_name", "unknown"))
        output_data = response.get("data", "") if isinstance(response, dict) else ""
        if output_data:
            output_data = api_payload(str(output_data), api_name, "out")
        params = ""
        if input_parts:
            params = "输入=" + ";".join(input_parts)
        if output_data:
            params += (" | " if params else "") + "输出=" + str(output_data)
        return_text = response.get("return_text", "") if isinstance(response, dict) else ""
        return_code = str(record.get("status_code", ""))
        if return_text:
            return_code += f"/{return_text}"
        if self.monitor_flow is not None:
            self.monitor_flow.insert("", "end", values=(
                batch_name, index, record.get("phase", ""), api_name,
                record.get("step_id", ""), return_code, record.get("traffic_status", ""),
                params, record.get("invocation_id", ""),
            ))

    def build_plan_summary(self):
        """Build the compact, human-readable plan card shown above logs."""
        parent = getattr(self, "monitor_body", None) or self.monitor
        self.plan_card = tk.Frame(parent, bg="#ffffff", highlightthickness=1,
                                  highlightbackground="#d9e0e7", padx=22, pady=14)
        self.plan_card.pack(fill="x", pady=(0, 12))

        title_row = tk.Frame(self.plan_card, bg="#ffffff")
        title_row.pack(fill="x")
        tk.Label(title_row, text="目标 API：", bg="#ffffff", fg="#111827",
                 font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        tk.Label(title_row, textvariable=self.plan_target, bg="#f3f4f6", fg="#374151",
                 padx=8, pady=3, font=("Consolas", 13)).pack(side="left", padx=(4, 0))
        self.plan_badge = tk.Label(title_row, textvariable=self.plan_badge_text,
                                   bg="#dcfce7", fg="#15803d", padx=10, pady=4,
                                   font=("Microsoft YaHei UI", 10, "bold"))
        self.plan_badge.pack(side="right")

        stats = tk.Frame(self.plan_card, bg="#ffffff")
        stats.pack(fill="x", pady=(14, 10))
        for column in range(3):
            stats.columnconfigure(column, weight=1)
        for column, label, variable in (
            (0, "规划步骤", self.plan_steps_count),
            (1, "检索片段", self.plan_chunks_count),
            (2, "NC 程序", self.plan_nc_required),
        ):
            cell = tk.Frame(stats, bg="#ffffff")
            cell.grid(row=0, column=column, sticky="w", padx=(0, 46))
            tk.Label(cell, text=label, bg="#ffffff", fg="#6b7280",
                     font=("Microsoft YaHei UI", 10)).pack(anchor="w")
            tk.Label(cell, textvariable=variable, bg="#ffffff", fg="#111827",
                     font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w", pady=(2, 0))

        steps_wrap = tk.Frame(self.plan_card, bg="#ffffff", height=176)
        steps_wrap.pack(fill="x", expand=False)
        steps_wrap.pack_propagate(False)
        self.plan_steps_canvas = tk.Canvas(steps_wrap, bg="#ffffff", highlightthickness=0)
        self.plan_steps_scroll = ttk.Scrollbar(steps_wrap, orient="vertical",
                                               command=self.plan_steps_canvas.yview)
        self.plan_steps_body = tk.Frame(self.plan_steps_canvas, bg="#ffffff")
        self.plan_steps_window = self.plan_steps_canvas.create_window(
            (0, 0), window=self.plan_steps_body, anchor="nw")
        self.plan_steps_canvas.configure(yscrollcommand=self.plan_steps_scroll.set)
        self.plan_steps_canvas.pack(side="left", fill="both", expand=True)
        self.plan_steps_scroll.pack(side="right", fill="y")
        self.plan_steps_body.bind(
            "<Configure>",
            lambda _event: self.plan_steps_canvas.configure(
                scrollregion=self.plan_steps_canvas.bbox("all")))
        self.plan_steps_canvas.bind(
            "<Configure>",
            lambda event: self.plan_steps_canvas.itemconfigure(
                self.plan_steps_window, width=event.width))

    def _set_plan_badge(self, text, color="#15803d", background="#dcfce7"):
        self.plan_badge_text.set(text)
        self.plan_badge.configure(fg=color, bg=background)

    def _extract_plan_target(self, data):
        for key in ("target_api", "target", "api_name"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        values = data.get("target_apis") or data.get("required_target_functions")
        if isinstance(values, (list, tuple)) and values:
            return ", ".join(str(value) for value in values[:3])
        task = str(data.get("task", ""))
        match = re.search(r"目标共\s*\d+\s*个唯一 API：\s*([a-zA-Z0-9_, ]+)", task)
        if match:
            return match.group(1).strip()
        names = re.findall(r"\b(?:cnc|pmc)_[A-Za-z0-9_]+\b", task)
        return names[0] if names else "未识别目标"

    def render_plan_summary(self, data):
        target = self.current_plan_target or self._extract_plan_target(data)
        self.current_plan_target = target
        self.plan_target.set(target)
        self.plan_steps_count.set(str(data.get("plan_steps", len(data.get("steps") or []))))
        self.plan_chunks_count.set(str(data.get("retrieved_chunks", "—")))
        nc_spec = data.get("nc_program_spec") or {}
        goals = " ".join(str(item) for item in nc_spec.get("block_goals") or [])
        program_name = str(nc_spec.get("program_name") or "").strip()
        needs_nc = bool(program_name) and "no NC" not in goals.lower()
        self.plan_nc_required.set("需要" if needs_nc else "不需要")
        self._set_plan_badge("规划完成")

        for child in self.plan_steps_body.winfo_children():
            child.destroy()
        steps = data.get("steps") or []
        if not steps:
            tk.Label(self.plan_steps_body, text="规划步骤将在规划智能体完成后显示。",
                     bg="#ffffff", fg="#6b7280", anchor="w").pack(fill="x", pady=8)
            return
        for index, step in enumerate(steps, start=1):
            row = tk.Frame(self.plan_steps_body, bg="#ffffff")
            row.pack(fill="x", pady=(0, 8))
            tk.Label(row, text=str(index), bg="#ffffff", fg="#111827",
                     font=("Microsoft YaHei UI", 11)).pack(side="left", padx=(2, 14), anchor="n")
            detail = tk.Frame(row, bg="#ffffff", highlightthickness=0)
            detail.pack(side="left", fill="x", expand=True)
            function = step.get("protocol_function") or step.get("interface_name") or "未命名步骤"
            repeat = int(step.get("repeat") or 1)
            suffix = f"，计划调用 {repeat} 次" if repeat > 1 else ""
            tk.Label(detail, text=f"{step.get('step_id', 'S—')} · {function}{suffix}",
                     bg="#ffffff", fg="#111827", anchor="w",
                     font=("Microsoft YaHei UI", 11, "bold")).pack(fill="x")
            tk.Label(detail, text=str(step.get("action") or "未提供步骤说明"),
                     bg="#ffffff", fg="#6b7280", anchor="w", justify="left",
                     wraplength=760, font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(4, 0))

    def build_graph(self):
        """Build a Neo4j Browser-like TRCM graph panel."""
        header = ttk.Frame(self.graph_tab)
        header.pack(fill="x")
        ttk.Label(header, text="流量关系依赖图", font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        ttk.Label(
            header,
            text="FANUC 0i-D · 节点颜色表示连接度，边颜色表示关系证据状态。",
        ).pack(side="left", padx=(14, 0))

        controls = ttk.Frame(self.graph_tab)
        controls.pack(fill="x", pady=(12, 8))
        ttk.Label(controls, text="API 名称").pack(side="left")
        search = ttk.Entry(controls, textvariable=self.graph_search, width=32)
        search.pack(side="left", padx=8)
        search.bind("<Return>", lambda _event: self.refresh_graph())
        ttk.Button(controls, text="查询", command=self.refresh_graph).pack(side="left")
        ttk.Button(controls, text="显示整体", command=self.show_full_graph).pack(side="left", padx=8)
        ttk.Button(controls, text="适配视图", command=self.fit_graph).pack(side="left")
        ttk.Button(controls, text="＋", width=3, command=lambda: self.zoom_graph(1.2)).pack(side="left", padx=(8, 2))
        ttk.Button(controls, text="－", width=3, command=lambda: self.zoom_graph(1 / 1.2)).pack(side="left")
        ttk.Button(controls, text="打开 Neo4j Browser", command=self.open_neo4j_browser).pack(side="right")
        ttk.Label(controls, textvariable=self.graph_status).pack(side="left", padx=12)

        body = ttk.Frame(self.graph_tab)
        body.pack(fill="both", expand=True)
        self.graph_canvas = tk.Canvas(body, background="#0f172a", highlightthickness=1,
                                      highlightbackground="#334155")
        x_scroll = ttk.Scrollbar(body, orient="horizontal", command=self.graph_canvas.xview)
        y_scroll = ttk.Scrollbar(body, orient="vertical", command=self.graph_canvas.yview)
        self.graph_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        details = ttk.LabelFrame(body, text="选中对象属性", padding=8)
        details.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(10, 0))
        self.graph_detail_title = ttk.Label(details, text="未选择对象", font=("Microsoft YaHei UI", 10, "bold"))
        self.graph_detail_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.graph_detail_tree = ttk.Treeview(details, columns=("property", "value"), show="headings", height=20)
        self.graph_detail_tree.heading("property", text="属性")
        self.graph_detail_tree.heading("value", text="值")
        self.graph_detail_tree.column("property", width=130, minwidth=100, anchor="w")
        self.graph_detail_tree.column("value", width=260, minwidth=150, anchor="w")
        detail_scroll = ttk.Scrollbar(details, orient="vertical", command=self.graph_detail_tree.yview)
        self.graph_detail_tree.configure(yscrollcommand=detail_scroll.set)
        self.graph_detail_tree.grid(row=1, column=0, sticky="nsew")
        detail_scroll.grid(row=0, column=1, sticky="ns")
        detail_scroll.grid_configure(row=1)
        copy_button = ttk.Button(details, text="复制选中值", command=self._copy_graph_detail_value)
        copy_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        details.rowconfigure(1, weight=1)
        details.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(2, weight=0)
        self.graph_canvas.bind("<Control-MouseWheel>", self._graph_zoom_event)
        self.graph_canvas.bind("<ButtonPress-2>", self._graph_pan_start)
        self.graph_canvas.bind("<B2-Motion>", self._graph_pan_move)
        self.graph_canvas.bind("<ButtonPress-1>", self._graph_node_press)
        self.graph_canvas.bind("<B1-Motion>", self._graph_node_move)
        self.graph_canvas.bind("<ButtonRelease-1>", self._graph_node_release)
        self.graph_canvas.bind("<Motion>", self._graph_hover)

        legend = tk.Frame(self.graph_tab, bg="#0f172a", padx=10, pady=6)
        legend.pack(fill="x", pady=(6, 0))
        tk.Label(legend, text="关系证据", bg="#0f172a", fg="#cbd5e1",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")
        for color, label in (("#34d399", "运行验证"), ("#38bdf8", "文档确认"),
                             ("#a78bfa", "SDK 校验"), ("#f59e0b", "候选")):
            tk.Label(legend, text=f"  ● {label}", bg="#0f172a", fg=color,
                     font=("Microsoft YaHei UI", 9)).pack(side="left")

    def show_full_graph(self):
        self.graph_search.set("")
        self.graph_zoom = 1.0
        self.refresh_graph()

    def open_neo4j_browser(self):
        """Open the same graph backend in Neo4j Browser for full inspection."""
        webbrowser.open("http://localhost:7474/browser/")

    def zoom_graph(self, factor):
        self.graph_zoom = max(0.35, min(3.0, self.graph_zoom * float(factor)))
        if self.graph_data is not None:
            self.render_graph(self.graph_data, cache=False)

    def fit_graph(self):
        self.graph_zoom = 1.0
        if self.graph_data is not None:
            self.render_graph(self.graph_data, cache=False)

    def _graph_zoom_event(self, event):
        self.zoom_graph(1.15 if event.delta > 0 else 1 / 1.15)

    def _graph_pan_start(self, event):
        self.graph_pan_origin = (event.x, event.y)

    def _graph_pan_move(self, event):
        if self.graph_pan_origin is None:
            return
        self.graph_canvas.scan_dragto(event.x, event.y, gain=1)
        self.graph_pan_origin = (event.x, event.y)

    def _graph_hover(self, event):
        current = self.graph_canvas.find_withtag("current")
        if not current:
            return
        for tag in self.graph_canvas.gettags(current[0]):
            if tag.startswith("node:"):
                self.graph_status.set(f"节点：{tag[5:]}")
                return

    def _graph_current_node(self):
        current = self.graph_canvas.find_withtag("current")
        if not current:
            return None
        for tag in self.graph_canvas.gettags(current[0]):
            if tag.startswith("node:"):
                return tag[5:]
        return None

    def _graph_node_press(self, event):
        self.graph_drag_node = self._graph_current_node()
        self.graph_click_origin = (event.x, event.y)
        if self.graph_drag_node:
            self.graph_canvas.configure(cursor="fleur")
            self.graph_drag_last_position = self.graph_positions.get(self.graph_drag_node)
            self.graph_pan_origin = None
        else:
            # Dragging empty canvas space pans the whole dependency graph,
            # while dragging a node keeps the existing node-layout behavior.
            self.graph_pan_origin = (event.x, event.y)
            self.graph_canvas.configure(cursor="fleur")

    def _graph_node_move(self, event):
        if not self.graph_drag_node:
            if self.graph_pan_origin is not None:
                self.graph_canvas.scan_dragto(event.x, event.y, gain=1)
                self.graph_pan_origin = (event.x, event.y)
            return
        if self.graph_drag_node not in self.graph_positions:
            return
        position = (
            self.graph_canvas.canvasx(event.x) / max(self.graph_zoom, 0.01),
            self.graph_canvas.canvasy(event.y) / max(self.graph_zoom, 0.01),
        )
        previous = self.graph_drag_last_position or self.graph_positions[self.graph_drag_node]
        delta_x, delta_y = position[0] - previous[0], position[1] - previous[1]
        self.graph_positions[self.graph_drag_node] = position
        # Keep the graph readable without an always-on simulation: direct
        # neighbours follow a fraction of the manual drag displacement.
        if self.graph_data is not None:
            neighbours = set()
            for relation in self.graph_data.get("relations", []):
                if not isinstance(relation, dict):
                    continue
                source = str(relation.get("source_api", ""))
                target = str(relation.get("target_api", ""))
                if source == self.graph_drag_node:
                    neighbours.add(target)
                elif target == self.graph_drag_node:
                    neighbours.add(source)
            for neighbour in neighbours:
                if neighbour not in self.graph_positions:
                    continue
                nx, ny = self.graph_positions[neighbour]
                self.graph_positions[neighbour] = (nx + delta_x * 0.22, ny + delta_y * 0.22)
        self.graph_drag_last_position = position
        if self.graph_data is not None:
            self.render_graph(self.graph_data, cache=False)

    def _graph_node_release(self, event):
        origin = self.graph_click_origin
        was_click = origin is not None and abs(event.x - origin[0]) <= 4 and abs(event.y - origin[1]) <= 4
        if was_click:
            self._graph_select_current()
        if self.graph_drag_node:
            self.graph_canvas.configure(cursor="")
            self.graph_status.set(f"已定位节点：{self.graph_drag_node}（可继续拖动）")
        self.graph_drag_node = None
        self.graph_drag_last_position = None
        self.graph_click_origin = None
        self.graph_pan_origin = None
        self.graph_canvas.configure(cursor="")

    def _set_graph_details(self, title: str, payload: Any) -> None:
        """Render selected node/edge data as a Neo4j-like key/value table."""
        if self.graph_detail_tree is None:
            return
        self.graph_detail_title.configure(text=title)
        self.graph_detail_values = {}
        for item in self.graph_detail_tree.get_children():
            self.graph_detail_tree.delete(item)
        rows = payload.items() if isinstance(payload, dict) else (("value", payload),)
        for key, value in rows:
            raw = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
            display = raw if len(raw) <= 180 else raw[:177] + "..."
            item_id = self.graph_detail_tree.insert("", "end", values=(str(key), display))
            self.graph_detail_values[item_id] = raw

    def _copy_graph_detail_value(self) -> None:
        if self.graph_detail_tree is None:
            return
        selected = self.graph_detail_tree.selection()
        if not selected:
            return
        value = self.graph_detail_values.get(selected[0], "")
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.graph_status.set("已复制选中属性值")

    def _graph_select_current(self):
        current = self.graph_canvas.find_withtag("current")
        if not current or not self.graph_data:
            return
        tags = self.graph_canvas.gettags(current[0])
        relation_index = next((int(tag[9:]) for tag in tags if tag.startswith("relation:") and tag[9:].isdigit()), None)
        if relation_index is not None:
            relations = self.graph_data.get("relations", [])
            if 0 <= relation_index < len(relations):
                relation = relations[relation_index]
                self._set_graph_details("关系属性", relation)
                self.graph_status.set(
                    f"关系：{relation.get('source_api', '')} → {relation.get('target_api', '')}"
                )
            return
        name = next((tag[5:] for tag in tags if tag.startswith("node:")), None)
        if not name:
            return
        node = next((item for item in self.graph_data.get("nodes", [])
                     if isinstance(item, dict) and str(item.get("name", "")) == name), {"name": name})
        related = [item for item in self.graph_data.get("relations", [])
                   if isinstance(item, dict) and (str(item.get("source_api", "")) == name or str(item.get("target_api", "")) == name)]
        self._set_graph_details("API 节点属性 · " + name, node)
        if related and self.graph_detail_tree is not None:
            parent = self.graph_detail_tree.insert("", "end", values=("关联关系", f"{len(related)} 条"))
            self.graph_detail_values[parent] = json.dumps(related, ensure_ascii=False, default=str)
        self.graph_status.set(f"已选中节点：{name}（{len(related)} 条关联关系）")

    def refresh_graph(self):
        if self.graph_querying:
            return
        query = self.graph_search.get().strip()
        self.graph_querying = True
        self.graph_status.set("正在查询关系库…")
        threading.Thread(target=self._query_graph, args=(query,), daemon=True).start()

    def refresh_knowledge_status(self):
        """Show the actual graph/vector source state before any capture starts."""
        def inspect():
            vector_path = Path(self.values["vector"].get()).expanduser()
            try:
                kb = KnowledgeBase(
                    vector_db_dir=vector_path,
                    collection_name=("heidenhain_lsv2_knowledge" if self.values["protocol"].get() == "lsv2" else "focas_knowledge"),
                )
                vector = kb.status
            except Exception as exc:
                vector = {"status": "unavailable", "vector_db": {"error": str(exc)},
                          "fallback_available": False, "chunk_count": 0}
            # Start the configured local Neo4j server and wait for Bolt before
            # rendering the status banner.  Reading ``status`` first races the
            # launcher and shows a false "knowledge base disconnected" dialog
            # during normal startup.
            try:
                startup = self.graph_service.ensure_local_server_started()
                graph = self.graph_service.status
                if startup.get("status") == "error":
                    graph = startup
            except Exception as exc:
                graph = {"status": "error", "available": False, "backend": "neo4j", "error": str(exc)}
            self.events.put(("knowledge_status", {"vector": vector, "graph": graph}))
        threading.Thread(target=inspect, daemon=True).start()

    def _render_knowledge_status(self, payload):
        vector = payload.get("vector", {})
        graph = payload.get("graph", {})
        vector_status = vector.get("status", "unavailable")
        graph_status = graph.get("status", "unconfigured")
        vector_error = ((vector.get("vector_db") or {}).get("error") or "").strip()
        graph_error = str(graph.get("error") or "").strip()
        if graph_status in {"configured", "connected"} and vector_status == "vector":
            message = "知识库已连接：Neo4j 关系图 + RAG 向量库"
        elif vector_status == "fallback":
            message = "知识库部分可用：关系图未连接，当前使用本地降级知识"
            if graph_error:
                message += f"（{graph_error}）"
        elif vector_status in {"unavailable", "empty"} and graph_status not in {"configured", "connected"}:
            message = "知识库不可用：Neo4j 关系图和 RAG 向量库均未连接"
        else:
            message = "知识库未完全连接：" + "；".join(item for item in (
                f"关系图：{graph_error or graph_status}",
                f"向量库：{vector_error or vector_status}",
            ) if item)
        self.knowledge_status.set(message)
        if graph_status not in {"configured", "connected"} or vector_status in {"unavailable", "empty"}:
            self.status.set(message)
            if not self._knowledge_warning_shown:
                self._knowledge_warning_shown = True
                self.root.after(0, lambda: messagebox.showwarning("知识库未连接", message,
                                                                  parent=self.root))

    def _query_graph(self, query):
        try:
            self.graph_service.ensure_local_server_started()
            result = self.graph_service.call(
                "get_graph_overview",
                query=query,
                device_scope="fanuc:0i-D",
            )
        except Exception as exc:
            result = {"available": False, "backend": "unknown", "error": str(exc)}
        self.events.put(("graph_result", result))

    def render_graph(self, result, *, cache=True):
        if cache:
            self.graph_data = result
        self.graph_canvas.delete("all")
        self.graph_node_items = {}
        if not result.get("available"):
            backend = result.get("backend", "none")
            error = result.get("error", "关系库未配置")
            self.graph_status.set(f"关系图不可用（{backend}）：{error}")
            self._render_knowledge_status({"graph": {"status": "error", "error": error},
                                           "vector": {"status": "unknown"}})
            self.graph_canvas.create_text(
                30, 30, anchor="nw", text="暂时无法读取关系图。请检查 Neo4j 服务和连接配置。",
                fill="#fca5a5", font=("Microsoft YaHei UI", 12, "bold"),
            )
            return

        nodes = [node for node in result.get("nodes", []) if isinstance(node, dict) and node.get("name")]
        relations = [relation for relation in result.get("relations", []) if isinstance(relation, dict)]
        names = [str(node["name"]) for node in nodes]
        if not names:
            self.graph_status.set("没有找到关系节点")
            self.graph_canvas.create_text(30, 30, anchor="nw", text="当前范围没有可显示的 API。", fill="#cbd5e1")
            return

        # A deterministic ring layout gives a stable Browser-like graph. A
        # search result puts the matching API in the centre and its neighbours
        # around it; the full 308-node view uses degree-ranked rings. The
        # layout stays fixed until the operator drags a node.
        degree = {name: 0 for name in names}
        for relation in relations:
            source, target = str(relation.get("source_api", "")), str(relation.get("target_api", ""))
            if source in degree:
                degree[source] += 1
            if target in degree:
                degree[target] += 1
        query_text = str(result.get("query", "")).strip()
        query = query_text.casefold()
        focal = next((name for name in names if name.casefold() == query), None)
        if focal is None and query:
            focal = next((name for name in names if query in name.casefold()), None)
        existing_positions = getattr(self, "graph_positions", {})
        positions = dict(existing_positions) if not cache and set(existing_positions) == set(names) else {}
        if not positions:
            ordered = sorted(names, key=lambda name: (-degree.get(name, 0), name))
            if focal:
                ordered.remove(focal)
                ordered.insert(0, focal)
            centre_x, centre_y = 800, 620
            if ordered:
                positions[ordered[0]] = (centre_x, centre_y)
            index, ring = 1, 1
            while index < len(ordered):
                count = min(len(ordered) - index, max(8, ring * 12))
                radius = 150 + ring * 125
                for slot in range(count):
                    angle = (2 * math.pi * slot / count) - math.pi / 2
                    positions[ordered[index + slot]] = (
                        centre_x + math.cos(angle) * radius,
                        centre_y + math.sin(angle) * radius,
                    )
                index += count
                ring += 1
        self.graph_positions = positions

        relation_colors = {
            "runtime_verified": "#34d399",
            "documented": "#38bdf8",
            "sdk_validated": "#a78bfa",
            "proposed": "#f59e0b",
        }
        relation_labels = {
            "filters": "参数筛选",
            "prepares": "前置准备",
            "REQUIRES": "需要",
            "PRECEDES": "先于",
            "PROVIDES_INPUT": "提供参数",
            "READS_AFTER_WRITE": "写后读",
            "SESSION_SEQUENCE": "会话顺序",
        }
        scale = self.graph_zoom
        # Several knowledge records can connect the same pair (for example a
        # dependency and an execution order). Draw one physical edge and
        # combine its semantic labels so the picture stays readable.
        grouped_relations = {}
        for relation in relations:
            endpoints = (str(relation.get("source_api", "")), str(relation.get("target_api", "")))
            key = tuple(sorted(endpoints))
            grouped_relations.setdefault(key, []).append(relation)
        for _edge_key, relation_group in grouped_relations.items():
            # The grouping key is sorted only to merge parallel semantic
            # records. Rendering must retain the stored source -> target
            # direction; alphabetical ordering would reverse some edges.
            first_relation = relation_group[0]
            source = str(first_relation.get("source_api", ""))
            target = str(first_relation.get("target_api", ""))
            if source not in positions or target not in positions:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            color = next(
                (relation_colors[status] for status in ("runtime_verified", "documented", "sdk_validated", "proposed")
                 if any(str(item.get("confidence_status", "")) == status for item in relation_group)),
                "#64748b",
            )
            x1, y1, x2, y2 = (value * scale for value in (x1, y1, x2, y2))
            directed_pairs = {(str(item.get("source_api", "")), str(item.get("target_api", ""))) for item in relation_group}
            relation_indexes = [index for index, item in enumerate(relations) if item in relation_group]
            self.graph_canvas.create_line(
                x1, y1, x2, y2, fill=color,
                width=max(1, int(1.5 * scale)),
                arrow=tk.BOTH if len(directed_pairs) > 1 else tk.LAST,
                tags=("relation", *(f"relation:{index}" for index in relation_indexes)),
            )
            if len(relations) <= 120:
                labels = []
                for item in relation_group:
                    label = relation_labels.get(str(item.get("relation_type", "")), str(item.get("relation_type", "")))
                    if label and label not in labels:
                        labels.append(label)
                self.graph_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2 - 7,
                                              text=" / ".join(labels), fill=color,
                                              font=("Microsoft YaHei UI", max(7, int(8 * scale))),
                                              tags=("relation", *(f"relation:{index}" for index in relation_indexes)))
        for name, (x, y) in positions.items():
            x, y = x * scale, y * scale
            radius = min(30, max(13, 13 + degree.get(name, 0) * 1.2)) * scale
            fill = "#f59e0b" if name == focal else ("#164e63" if degree.get(name, 0) else "#1e293b")
            outline = "#f8fafc" if name == focal else "#38bdf8"
            tag = f"node:{name}"
            self.graph_node_items[name] = self.graph_canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=fill, outline=outline, width=max(1, int(1.5 * scale)), tags=("node", tag),
            )
            label = name if len(names) <= 120 or degree.get(name, 0) > 0 or name == focal else ""
            if label:
                self.graph_canvas.create_text(x, y, text=label, fill="#e2e8f0",
                                              font=("Consolas", max(7, int(8 * scale))), tags=("label", tag))
        self.graph_canvas.configure(scrollregion=(0, 0, max(1600, int(1600 * scale)), max(1200, int(1200 * scale))))
        prefix = f"搜索“{query_text}” · " if query_text else "整体 · "
        self.graph_status.set(
            f"{prefix}{len(names)} 个 API，{len(relations)} 条关系，{len(grouped_relations)} 条连线 · 设备范围 FANUC 0i-D"
        )

    def config(self):
        current = self.template.get("1.0", "end-1c")
        if self.template_mode == "scene":
            self.scene_template_text = current
        else:
            self.full_template_text = current
        result = dict({k: v.get() for k, v in self.values.items()},
                      template=self.scene_template_text, full_template=self.full_template_text,
                      capture_mode=self.capture_mode.get(), selected=[])
        result["controller_model"] = detect_sdk_model(result["header"])
        if result.get("protocol") == "ezsocket":
            result["ezsocket_test_mode"] = self.ezsocket_test_mode.get()
            result["ezsocket_com_machine"] = self.ezsocket_com_machine.get().strip()
        if result.get("protocol") == "lsv2" and self.capture_mode.get() in {"auto_no_parameter", "read_only_priority"}:
            # These FOCAS-specific modes have no LSV2 planner equivalent;
            # map both to the complete TNC640 read-only catalog.
            result["capture_mode"] = "all_apis"
            result["selected_apis"] = [row["function"] for row in self.catalog
                                        if not self.api_unavailable_reason(row["function"])]
            return result
        if self.capture_mode.get() == "single_scene":
            scene = self.scene_choices.get(self.selected_scene.get())
            result["selected"] = [scene["scenario_id"]] if scene else []
        elif self.capture_mode.get() == "auto_no_parameter":
            target_names = set(auto_no_parameter_functions(result))
            result["selected_apis"] = [
                row["function"] for row in self.catalog
                if row["function"] in target_names
                and not self.api_unavailable_reason(row["function"])
            ]
        elif self.capture_mode.get() == "read_only_priority":
            # These targets contain no motion-only APIs, so never trigger an
            # NC program start just because a previous mode had it enabled.
            result["ui_start"] = False
            target_names = set(read_only_priority_functions(result))
            result["selected_apis"] = [
                row["function"] for row in self.catalog
                if row["function"] in target_names
                and not self.api_unavailable_reason(row["function"])
            ]
        else:
            result["selected_apis"] = [row["function"] for row in self.catalog
                if (self.capture_mode.get() == "all_apis" or row["function"] in self.selected_api_set)
                and not self.api_unavailable_reason(row["function"])]
        return result

    def signature(self, config, kind):
        if kind == "simulator" and config.get("protocol") == "lsv2":
            keys = ("protocol", "host", "port", "timeout")
        elif kind == "simulator" and config.get("protocol") == "ezsocket":
            keys = ("protocol", "host", "port", "timeout", "ezsocket_test_mode", "ezsocket_com_machine")
        else:
            keys = ("protocol", "controller_model", "host", "port", "dll", "bridge") if kind == "simulator" else ("base_url", "model", "api_key", "wire_api", "timeout")
        return tuple(config[k] for k in keys)

    def changed(self, *_):
        if getattr(self, "_switching_protocol", False):
            return
        config = self.config()
        for kind, status in (("simulator", self.sim_status), ("llm", self.llm_status)):
            if self.verified.get(kind) != self.signature(config, kind):
                status.set("配置已变化，请测试连接")
        self.load_history_for_config(config)
        self.preview()

    def _reset_monitor_for_target_change(self, previous, current, *, force=False):
        """Clear evidence from the previous API selection.

        A run directory is shared by configuration changes, so its path alone
        cannot identify the API currently shown in the monitor.  Reset the
        live panes when the target set changes and invalidate the history
        short-cut; the next run will populate them with the new target's
        events and evidence.
        """
        previous = frozenset(previous or ())
        current = frozenset(current or ())
        if not force and previous == current:
            return
        self._ignore_active_capture_events = bool(self.busy and self.action == "capture")
        self._loaded_history_path = None
        self.output = None
        self.outcome = None
        self._terminal_capture_event = False
        if self.monitor_events is not None:
            for item in self.monitor_events.get_children():
                self.monitor_events.delete(item)
        if self.monitor_flow is not None:
            for item in self.monitor_flow.get_children():
                self.monitor_flow.delete(item)
        if self.log is not None:
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("1.0", self.log_placeholder)
            self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.metrics.set("耗时 00:00   |   尚无采集文件")
        self.scene_status.set("已更换 API，等待开始采集")
        self.monitor_stage.set("已更换 API，等待开始采集")
        self.details_hint.set("尚未加载本次 API 的运行记录；开始采集后这里会显示完整原始日志。")
        self.flow_hint.set("本次 API 的调用记录会直接写入详细日志。")
        self._clear_failure_reason()
        self._clear_final_capture_status()

    def _set_final_capture_status(self, outcome, detail=""):
        if not hasattr(self, "final_capture_banner"):
            return
        styles = {
            "success": ("采集完成", "#166534", "#dcfce7"),
            "partial": ("采集部分完成", "#92400e", "#fef3c7"),
            "failure": ("采集失败", "#991b1b", "#fee2e2"),
        }
        text, foreground, background = styles.get(str(outcome), ("", "#475569", "#f1f5f9"))
        if detail:
            text += "：" + str(detail).strip()[:300]
        self.final_capture_status.set(text)
        self.final_capture_banner.configure(fg=foreground, bg=background)

    def _clear_final_capture_status(self):
        if hasattr(self, "final_capture_status"):
            self.final_capture_status.set("")
        if hasattr(self, "final_capture_banner"):
            self.final_capture_banner.configure(fg="#475569", bg="#eef2f6")

    def load_history_for_config(self, config):
        """Populate the monitor from an existing run without re-running it."""
        candidates = []
        for key in ("resume_from", "output"):
            raw = str(config.get(key, "")).strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_dir() and ((path / "settings.json").is_file()
                                  or (path / "full_api_scheduler_state.json").is_file()):
                candidates.append(path)
            elif path.is_dir():
                # Output roots may contain named groups (for example
                # ``targeted_program_api_...``) around the timestamped run.
                # Search recursively so the monitor can reopen the actual
                # run instead of silently showing an empty view.
                try:
                    runs = [item.parent for item in path.rglob("settings.json")
                            if item.is_file()]
                    runs.extend(item.parent for item in path.rglob("full_api_scheduler_state.json")
                                if item.is_file())
                except OSError:
                    runs = []
                unique_runs = {item.resolve() for item in runs}
                def history_score(item):
                    has_evidence = any(
                        candidate.is_file() and candidate.stat().st_size > 0
                        for candidate in (
                            item / "desktop.log",
                            item / "summary_zh.txt",
                            item / "coverage_progress.json",
                        )
                    ) or any(item.glob("batch_*/execution/*"))
                    try:
                        modified = item.stat().st_mtime
                    except OSError:
                        modified = 0
                    return (int(has_evidence), modified)
                candidates.extend(sorted(
                    unique_runs,
                    key=history_score,
                    reverse=True,
                )[:1])
        path = next((candidate.resolve() for candidate in candidates), None)
        if path is None or path == getattr(self, "_loaded_history_path", None):
            return
        self._loaded_history_path = path
        self._load_history(path)

    def _load_history(self, run_path):
        """Load human-readable prior evidence into both monitor tabs."""
        try:
            self.output = run_path
            # Reopening a run should replace the previous monitor contents.
            # Without this reset, a short desktop.log could leave the panel
            # looking empty or append unrelated runs after configuration changes.
            if self.monitor_events is not None:
                for item in self.monitor_events.get_children():
                    self.monitor_events.delete(item)
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("1.0", self.log_placeholder)
            self.log.configure(state="disabled")
            if self.monitor_flow is not None:
                for item in self.monitor_flow.get_children():
                    self.monitor_flow.delete(item)
            scheduler_path = run_path / "full_api_scheduler_state.json"
            scheduler = json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.is_file() else {}
            targets = scheduler.get("targets", {}) if isinstance(scheduler, dict) else {}
            status_counts = {}
            for name, record in targets.items():
                if not isinstance(record, dict):
                    continue
                status = str(record.get("status", "pending"))
                status_counts[status] = status_counts.get(status, 0) + 1
                label = {
                    "success": "成功", "history_skipped": "历史跳过", "failed": "失败",
                    "blocked": "阻断", "partial": "部分采集", "pending": "未处理",
                }.get(status, status)
                if self.monitor_events is not None:
                    self.monitor_events.insert(
                        "", "end", values=("历史", f"{name}：{label}"),
                    )

            # Full API runs expose a compact progress file even when the
            # desktop process was closed before it could emit its final event.
            progress_candidates = (
                run_path / "coverage_progress.json",
                run_path / "full_api_coverage_state.json",
            )
            progress = {}
            for progress_path in progress_candidates:
                if progress_path.is_file():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        progress = {}
                    if isinstance(progress, dict):
                        break
            selected_functions = progress.get("selected_api_functions")
            total = (
                progress.get("selected_api_count")
                or progress.get("total")
                or (len(selected_functions) if isinstance(selected_functions, list) else None)
                or progress.get("manifest_function_count")
            )
            completed = progress.get("captured_function_count")
            if completed is None:
                completed = progress.get("covered")
            if total is None and targets:
                total = len(targets)
            if completed is None:
                completed = sum(count for status, count in status_counts.items()
                                if status in {"success", "covered_success", "history_skipped"})
            if total is not None:
                total = int(total)
                completed = min(int(completed or 0), max(total, 0))
                self.progress["value"] = completed / max(total, 1) * 100
                self.scene_status.set(f"历史采集进展 · 已完成 {completed}/{total}")
                if self.monitor_events is not None:
                    self.monitor_events.insert(
                        "", "end", values=("历史", f"总体进展：已完成 {completed}/{total}"),
                    )
                self.monitor_stage.set(
                    "历史记录已加载：" + "、".join(
                        f"{label}{status_counts.get(status, 0)}"
                        for status, label in (("success", "成功 "), ("history_skipped", "历史跳过 "),
                                              ("partial", "部分 "), ("failed", "失败 "),
                                              ("blocked", "阻断 "))
                        if status_counts.get(status)
                    )
                )
            history_errors = []
            for item in (progress.get("last_batch_errors") or []):
                if str(item).strip():
                    history_errors.append(str(item))
            for item in (progress.get("scenarios") or {}).values():
                if isinstance(item, dict) and item.get("error"):
                    history_errors.append(str(item["error"]))
            if history_errors or status_counts.get("failed"):
                self._set_failure_reason("；".join(dict.fromkeys(history_errors)) or "历史记录标记为失败，但未保存具体错误原因")
            self.metrics.set(
                f"历史运行 {run_path.name}  |  进展 {completed}/{total if total is not None else '—'}  |  "
                f"成功 {status_counts.get('success', status_counts.get('covered_success', 0))}  |  "
                f"失败 {status_counts.get('failed', 0)}  |  部分 {status_counts.get('partial', 0)}"
            )

            log_lines = []
            desktop_log = run_path / "desktop.log"
            if desktop_log.is_file():
                log_lines.extend(desktop_log.read_text(encoding="utf-8", errors="replace").splitlines())

            # desktop.log is only the monitor event stream and is often just
            # a few lines after a worker is restarted.  The actual historical
            # evidence lives in each batch's execution directory, so load it
            # even when desktop.log exists.
            for batch in sorted(run_path.glob("batch_*")):
                if not batch.is_dir():
                    continue
                summary = batch / "summary_zh.txt"
                stdout = batch / "execution" / "api_script_stdout.txt"
                api_logs = batch / "execution" / "api_logs.jsonl"
                tool_calls = batch / "execution" / "tool_calls.jsonl"
                if summary.is_file():
                    log_lines.append(f"===== 历史批次 {batch.name} 摘要 =====")
                    log_lines.extend(summary.read_text(encoding="utf-8", errors="replace").splitlines())
                if stdout.is_file():
                    log_lines.append(f"===== 历史批次 {batch.name} 执行输出 =====")
                    log_lines.extend(stdout.read_text(encoding="utf-8", errors="replace").splitlines())
                if api_logs.is_file():
                    log_lines.append(f"===== 历史批次 {batch.name} FOCAS 调用明细 =====")
                    try:
                        api_lines = api_logs.read_text(encoding="utf-8", errors="replace").splitlines()
                    except OSError:
                        api_lines = []
                    for index, raw_line in enumerate(api_lines, 1):
                        try:
                            record = json.loads(raw_line)
                        except (ValueError, TypeError):
                            log_lines.append(raw_line)
                            continue
                        self._append_flow_record(batch.name, index, record)
                        response = record.get("response") or {}
                        input_data = record.get("input_parameters") or {}
                        input_raw = str(input_data.get("raw", "")) if isinstance(input_data, dict) else str(input_data)
                        # Handles are connection metadata, not useful API
                        # input diversity; keep them out of the compact UI line.
                        input_parts = [
                            part for part in input_raw.split(";")
                            if part and part.split("=", 1)[0].strip().lower() not in {"flibhndl", "handle"}
                        ]
                        api_name = str(record.get("protocol_function") or record.get("interface_name", "unknown"))
                        output_data = response.get("data", "") if isinstance(response, dict) else ""
                        if output_data:
                            output_data = api_payload(str(output_data), api_name, "out")
                        return_text = response.get("return_text", "") if isinstance(response, dict) else ""
                        status_code = record.get("status_code", "")
                        status_text = f"{status_code}"
                        if return_text:
                            status_text += f"/{return_text}"
                        details = [
                            f"[{batch.name}] API调用 {api_name}",
                            f"步骤={record.get('step_id', '')}",
                            f"返回码={status_text}",
                            f"流量={record.get('traffic_status', '')}",
                            f"调用ID={record.get('invocation_id', '')}",
                        ]
                        if record.get("transaction_id"):
                            details.append(f"事务ID={record['transaction_id']}")
                        if input_parts:
                            details.append("输入=" + ";".join(input_parts))
                        if output_data:
                            details.append("输出=" + str(output_data))
                        if record.get("error"):
                            details.append("错误=" + str(record["error"]))
                        log_lines.append(" | ".join(details))
                # The deterministic logger writes fixed-column projections in
                # addition to the legacy technical records.  Show both tables
                # in the detail pane so input and output rows can be inspected
                # independently while invocation IDs remain available in the
                # structured api_calls.jsonl artifact.
                fixed_input = batch / "execution" / "data" / "input_log.csv"
                fixed_output = batch / "execution" / "data" / "output_log.csv"
                for title, path, fields in (
                    ("Input Log", fixed_input, ("timestamp", "api_name", "api_parameter_count", "params")),
                    ("Output Log", fixed_output, ("timestamp", "api_name", "return_code", "return_desc", "success_data")),
                ):
                    if not path.is_file():
                        continue
                    log_lines.append(f"===== 历史批次 {batch.name} {title} =====")
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as stream:
                            for row in csv.DictReader(stream):
                                log_lines.append(" | ".join(
                                    f"{field}={row.get(field, '')}" for field in fields
                                ))
                    except (OSError, csv.Error):
                        log_lines.append(f"{title} 文件读取失败：{path.name}")
                if tool_calls.is_file():
                    log_lines.append(f"===== 历史批次 {batch.name} 工具调用 =====")
                    for raw_line in tool_calls.read_text(encoding="utf-8", errors="replace").splitlines():
                        try:
                            record = json.loads(raw_line)
                        except (ValueError, TypeError):
                            log_lines.append(raw_line)
                            continue
                        tool = record.get("tool") or record.get("tool_name") or record.get("name", "unknown")
                        result = record.get("result")
                        status = record.get("status") or (
                            result.get("status", "") if isinstance(result, dict) else ""
                        )
                        log_lines.append(
                            f"[{batch.name}] 工具调用 {tool} | 状态={status} | "
                            f"输入={json.dumps(record.get('input', record.get('arguments', {})), ensure_ascii=False, default=str)}"
                        )
            # The detail pane is the authoritative human-readable trace. Keep
            # every loaded line so the beginning of a run is not lost when a
            # user scrolls upward.
            for line in log_lines:
                self.append_log(line, persist=False, raw=True)
            self.details_hint.set(
                f"已加载 {len(log_lines)} 条日志；完整技术字段、返回码和错误证据如下。"
                if log_lines else "当前批次没有可显示的详细日志。"
            )
            flow_count = len(self.monitor_flow.get_children()) if self.monitor_flow is not None else 0
            self.flow_hint.set(
                f"已加载 {flow_count} 条真实 API 调用记录；可横向滚动查看参数、返回码和调用 ID。"
                if flow_count else "当前批次没有可显示的 API 调用记录。"
            )
            self.status.set(f"已加载历史采集记录：{run_path.name}")
        except (OSError, ValueError, TypeError) as exc:
            self.status.set(f"历史采集记录加载失败：{exc}")

    def select_all(self, selected):
        previous = set(self.selected_api_set)
        self.capture_mode.set("all_apis" if selected else "single_api")
        if not selected:
            self.selected_api_set = set()
        self.change_capture_mode()
        self.refresh_api_list()
        self._reset_monitor_for_target_change(previous, self.selected_api_set)

    def clear_api_selection(self):
        previous = set(self.selected_api_set)
        self.selected_api_set.clear()
        self.refresh_api_list()
        self.api_detail.set("尚未选择 API，请从列表中选择一个函数")
        self._reset_monitor_for_target_change(previous, self.selected_api_set)

    def select_filtered(self):
        previous = set(self.selected_api_set)
        self.selected_api_set = set([name for name in self.visible_api_names if not self.api_unavailable_reason(name)][:1])
        self.refresh_api_list()
        self._reset_monitor_for_target_change(previous, self.selected_api_set)

    def api_unavailable_reason(self, function):
        if self.values["protocol"].get() == "lsv2":
            from .heidenhain_catalog import HEIDENHAIN_READ_ONLY_API_NAMES
            if function not in HEIDENHAIN_READ_ONLY_API_NAMES:
                return "该海德汉 API 当前不作为 TNC640 默认采集目标：需要额外参数/权限，或控制器不适用"
            return ""
        if self.values["protocol"].get() == "ezsocket":
            return ""
        return unavailable_reason(function, {
            "protocol": self.values["protocol"].get(),
            "controller_model": self.values["controller_model"].get(),
        })

    def available_api_count(self):
        return sum(not self.api_unavailable_reason(row["function"]) for row in self.catalog)

    def change_controller_model(self, *_):
        if getattr(self, "_switching_protocol", False):
            return
        self.values["controller_model"].set(detect_sdk_model(self.values["header"].get()))
        self.refresh_api_list()
        self.update_scene_selection()

    def on_api_click(self, event):
        index = self.listbox.nearest(event.y)
        bounds = self.listbox.bbox(index)
        if bounds and bounds[1] <= event.y < bounds[1] + bounds[3] and index < len(self.visible_api_names):
            reason = self.api_unavailable_reason(self.visible_api_names[index])
            if reason:
                self.api_detail.set(reason)
                messagebox.showinfo("API 不可选", reason, parent=self.root)
                return "break"

    def refresh_api_list(self):
        if not hasattr(self, "listbox"):
            return
        is_lsv2 = self.values["protocol"].get() == "lsv2"
        is_ezsocket = self.values["protocol"].get() == "ezsocket"
        self.catalog = (
            heidenhain_tnc640_catalog() if is_lsv2 else
            list(load_ezsocket_manifest().values()) if is_ezsocket else
            function_manifest()
        )
        if hasattr(self, "api_picker_title"):
            self.api_picker_title.set(
                "海德汉 TNC640 · LSV-2 Library API（括号内为 pyLSV2 映射）· 先选分类，再选择一个操作"
                if is_lsv2 else ("EZSocket API · 先选分类，再选择一个函数" if is_ezsocket else "FOCAS API · 先选分类，再选择一个函数")
            )
        query = self.api_search.get().strip().casefold()
        selected_category = self.api_category.get()
        categories = ["全部分类", *sorted({
            str(row.get("category", "")).strip() or "未分类" for row in self.catalog
        }, key=str.casefold)]
        if hasattr(self, "category_combo"):
            self.category_combo.configure(values=categories)
        if selected_category not in categories:
            selected_category = "全部分类"
            self.api_category.set(selected_category)
        self._refreshing_api_list = True
        try:
            self.selected_api_set = {name for name in self.selected_api_set if not self.api_unavailable_reason(name)}
            self.listbox.delete(0, "end")
            self.visible_api_names = []
            for row in self.catalog:
                name = str(row["function"])
                category = str(row.get("category", "")).strip() or "未分类"
                description = str(row.get("description", ""))
                if selected_category != "全部分类" and category != selected_category:
                    continue
                if query and query not in f"{name} {category} {description}".casefold():
                    continue
                index = len(self.visible_api_names)
                self.visible_api_names.append(name)
                reason = self.api_unavailable_reason(name)
                library_name = str(row.get("library_function", "")).strip()
                library_category = str(row.get("library_category", "")).strip()
                display_category = f"{category} / {library_category}" if library_category else category
                display_name = f"{library_name}  ({name})" if library_name else name
                self.listbox.insert("end", f"{display_name}  ·  {display_category}" + ("  [不可选]" if reason else ""))
                if reason:
                    self.listbox.itemconfigure(index, foreground="#9ca3af", selectforeground="#9ca3af", selectbackground="#f1f3f5")
                if name in self.selected_api_set:
                    self.listbox.selection_set(index)
            self.api_selected_count.set(
                f"已选 {len(self.selected_api_set)} / {self.available_api_count()} 个可选 API；当前显示 {len(self.visible_api_names)} 个"
            )
            self.all_api_info.configure(text=f"将按函数清单分批采集全部 {self.available_api_count()} 个可选 API。")
            if self.visible_api_names:
                self.show_api_detail(self.visible_api_names[0])
            else:
                self.api_detail.set("当前筛选没有匹配的 API")
        finally:
            self._refreshing_api_list = False
        if hasattr(self, "template"):
            self.preview()

    def on_api_selection(self, _event=None):
        if self._refreshing_api_list:
            return
        previous = set(self.selected_api_set)
        chosen = {self.visible_api_names[index] for index in self.listbox.curselection()}
        blocked = [self.api_unavailable_reason(name) for name in chosen if self.api_unavailable_reason(name)]
        if blocked:
            self.refresh_api_list()
            self.api_detail.set(blocked[0])
            messagebox.showinfo("API 不可选", blocked[0], parent=self.root)
            return
        self.selected_api_set = chosen
        self._reset_monitor_for_target_change(previous, self.selected_api_set)
        self.api_selected_count.set(
            f"已选 {len(self.selected_api_set)} / {self.available_api_count()} 个可选 API；当前显示 {len(self.visible_api_names)} 个"
        )
        if self.listbox.curselection():
            self.show_api_detail(self.visible_api_names[self.listbox.curselection()[0]])
        self.preview()

    def show_api_detail(self, function):
        row = next((item for item in self.catalog if item["function"] == function), None)
        if row is not None:
            if self.values["protocol"].get() == "lsv2":
                header_status = "RemoTools LSV2 文档/pyLSV2 已登记"
            elif self.values["protocol"].get() == "ezsocket":
                header_status = "FCSB1224W000 OLE/COM 知识库已登记"
            else:
                header = Path(self.values["header"].get()) / "Fwlib32.h"
                try:
                    cache_key = (str(header), header.stat().st_mtime_ns)
                    if cache_key != self._header_cache_key:
                        self._header_cache_text = header.read_text(encoding="latin-1")
                        self._header_cache_key = cache_key
                    declared = bool(re.search(rf"\b{re.escape(function)}\s*\(", self._header_cache_text, re.IGNORECASE))
                    header_status = "头文件有原型" if declared else "头文件无原型"
                except OSError:
                    header_status = "头文件待检查"
            action = function.split("_", 1)[-1].lower()
            if self.values["protocol"].get() == "ezsocket":
                risk = "只读 API，可用于状态测试" if row.get("read_only") else "写入/控制 API，默认禁止自动测试"
            else:
                risk = (row.get("risk") if self.values["protocol"].get() == "lsv2" else None) or ("可能改变设备状态，需核验安全条件" if action.startswith(
                ("wr", "del", "clr", "reset", "set", "dwn", "download", "dnc", "start", "stop")
                ) else "操作风险待核验")
            chinese = chinese_api_description(function, row.get("description", ""))
            self.api_detail.set(
                f"{function} · {row.get('category', '未分类')}\n"
                f"LSV-2 Library：{row.get('library_function', '未登记')} · 分类：{row.get('library_category', '未登记')}\n"
                f"中文说明：{chinese}\n"
                f"原始说明：{row.get('description', '暂无说明')}\n"
                f"{header_status}；{risk}（仅提示，不能代替运行时验证）\n"
                f"{self.api_unavailable_reason(function) or '当前型号筛选下可选'}"
            )

    def test_ezsocket_com(self):
        """Run the documented read-only EZSocket COM API from the desktop."""
        if self.values["protocol"].get() != "ezsocket":
            return
        self.ezsocket_test_mode.set("com")
        self.launch("simulator")

    def test_device_connection(self):
        self.ezsocket_test_mode.set("tcp")
        self.launch("simulator")

    def reset_template(self):
        self.template.delete("1.0", "end")
        self.template.insert("1.0", PROMPT if self.template_mode == "scene" else FULL_API_PROMPT)
        self.preview()

    def begin_cycle_start_calibration(self):
        if self.busy or self.calibration_overlay is not None:
            return
        self.calibration_status.set("请在遮罩上点击 NCGuide 的 Cycle Start 按钮位置；本次点击不会触发按钮")
        self.status.set("Cycle Start 定位中：点击 NCGuide 上的 Cycle Start 位置")
        x, y, width, height = virtual_screen_geometry(self.root)
        self.root.withdraw()
        overlay = tk.Toplevel(self.root)
        overlay.withdraw()
        self.calibration_overlay = overlay
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.22)
        overlay.configure(bg="#d62828", cursor="crosshair")
        overlay.geometry(f"{width}x{height}{x:+d}{y:+d}")
        instruction = tk.Label(
            overlay,
            text="点击 NCGuide 的 Cycle Start 按钮位置\n此次点击仅记录坐标，不会启动程序 · 按 Esc 取消",
            bg="#d62828", fg="white", font=("Microsoft YaHei UI", 17, "bold"), padx=24, pady=16,
        )
        instruction.place(relx=0.5, rely=0.08, anchor="n")
        overlay.bind("<Button-1>", self.capture_cycle_start_point)
        instruction.bind("<Button-1>", self.capture_cycle_start_point)
        overlay.bind("<Escape>", lambda _event: self.cancel_cycle_start_calibration())
        try:
            overlay.update_idletasks()
            overlay.deiconify()
            overlay.update_idletasks()
            cover_virtual_screen(overlay)
        except (OSError, tk.TclError) as exc:
            self.cancel_cycle_start_calibration()
            messagebox.showerror("定位遮罩创建失败", str(exc))
            return
        overlay.focus_force()
        overlay.grab_set()

    def capture_cycle_start_point(self, event):
        x, y = int(event.x_root), int(event.y_root)
        if os.name == "nt":
            point = wintypes.POINT()
            get_cursor_pos = ctypes.windll.user32.GetCursorPos
            get_cursor_pos.argtypes = [ctypes.POINTER(wintypes.POINT)]
            get_cursor_pos.restype = wintypes.BOOL
            if get_cursor_pos(ctypes.byref(point)):
                x, y = int(point.x), int(point.y)
        overlay = self.calibration_overlay
        self.calibration_overlay = None
        if overlay is not None:
            try:
                overlay.grab_release()
            except tk.TclError:
                pass
            overlay.destroy()
        # Keep this window hidden until WindowFromPoint can see the NCGuide
        # window underneath the now-destroyed calibration overlay.
        self.root.after(180, lambda: self.confirm_cycle_start_point(x, y))

    def confirm_cycle_start_point(self, x, y):
        title = window_title_at_point(x, y)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        if not title:
            self.calibration_status.set("定位失败：所选位置下方没有可识别的窗口，请重试")
            self.status.set("Cycle Start 定位失败，请重新定位")
            return
        self.values["start_x"].set(str(x))
        self.values["start_y"].set(str(y))
        # The detected title is the actual top-level target used by the click
        # helper, so it is safer than retaining a stale default title.
        self.values["window_title"].set(title)
        self.values["ui_start"].set(True)
        message = f"已确认：({x}, {y})，窗口“{title}”"
        self.calibration_status.set(message)
        self.status.set(message)
        self.tabs.select(self.scene_tab)

    def cancel_cycle_start_calibration(self):
        overlay = self.calibration_overlay
        self.calibration_overlay = None
        if overlay is not None:
            try:
                overlay.grab_release()
            except tk.TclError:
                pass
            overlay.destroy()
        self.root.deiconify()
        self.root.lift()
        self.calibration_status.set("已取消 Cycle Start 定位")
        self.status.set("Cycle Start 定位已取消")

    def update_scene_selection(self):
        scene = self.scene_choices.get(self.selected_scene.get(), {})
        blocked = [name for name in atomic_scene_api_names(scene) if self.api_unavailable_reason(name)]
        self.scene_detail.set(
            f"{scene.get('goal', '')}\n设备状态：{' → '.join(scene.get('device_states', []))}"
            f"\n目标 API：{', '.join(scene.get('target_apis', scene.get('main_apis', [])))}"
            + (f"\n将自动排除不可用 API：{', '.join(blocked)}" if blocked else "")
        )
        self.preview()

    def change_capture_mode(self):
        previous_mode = getattr(self, "_monitor_capture_mode", None)
        mode = self.capture_mode.get()
        self._monitor_capture_mode = mode
        for widget in (self.api_picker, self.scene_picker, self.all_api_info):
            widget.pack_forget()
        widget = {"single_api": self.api_picker, "single_scene": self.scene_picker,
                  "all_apis": self.all_api_info,
                  "auto_no_parameter": self.auto_info,
                  "read_only_priority": self.readonly_info}[mode]
        widget.pack(fill="x", padx=(0, 12), pady=(0, 8), after=self.capture_settings)
        self.update_scene_selection()
        if previous_mode is not None and previous_mode != mode:
            self._reset_monitor_for_target_change((), (), force=True)

    def sync_template_mode(self):
        mode = "scene" if self.capture_mode.get() == "single_scene" else "full"
        if self.template_mode != mode:
            current = self.template.get("1.0", "end-1c")
            if self.template_mode == "scene":
                self.scene_template_text = current
            else:
                self.full_template_text = current
            self.template_mode = mode
            self.template.delete("1.0", "end")
            self.template.insert("1.0", self.scene_template_text if mode == "scene" else self.full_template_text)
        self.template_title.set("场景采集提示词模板" if mode == "scene" else "API 采集提示词模板")
        self.template_help.set(
            "$scope_title / $goal / $apis / $states / $operations\n$host / $port / $samples / $interval / $requirements"
            if mode == "scene" else
            "$manifest / $function_count / $selected_functions / $coverage_scope\n$host / $port / $device / $samples / $interval / $requirements"
        )

    def preview(self):
        self.sync_template_mode()
        config = self.config()
        if self.capture_mode.get() == "single_scene":
            title = f"单批次采集（场景） · {self.selected_scene.get()}"
        else:
            count = len(config["selected_apis"])
            scope = {"all_apis": "全部 API", "read_only_priority": "只读优先",
                     "auto_no_parameter": "AUTO 无参数前置"}.get(
                self.capture_mode.get(), "单个 API"
            )
            title = f"{scope}采集 · 已选 {count} / {self.available_api_count()} 个可选 API"
        self.preview_label.configure(text=title)
        try:
            text = "\n\n────────────\n\n".join(render_selected_prompts(config))
        except (KeyError, ValueError) as exc:
            text = f"模板错误：{exc}"
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text or ("请选择批次场景" if self.capture_mode.get() == "single_scene" else "请选择一个 API"))
        self.preview_text.configure(state="disabled")

    def start(self):
        if self.busy:
            return
        config = self.config()
        self.refresh_knowledge_status()
        cycle_error = cycle_start_configuration_error(config)
        if cycle_error:
            self.tabs.select(self.scene_tab)
            self.api_canvas.yview_moveto(1.0)
            self.status.set("请先完成 Cycle Start 定位，再开始采集")
            messagebox.showwarning("需要定位 Cycle Start", cycle_error, parent=self.root)
            return
        errors = validate(config, capture=True)
        checks = [("simulator", "仿真器")]
        if config.get("protocol", "focas") != "lsv2":
            checks.append(("llm", "模型"))
        for kind, label in checks:
            if self.verified.get(kind) != self.signature(config, kind):
                errors.append(f"{label}尚未通过当前配置的连接测试")
        if errors:
            messagebox.showerror("目前不能采集", "\n".join(errors))
            return
        self.launch("capture")

    def launch(self, action):
        if self.busy:
            return
        config = self.config()
        if action in ("preflight", "capture"):
            errors = validate(config, capture=True)
            if errors:
                messagebox.showerror("配置未完成", "\n".join(errors))
                return
        self.snapshot = dict(config, action=action)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        python = Path(sys.executable).with_name("python.exe") if os.name == "nt" else Path(sys.executable)
        try:
            self.process = subprocess.Popen([str(python), "-u", "-m", "smart_traffic_agent.desktop_backend"],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.process.stdin.write(json.dumps(self.snapshot, ensure_ascii=False) + "\n")
            self.process.stdin.close()
        except Exception as exc:
            if self.process and self.process.poll() is None:
                self.process.kill()
            messagebox.showerror("启动失败", str(exc))
            return
        self.busy, self.cancelled = True, False
        self._ignore_active_capture_events = False
        self._terminal_capture_event = False
        self.action = action
        self.started = time.monotonic()
        self.outcome = None
        for button in self.start_buttons:
            button.configure(state="disabled")
        for button in self.stop_buttons:
            button.configure(state="normal")
        self.status.set("正在运行；使用启动时的配置快照")
        self.append_log(f"开始：{action}")
        if action == "capture":
            self.output = None
            self._clear_failure_reason()
            self.progress["value"] = 0
            self.tabs.select(self.monitor)
            self.scene_status.set("采集前检查中")
        process = self.process
        def read():
            for line in process.stdout:
                self.events.put(("line", line.rstrip()))
            self.events.put(("exit", process.wait()))
        threading.Thread(target=read, daemon=True).start()

    def append_log(self, text, alert=False, persist=True, raw=False):
        for secret in (self.values["api_key"].get(), self.snapshot.get("api_key", "")):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        if raw:
            line = str(text).rstrip("\r\n") + "\n"
        else:
            line = time.strftime("%H:%M:%S") + "  " + localize_technical_text(text) + "\n"
        view_top, view_bottom = self.log.yview()
        follow_tail = view_bottom >= 0.99
        top_line = int(self.log.index("@0,0").split(".")[0])
        self.log.configure(state="normal")
        if self.log.get("1.0", "end-1c") == self.log_placeholder:
            self.log.delete("1.0", "end")
        self.log.insert("end", line, "alert" if alert else "")
        if follow_tail:
            self.log.see("end")
        elif top_line > 1:
            # Preserve the user's scroll position while new live records are
            # appended above the visible tail.
            self.log.yview_moveto(view_top)
        self.log.configure(state="disabled")
        if persist and self.output and self.action == "capture":
            try:
                with (self.output / "desktop.log").open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass

    def _set_failure_reason(self, reason):
        """Keep the concrete failure cause visible above the detailed log."""
        raw = str(reason or "采集进程未提供具体错误信息").strip()
        # Failure events may contain the same repair error repeated several
        # times. Keep the banner readable while leaving the full evidence in
        # the detailed log and output files.
        candidates = re.findall(
            r"C\+\+ API script execution failed[^,;\]]+|"
            r"CAPTURE_ARTIFACT_ERROR:[^,;\]]+|"
            r"CSV (?:schema|freshness) mismatch:[^,;\]]+|"
            r"EW_[A-Z_]+|Stopped repair loop[^,;\]]+",
            raw,
        )
        if candidates:
            compact = []
            for item in candidates:
                item = item.strip()
                if item and item not in compact:
                    compact.append(item)
            raw = "；".join(compact[:5])
        if "3221226505" in raw:
            raw += "（0xC0000409：原生程序快速失败，属于栈/缓冲区破坏类崩溃）"
        text = localize_technical_text(raw)
        if not text:
            text = "采集进程未提供具体错误信息"
        if len(text) > 700:
            text = text[:697] + "..."
        self.failure_reason.set("采集失败原因：" + text)
        self.status.set("采集失败：" + text[:180])

    def _clear_failure_reason(self):
        self.failure_reason.set("失败原因将在采集失败后显示。")

    def handle_event(self, row):
        name, data = row["event"], row["data"]
        if name == "agent_start" and "PlanningAgent" in str(data.get("agent", "")):
            self.current_plan_target = self._extract_plan_target(data)
            self.plan_target.set(self.current_plan_target)
            self._set_plan_badge("正在规划", "#1d4ed8", "#dbeafe")
        elif name == "agent_complete" and "PlanningAgent" in str(data.get("agent", "")):
            self.render_plan_summary(data)
        elif name == "agent_error" and "PlanningAgent" in str(data.get("agent", "")):
            self._set_plan_badge("规划失败", "#b91c1c", "#fee2e2")
        elif name == "success" and hasattr(self, "plan_badge"):
            self._set_plan_badge("采集完成")
            if self.action == "capture":
                self._set_final_capture_status("success", data.get("message", ""))
        elif name == "partial" and hasattr(self, "plan_badge"):
            self._set_plan_badge("部分采集", "#b45309", "#fef3c7")
            if self.action == "capture":
                self._set_final_capture_status("partial", data.get("message", ""))
        elif name == "failure" and hasattr(self, "plan_badge"):
            self._set_plan_badge("采集失败", "#b91c1c", "#fee2e2")
            if self.action == "capture":
                self._set_final_capture_status("failure", data.get("message") or data.get("error", ""))
        if name in {"failure", "scenario_failed", "coverage_batch_failed", "agent_error", "tool_error"}:
            self._set_failure_reason(data.get("error") or data.get("message") or "采集失败")
        self.update_monitor_summary(name, data)
        if name == "operator_action_required":
            self.show_operator_action_prompt(data)
        elif name == "operator_action_verified":
            self.finish_operator_action_prompt("已检测到状态恢复，采集正在继续。", success=True)
        elif name == "operator_action_timeout":
            self.finish_operator_action_prompt("等待超时。依赖该状态的 API 将暂缓，全 API 采集会继续处理其他 API。", success=False)
        if name == "knowledge_status":
            self._render_knowledge_status(data)
        if name == "output":
            self.output = Path(data["path"])
        if name == "scenario":
            self.scene_status.set(f"场景 {data['index']} / {data['total']}  ·  {data['name']}")
        if name in ("scenario_done", "scenario_partial", "scenario_failed"):
            self.progress["value"] = data["index"] / data["total"] * 100
        if name in ("coverage_batch", "coverage_progress"):
            covered, total = int(data["covered"]), int(data["total"])
            resolved = int(data.get("resolved", covered))
            self.progress["value"] = covered / max(total, 1) * 100
            label = "单 API 采集" if data.get("single_api") else f"API 采集 · 第 {data['batch']} 批"
            self.scene_status.set(f"{label} · 实采 {covered}/{total} · 已判定 {resolved}/{total}")
        if name in ("agent_start", "tool_start", "router_decision"):
            if name == "agent_start":
                self.status.set(f"{agent_label(data.get('agent'))}正在工作")
            elif name == "router_decision":
                self.status.set(f"{stage_label(data.get('next_stage'))}：正在判断下一步")
            else:
                self.status.set(f"工具正在执行：{data.get('tool', data.get('tool_name', '未命名工具'))}")
        if name in ("success", "partial", "failure"):
            self.outcome = name
            if self.action == "capture":
                # The terminal event is the end of the live monitor stream.
                # Do not replay historical API records after this point.
                self._terminal_capture_event = True
            self.status.set(self.monitor_stage.get())
        self.append_log(
            format_event(name, data),
            alert=name in ("success", "partial", "failure", "scenario_failed", "agent_error", "tool_error"),
        )

    def show_operator_action_prompt(self, data):
        action = str(data.get("action", ""))
        target = str(data.get("target", ""))
        seconds = int(data.get("timeout_seconds", 60))
        if action in ("remote", "mode"):
            title = "需要切换仿真器模式"
            required_mode = str(data.get("required_mode") or ("REMOTE" if action == "remote" else "目标模式"))
            instruction = f"请在 FANUC 主操作面板切换到 {required_mode}。系统会自动检查模式和控制器状态，符合条件后继续采集。"
        else:
            title = "需要人工复位"
            instruction = "请先解除急停（如有），再在仿真器操作面板点击 RESET。系统会检查报警、急停和运行状态，恢复后继续采集。"
        if target:
            instruction += f"\n\n当前操作：{target}"
        if self.operator_prompt is not None:
            try:
                self.operator_prompt.destroy()
            except tk.TclError:
                pass
        dialog = tk.Toplevel(self.root)
        self.operator_prompt = dialog
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        self.operator_prompt_label = ttk.Label(body, text=instruction, wraplength=430, justify="left")
        self.operator_prompt_label.pack(anchor="w")
        timer_text = "自动检测中，等待你完成操作后继续。" if action in ("remote", "mode") else f"自动检测中，最多等待 {seconds} 秒。"
        self.operator_prompt_timer = tk.StringVar(value=timer_text)
        ttk.Label(body, textvariable=self.operator_prompt_timer, foreground="#176a83").pack(anchor="w", pady=(14, 10))
        ttk.Button(body, text="知道了", command=dialog.destroy).pack(anchor="e")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max(30, (self.root.winfo_width() - dialog.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(30, (self.root.winfo_height() - dialog.winfo_height()) // 3)
        dialog.geometry(f"+{x}+{y}")
        # Bring the request above NCGuide long enough for the operator to see
        # it.  Keep it visible until the operator acknowledges it; the old
        # 800 ms topmost window was commonly covered by NCGuide before it was
        # noticed.  No modal grab is used, so NCGuide remains interactive.
        try:
            dialog.attributes("-topmost", True)
            dialog.lift()
            dialog.focus_force()
        except tk.TclError:
            pass

    def finish_operator_action_prompt(self, message, *, success):
        dialog = self.operator_prompt
        if dialog is None:
            return
        try:
            dialog.title("状态已恢复" if success else "人工操作等待结束")
            if self.operator_prompt_label is not None:
                self.operator_prompt_label.configure(text=message)
            if self.operator_prompt_timer is not None:
                self.operator_prompt_timer.set("状态核验完成" if success else "本轮已跳过相关 API")
            dialog.after(3500, dialog.destroy)
        except tk.TclError:
            pass
        self.operator_prompt = None
        self.operator_prompt_label = None
        self.operator_prompt_timer = None

    def update_monitor_summary(self, name, data):
        stages = {"planning": "正在规划参数与依赖", "code_generation": "正在生成采集代码",
                  "execution": "正在执行采集", "repair_code": "正在修复采集代码",
                  "repair_plan": "正在调整采集方案", "complete": "正在汇总结果"}
        message = ""
        if name == "router_decision":
            stage = data.get("next_stage", "")
            self.monitor_stage.set(stages.get(stage, f"正在处理：{stage_label(stage)}"))
        elif name in ("coverage_batch", "coverage_progress"):
            label = "单 API 采集" if data.get("single_api") else f"第 {data.get('batch', '—')} 批"
            message = f"{label} · 已完成 {data.get('covered', 0)} / {data.get('total', 0)} 个 API"
            if name == "coverage_batch":
                message += " · 开始处理"
            self.monitor_stage.set(message)
        elif name == "coverage_batch_failed":
            targets = data.get("targets", [])
            errors = list(dict.fromkeys(str(e) for e in data.get("errors", [])))
            label = "单 API 采集失败" if data.get("single_api") else f"第 {data.get('batch', '—')} 批未通过"
            message = f"{label} · 待处理 API：{', '.join(targets) or '查看详细日志'}"
            if errors:
                message += " · " + "; ".join(errors)[:240]
        elif name in ("status", "success", "partial", "failure"):
            message = str(data.get("message", ""))
            if name == "failure":
                codes = list(dict.fromkeys(re.findall(r"EW_\w+|PARAMETER_COVERAGE_INCOMPLETE|PROGRAM_\w+|API_PREREQUISITE_BLOCKED", message)))
                message = "采集未完成 · " + ("、".join(codes) if codes else localize_technical_text(message[:200])) + "；已保存的数据保留，具体原因见详细日志。"
            elif name == "partial":
                message = "采集部分完成 · " + message[:190] + "；流量质量可在离线审计中查看。"
            self.monitor_stage.set(message[:240])
        if message and self.monitor_events is not None:
            item = self.monitor_events.insert("", "end", values=(time.strftime("%H:%M:%S"), message))
            children = self.monitor_events.get_children()
            if len(children) > 200:
                self.monitor_events.delete(children[0])
            self.monitor_events.see(item)

    def poll(self):
        for _ in range(150):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "models":
                self.model_refresh_button.configure(state="normal")
                if value.get("error"):
                    self.llm_status.set(f"获取模型失败：{value['error']}；仍可手动填写模型名称")
                else:
                    models = value["models"]
                    current = self.values["model"].get().strip()
                    if current and current not in models:
                        models = [current, *models]
                    self.model_combo.configure(values=models)
                    self.llm_status.set(f"已获取 {len(value['models'])} 个可用模型，请选择后测试调用")
            elif kind == "graph_result":
                self.graph_querying = False
                self.render_graph(value)
            elif kind == "knowledge_status":
                self._render_knowledge_status(value)
            elif kind == "line":
                if self._ignore_active_capture_events or (
                    self.action == "capture" and self._terminal_capture_event
                ):
                    continue
                if value.startswith(EVENT):
                    try:
                        self.handle_event(json.loads(value[len(EVENT):]))
                    except (ValueError, KeyError, TypeError):
                        self.append_log(value)
                else:
                    self.append_log(value, alert=any(word in value.lower() for word in ("error", "failed", "failure", "失败")))
            else:
                # The live stream is the monitor source of truth.  Do not
                # replay desktop.log/API history when the worker exits: the
                # terminal success/failure event already closes this run.
                self.busy = False
                for button in self.start_buttons:
                    profile = protocol_profile(self.config())
                    button.configure(
                        state="normal" if profile.get("capture_available", profile.get("available", False)) else "disabled"
                    )
                for button in self.stop_buttons:
                    button.configure(state="disabled")
                if self.cancelled:
                    self.status.set("已停止采集进程；部分文件已保留。NCGuide 程序如仍运行，请在仿真器中停止。")
                elif value == 0 and self.outcome in {"success", "partial"}:
                    for test in (("simulator", "llm") if self.action == "preflight" else (self.action,)):
                        if test in ("simulator", "llm"):
                            status = self.sim_status if test == "simulator" else self.llm_status
                            self.verified[test] = self.signature(self.snapshot, test)
                            status.set("连接成功" if self.signature(self.config(), test) == self.verified[test] else "配置已变化，请重新测试")
                            if test == "simulator" and self.signature(self.config(), test) == self.verified[test]:
                                profile = protocol_profile(self.config())
                                enabled = profile.get("capture_available", profile.get("available", False))
                                for button in self.start_buttons:
                                    button.configure(state="normal" if enabled else "disabled")
                    if self.action == "capture":
                        self.scene_status.set("所选 API 采集部分完成" if self.outcome == "partial" else "所选 API 采集完成")
                        self._set_final_capture_status(self.outcome)
                else:
                    if self.action in ("simulator", "llm", "preflight"):
                        for test in (("simulator", "llm") if self.action == "preflight" else (self.action,)):
                            self.verified.pop(test, None)
                            (self.sim_status if test == "simulator" else self.llm_status).set("测试失败，详见监控日志")
                    if self.outcome != "failure":
                        reason = f"后端进程异常退出（退出码 {value}）；请检查详细日志和本次输出目录"
                        self._set_failure_reason(reason)
                        self.status.set(reason)
                    self.tabs.select(self.monitor)
                    if self.action == "capture":
                        self.scene_status.set("采集未完成 · 已完成 API 与数据已保留")
                        self._set_final_capture_status("failure", self.failure_reason.get())
        if self.busy:
            elapsed = int(time.monotonic() - self.started)
            if elapsed != getattr(self, "last_elapsed", -1):
                self.last_elapsed = elapsed
                files = []
                if self.output and self.action == "capture":
                    try:
                        files = [(p.suffix, p.stat().st_size) for p in self.output.rglob("*") if p.is_file() and p.suffix in (".pcap", ".csv")]
                    except OSError:
                        pass
                self.metrics.set(f"耗时 {elapsed // 60:02d}:{elapsed % 60:02d}   |   PCAP {sum(s == '.pcap' for s, _ in files)}   |   CSV {sum(s == '.csv' for s, _ in files)}   |   已写入 {sum(n for _, n in files) / 1024:.1f} KB")
        self.root.after(150, self.poll)

    def stop(self):
        if not self.busy or not self.process:
            return
        self.cancelled = True
        for button in self.stop_buttons:
            button.configure(state="disabled")
        self.status.set("正在终止采集进程及其子进程…")
        process = self.process
        def terminate():
            if os.name == "nt":
                result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode and process.poll() is None:
                    self.events.put(("line", "停止失败，请检查任务管理器中的采集进程"))
            else:
                process.terminate()
        threading.Thread(target=terminate, daemon=True).start()

    def open_output(self):
        """Open the active run directory and report failures in the UI.

        ``os.startfile`` raises from a Tk callback when Explorer cannot open a
        path.  That exception is only visible in the launching console, which
        made the button appear inert in the packaged desktop app.
        """
        candidates = []
        if self.output:
            candidates.append(Path(self.output))
        configured = str(self.values["output"].get()).strip()
        if configured:
            candidates.append(Path(configured).expanduser())
        path = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if path is None:
            self.status.set("输出目录尚未生成")
            messagebox.showinfo("输出目录", "尚未生成可打开的输出目录", parent=self.root)
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self.status.set(f"已打开输出目录：{path}")
        except (OSError, RuntimeError) as exc:
            # Explorer can be unavailable in a restricted desktop session;
            # retain a visible error instead of silently swallowing the click.
            self.status.set(f"打开输出目录失败：{exc}")
            messagebox.showerror("打开输出目录失败", f"{path}\n{exc}", parent=self.root)

    def close(self):
        if self.calibration_overlay is not None:
            self.cancel_cycle_start_calibration()
        if self.busy:
            if not messagebox.askyesno("退出 SMPAgent", "任务仍在运行，是否终止采集进程并退出？"):
                return
            self.stop()
            self.wait_close()
        else:
            self.root.destroy()

    def wait_close(self):
        if self.busy:
            self.root.after(200, self.wait_close)
        else:
            self.root.destroy()


def main():
    os.chdir(ROOT)
    enable_windows_dpi_awareness()
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()


def enable_windows_dpi_awareness():
    if os.name != "nt":
        return
    try:
        set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = wintypes.BOOL
        if set_context(ctypes.c_void_p(-4)):  # PER_MONITOR_AWARE_V2
            return
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def virtual_screen_geometry(root=None):
    if os.name != "nt" and root is not None:
        try:
            root.update_idletasks()
            return (
                int(root.winfo_vrootx()),
                int(root.winfo_vrooty()),
                int(root.winfo_vrootwidth()),
                int(root.winfo_vrootheight()),
            )
        except tk.TclError:
            pass
    if os.name != "nt":
        return 0, 0, 1920, 1080
    user32 = ctypes.windll.user32
    return (
        int(user32.GetSystemMetrics(76)),
        int(user32.GetSystemMetrics(77)),
        int(user32.GetSystemMetrics(78)),
        int(user32.GetSystemMetrics(79)),
    )


def cover_virtual_screen(overlay):
    """Cover the native virtual desktop, including its bottom edge and taskbar."""
    x, y, width, height = virtual_screen_geometry(overlay)
    if os.name != "nt":
        overlay.geometry(f"{width}x{height}{x:+d}{y:+d}")
        return
    user32 = ctypes.windll.user32
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    hwnd = user32.GetAncestor(overlay.winfo_id(), 2) or overlay.winfo_id()
    if not user32.SetWindowPos(hwnd, wintypes.HWND(-1), x, y, width, height, 0x0040):
        raise OSError("Windows 无法将定位遮罩铺满屏幕")
    overlay.update_idletasks()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise OSError("无法读取定位遮罩的实际边界")
    actual = (rect.left, rect.top, rect.right, rect.bottom)
    expected = (x, y, x + width, y + height)
    if actual != expected:
        raise OSError(f"定位遮罩未覆盖整个屏幕：实际 {actual}，预期 {expected}")


def window_title_at_point(x, y):
    if os.name != "nt":
        return "NCGuide"

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    user32 = ctypes.windll.user32
    user32.WindowFromPoint.argtypes = [POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    hwnd = user32.WindowFromPoint(POINT(x, y))
    if not hwnd:
        return ""
    root_hwnd = user32.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT
    length = user32.GetWindowTextLengthW(root_hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(root_hwnd, buffer, length + 1)
    return buffer.value.strip()


if __name__ == "__main__":
    main()
