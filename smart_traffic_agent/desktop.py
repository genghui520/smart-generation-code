"""Native Tk desktop application for SMPAgent."""
from __future__ import annotations

import json
import os
import re
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import ctypes
import urllib.error
import urllib.request
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from ctypes import wintypes

from .desktop_backend import (
    ROOT, EVENT, PROMPT, FULL_API_PROMPT, defaults, render_selected_prompts,
    function_manifest, scenarios, validate, atomic_scene_api_names, cycle_start_configuration_error, chinese_api_description,
    read_only_priority_functions, auto_no_parameter_functions,
)


from .protocol_profiles import PROTOCOL_PROFILES, protocol_profile
from .api_compatibility import detect_sdk_model, unavailable_reason


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
        for panel, name in ((self.connection, "01  连接配置"), (self.scene_tab, "02  采集模式与提示词"), (self.monitor, "03  采集监控")):
            self.tabs.add(panel, text=name)
        self.build_connections()
        self.build_scenes()
        self.build_monitor()
        for value in self.values.values():
            value.trace_add("write", self.changed)
        self.values["header"].trace_add("write", self.change_controller_model)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(150, self.poll)
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
        self.field(sim, 2, "抓包网卡", "device", [r"\Device\NPF_Loopback"])
        ttk.Label(sim, text="SDK 自动识别型号").grid(row=5, column=0, sticky="w", padx=8)
        ttk.Label(sim, textvariable=self.values["controller_model"], wraplength=260).grid(row=5, column=1, sticky="w", padx=8)
        ttk.Label(sim, text="更换 SDK 目录后自动更新型号和可选 API。", wraplength=360).grid(
            row=6, column=0, columnspan=3, sticky="w", padx=8)
        self.device_test_button = ttk.Button(sim, text="测试设备连接", command=lambda: self.launch("simulator"))
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
        self._active_protocol = key
        profile = PROTOCOL_PROFILES[key]
        self._switching_protocol = True
        try:
            self.values["protocol"].set(key)
            for name, value in saved["values"].items():
                self.values[name].set(value)
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
        self.device_frame.configure(text="目标设备 · " + profile["target"])
        self.protocol_status.set("已接入 · 支持真实采集" if available else "已继承 FOCAS 全部配置与模板 · 协议采集适配器待接入")
        self.protocol_environment.configure(text="FOCAS 专用环境" if available else "运行环境 · 继承自 FOCAS，待适配")
        self.device_test_button.configure(state="normal" if available else "disabled")
        for button in self.start_buttons:
            button.configure(state="normal" if available else "disabled")
        self.protocol_environment.pack(fill="x", pady=12, before=self.output_environment)
        self.protocol_pending.pack_forget()
        self.capture_pane.pack(fill="both", expand=True)
        self.refresh_api_list()
        self.change_capture_mode()
        self.sim_status.set("尚未测试" if available else "配置继承自 FOCAS；该协议连接测试待接入")
        self.status.set(f"当前协议：{profile['label']} · " + ("请选择采集内容" if available else "已继承 FOCAS 内容，可编辑；真实采集待接入"))
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
        ttk.Label(left, text="FOCAS API · 先选分类，再选择一个函数").pack(anchor="w", pady=(0, 8))
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
        monitor_head = ttk.Frame(self.monitor)
        monitor_head.pack(fill="x")
        ttk.Label(monitor_head, textvariable=self.scene_status, font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        monitor_stop = ttk.Button(monitor_head, text="停止采集", command=self.stop, state="disabled")
        monitor_stop.pack(side="right")
        self.stop_buttons.append(monitor_stop)
        self.progress = ttk.Progressbar(self.monitor, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=12)
        ttk.Label(self.monitor, textvariable=self.metrics).pack(anchor="w", pady=(0, 10))
        self.monitor_stage = tk.StringVar(value="等待开始采集")
        ttk.Label(self.monitor, textvariable=self.monitor_stage, wraplength=850).pack(anchor="w", pady=(0, 10))
        # Reserve a real footer before the expanding notebook.  When the
        # notebook was packed first, small windows compressed this button to
        # a thin blank strip at the lower-right corner.
        monitor_footer = ttk.Frame(self.monitor)
        monitor_footer.pack(fill="x", side="bottom", pady=(8, 0))
        self.open_output_button = ttk.Button(
            monitor_footer, text="打开本次输出目录", command=self.open_output
        )
        self.open_output_button.pack(side="right", ipadx=8, ipady=3)
        views = ttk.Notebook(self.monitor)
        views.pack(fill="both", expand=True)
        overview = ttk.Frame(views, padding=10)
        details = ttk.Frame(views)
        views.add(overview, text="采集进展")
        views.add(details, text="详细日志")
        ttk.Label(overview, text="只显示批次进度、结果和需要处理的问题；完整调用记录在详细日志中。", wraplength=850).pack(anchor="w", pady=(0, 10))
        self.monitor_events = ttk.Treeview(overview, columns=("time", "message"), show="headings")
        self.monitor_events.heading("time", text="时间")
        self.monitor_events.heading("message", text="采集进展")
        self.monitor_events.column("time", width=85, stretch=False)
        self.monitor_events.column("message", width=700)
        event_scroll = ttk.Scrollbar(overview, command=self.monitor_events.yview)
        event_scroll.pack(side="right", fill="y")
        self.monitor_events.configure(yscrollcommand=event_scroll.set)
        self.monitor_events.pack(fill="both", expand=True)
        self.log = tk.Text(details, bg="#142333", fg="#d6e5ee", font=("Consolas", 10), wrap="word", relief="flat", state="disabled")
        self.log.tag_configure("alert", foreground="#ff5c5c", font=("Consolas", 10, "bold"))
        scrollbar = ttk.Scrollbar(details, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(fill="both", expand=True)

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
        keys = ("protocol", "controller_model", "host", "port", "dll", "bridge") if kind == "simulator" else ("base_url", "model", "api_key", "wire_api", "timeout")
        return tuple(config[k] for k in keys)

    def changed(self, *_):
        if getattr(self, "_switching_protocol", False):
            return
        config = self.config()
        for kind, status in (("simulator", self.sim_status), ("llm", self.llm_status)):
            if self.verified.get(kind) != self.signature(config, kind):
                status.set("配置已变化，请测试连接")
        self.preview()

    def select_all(self, selected):
        self.capture_mode.set("all_apis" if selected else "single_api")
        if not selected:
            self.selected_api_set = set()
        self.change_capture_mode()
        self.refresh_api_list()

    def clear_api_selection(self):
        self.selected_api_set.clear()
        self.refresh_api_list()
        self.api_detail.set("尚未选择 API，请从列表中选择一个函数")

    def select_filtered(self):
        self.selected_api_set = set([name for name in self.visible_api_names if not self.api_unavailable_reason(name)][:1])
        self.refresh_api_list()

    def api_unavailable_reason(self, function):
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
        self.catalog = function_manifest()
        query = self.api_search.get().strip().casefold()
        selected_category = self.api_category.get()
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
                self.listbox.insert("end", f"{name}  ·  {category}" + ("  [不可选]" if reason else ""))
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
        chosen = {self.visible_api_names[index] for index in self.listbox.curselection()}
        blocked = [self.api_unavailable_reason(name) for name in chosen if self.api_unavailable_reason(name)]
        if blocked:
            self.refresh_api_list()
            self.api_detail.set(blocked[0])
            messagebox.showinfo("API 不可选", blocked[0], parent=self.root)
            return
        self.selected_api_set = chosen
        self.api_selected_count.set(
            f"已选 {len(self.selected_api_set)} / {self.available_api_count()} 个可选 API；当前显示 {len(self.visible_api_names)} 个"
        )
        if self.listbox.curselection():
            self.show_api_detail(self.visible_api_names[self.listbox.curselection()[0]])
        self.preview()

    def show_api_detail(self, function):
        row = next((item for item in self.catalog if item["function"] == function), None)
        if row is not None:
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
            risk = ("可能改变设备状态，需核验安全条件" if action.startswith(
                ("wr", "del", "clr", "reset", "set", "dwn", "download", "dnc", "start", "stop")
            ) else "操作风险待核验")
            chinese = chinese_api_description(function, row.get("description", ""))
            self.api_detail.set(
                f"{function} · {row.get('category', '未分类')}\n"
                f"中文说明：{chinese}\n"
                f"原始说明：{row.get('description', '暂无说明')}\n"
                f"{header_status}；{risk}（仅提示，不能代替运行时验证）\n"
                f"{self.api_unavailable_reason(function) or '当前型号筛选下可选'}"
            )

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
        for widget in (self.api_picker, self.scene_picker, self.all_api_info):
            widget.pack_forget()
        mode = self.capture_mode.get()
        widget = {"single_api": self.api_picker, "single_scene": self.scene_picker,
                  "all_apis": self.all_api_info,
                  "auto_no_parameter": self.auto_info,
                  "read_only_priority": self.readonly_info}[mode]
        widget.pack(fill="x", padx=(0, 12), pady=(0, 8), after=self.capture_settings)
        self.update_scene_selection()

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
        cycle_error = cycle_start_configuration_error(config)
        if cycle_error:
            self.tabs.select(self.scene_tab)
            self.api_canvas.yview_moveto(1.0)
            self.status.set("请先完成 Cycle Start 定位，再开始采集")
            messagebox.showwarning("需要定位 Cycle Start", cycle_error, parent=self.root)
            return
        errors = validate(config, capture=True)
        for kind, label in (("simulator", "仿真器"), ("llm", "模型")):
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
            self.progress["value"] = 0
            self.tabs.select(self.monitor)
            self.scene_status.set("采集前检查中")
        process = self.process
        def read():
            for line in process.stdout:
                self.events.put(("line", line.rstrip()))
            self.events.put(("exit", process.wait()))
        threading.Thread(target=read, daemon=True).start()

    def append_log(self, text, alert=False):
        for secret in (self.values["api_key"].get(), self.snapshot.get("api_key", "")):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        line = time.strftime("%H:%M:%S") + "  " + text + "\n"
        view_top, view_bottom = self.log.yview()
        follow_tail = view_bottom >= 0.99
        top_line = int(self.log.index("@0,0").split(".")[0])
        self.log.configure(state="normal")
        self.log.insert("end", line, "alert" if alert else "")
        removed_lines = 0
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1000.0")
            removed_lines = 999
        if follow_tail:
            self.log.see("end")
        elif removed_lines:
            self.log.see(f"{max(1, top_line - removed_lines)}.0")
            self.log.yview_moveto(view_top)
        self.log.configure(state="disabled")
        if self.output and self.action == "capture":
            try:
                with (self.output / "desktop.log").open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass

    def handle_event(self, row):
        name, data = row["event"], row["data"]
        self.update_monitor_summary(name, data)
        if name == "operator_action_required":
            self.show_operator_action_prompt(data)
        elif name == "operator_action_verified":
            self.finish_operator_action_prompt("已检测到状态恢复，采集正在继续。", success=True)
        elif name == "operator_action_timeout":
            self.finish_operator_action_prompt("等待超时。依赖该状态的 API 将暂缓，全 API 采集会继续处理其他 API。", success=False)
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
            self.status.set(str(data.get("agent") or data.get("tool") or data.get("next_stage") or name))
        if name in ("success", "partial", "failure"):
            self.outcome = name
            self.status.set(self.monitor_stage.get())
        self.append_log(
            f"{name}  {json.dumps(data, ensure_ascii=False)}",
            alert=name in ("success", "partial", "failure", "scenario_failed"),
        )

    def show_operator_action_prompt(self, data):
        action = str(data.get("action", ""))
        target = str(data.get("target", ""))
        seconds = int(data.get("timeout_seconds", 60))
        if action == "remote":
            title = "需要切换仿真器模式"
            instruction = "请在 FANUC 主操作面板点击 REMOTE。系统会自动检查模式和控制器状态，符合条件后继续采集。"
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
        self.operator_prompt_timer = tk.StringVar(value=f"自动检测中，最多等待 {seconds} 秒。")
        ttk.Label(body, textvariable=self.operator_prompt_timer, foreground="#176a83").pack(anchor="w", pady=(14, 10))
        ttk.Button(body, text="知道了", command=dialog.destroy).pack(anchor="e")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max(30, (self.root.winfo_width() - dialog.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(30, (self.root.winfo_height() - dialog.winfo_height()) // 3)
        dialog.geometry(f"+{x}+{y}")

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
            self.monitor_stage.set(stages.get(stage, "正在处理采集任务"))
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
                message = "采集未完成 · " + ("、".join(codes) if codes else message[:200]) + "；已保存的数据保留，具体原因见详细日志。"
            elif name == "partial":
                message = "采集部分完成 · " + message[:190] + "；流量质量可在离线审计中查看。"
            self.monitor_stage.set(message[:240])
        if message:
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
            elif kind == "line":
                if value.startswith(EVENT):
                    try:
                        self.handle_event(json.loads(value[len(EVENT):]))
                    except (ValueError, KeyError, TypeError):
                        self.append_log(value)
                else:
                    self.append_log(value, alert=any(word in value.lower() for word in ("error", "failed", "failure", "失败")))
            else:
                self.busy = False
                for button in self.start_buttons:
                    button.configure(state="normal" if protocol_profile(self.config())["available"] else "disabled")
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
                    if self.action == "capture":
                        self.scene_status.set("所选 API 采集部分完成" if self.outcome == "partial" else "所选 API 采集完成")
                else:
                    if self.action in ("simulator", "llm", "preflight"):
                        for test in (("simulator", "llm") if self.action == "preflight" else (self.action,)):
                            self.verified.pop(test, None)
                            (self.sim_status if test == "simulator" else self.llm_status).set("测试失败，详见监控日志")
                    if self.outcome != "failure":
                        self.status.set(f"后端意外退出（{value}），请查看日志")
                    self.tabs.select(self.monitor)
                    if self.action == "capture":
                        self.scene_status.set("采集未完成 · 已完成 API 与数据已保留")
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
        path = self.output or Path(self.values["output"].get())
        if path.is_dir():
            os.startfile(path)
        else:
            messagebox.showinfo("输出目录", "尚未生成输出目录")

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
