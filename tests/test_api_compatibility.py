from unittest.mock import patch
from types import SimpleNamespace

from smart_traffic_agent.api_compatibility import unavailable_reason, compatibility_data, detect_sdk_model, SDK_UNKNOWN, f0id_api_names, f0id_scope_metadata
from smart_traffic_agent import desktop_backend as backend
from scripts.build_api_compatibility import parse_matrix


def test_source_matrix_and_transport():
    assert len(compatibility_data()) == 717
    config = {"controller_model": "0i-D"}
    assert not unavailable_reason("cnc_absolute", config)
    assert "不适用" in unavailable_reason("cnc_rdnodenum", config)
    assert "适用性尚未确认" in unavailable_reason("not_a_function", config)
    assert not unavailable_reason("cnc_rdnodenum", {})
    assert not unavailable_reason("cnc_rdnodenum", dict(config, protocol="s7comm"))


def test_parser_does_not_treat_missing_as_supported():
    result = parse_matrix(b'<root><chapter><item><fname>cnc_test</fname><e0id>X</e0id><h0id>O</h0id></item></chapter></root>')
    assert result == {"cnc_test": {"e0id": "X", "h0id": "O"}}


def test_f0id_scope_is_manifest_compatibility_intersection():
    names = f0id_api_names()
    assert len(names) == 308
    assert "cnc_rdnodenum" not in names
    assert "cnc_absolute" in names
    assert f0id_scope_metadata()["api_count"] == 308


def test_all_api_counts_prompts_and_validation(tmp_path):
    config = backend.defaults() | {"controller_model": "0i-D", "capture_mode": "all_apis", "api_key": "test"}
    (tmp_path / "Fwlib32.h").write_text("#define FS0IDD\n", encoding="ascii")
    config["header"] = str(tmp_path)
    names = backend.selected_api_names(config)
    assert len(names) == 308
    assert len(names) < len(backend.function_manifest())
    assert "cnc_rdnodenum" not in names
    assert "cnc_absolute" in names
    prompt = backend.render_full_api_prompt(config)
    assert "cnc_rdnodenum" not in prompt
    assert str(len(names)) in prompt
    config.update(capture_mode="single_api", selected_apis=["cnc_rdnodenum"])
    assert any("不适用" in error for error in backend.validate(config, capture=True))
    assert backend.selected_api_names(config) == []


def test_model_switch_click_keyboard_and_all_api_ui(tmp_path):
    import tkinter as tk
    from smart_traffic_agent.desktop import DesktopApp
    root = tk.Tk()
    root.withdraw()
    try:
        app = DesktopApp(root)
        app.selected_api_set = {"cnc_rdnodenum"}
        (tmp_path / "Fwlib32.h").write_text("#define FS0IDD\n", encoding="ascii")
        app.values["header"].set(str(tmp_path))
        assert app.values["controller_model"].get() == "0i-D"
        assert not hasattr(app, "controller_combo")
        assert not app.selected_api_set
        app.api_search.set("cnc_rdnodenum")
        root.update_idletasks()
        assert app.listbox.itemcget(0, "foreground") == "#9ca3af"
        with patch("smart_traffic_agent.desktop.messagebox.showinfo") as notice:
            app.listbox.selection_set(0)
            app.on_api_selection()
            assert not app.selected_api_set
            notice.assert_called_once()
        with patch("smart_traffic_agent.desktop.messagebox.showinfo") as notice:
            bounds = app.listbox.bbox(0)
            assert app.on_api_click(SimpleNamespace(y=bounds[1] + 1)) == "break"
            notice.assert_called_once()
        app.capture_mode.set("all_apis")
        app.change_capture_mode()
        assert len(app.config()["selected_apis"]) == app.available_api_count()
        assert "cnc_rdnodenum" not in app.config()["selected_apis"]
        assert str(app.available_api_count()) in app.all_api_info.cget("text")
        other = tmp_path / "other"
        other.mkdir()
        (other / "Fwlib32.h").write_text("#define FS0ID\n", encoding="ascii")
        app.values["header"].set(str(other))
        assert app.values["controller_model"].get() == "0i-A"
        assert len(app.config()["selected_apis"]) == app.available_api_count()
    finally:
        root.destroy()


def test_sdk_content_wins_over_directory_name_and_saved_model(tmp_path):
    path = tmp_path / "30i"
    path.mkdir()
    header = path / "Fwlib32.h"
    header.write_text("/* #define FS0ID */\n#define FS0IDD\n", encoding="ascii")
    assert detect_sdk_model(str(path)) == "0i-D"
    assert "不适用" in unavailable_reason("cnc_rdnodenum", {"header": str(path), "controller_model": "0i-F"})
    header.write_text("#define FS30D\n", encoding="ascii")
    assert detect_sdk_model(str(path)) == SDK_UNKNOWN
    assert detect_sdk_model(str(tmp_path / "missing")) == SDK_UNKNOWN
