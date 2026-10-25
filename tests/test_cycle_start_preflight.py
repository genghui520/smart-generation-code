from unittest.mock import MagicMock, patch

import pytest

from smart_traffic_agent import desktop_backend as backend


def config(function="cnc_absolute", **changes):
    return {"protocol": "focas", "capture_mode": "single_api", "selected_apis": [function],
            "ui_start": False, "start_x": "", "start_y": "", "window_title": "Main Panel", **changes}


@pytest.mark.parametrize("function", ["cnc_absolute", "cnc_actf", "cnc_rddynamic"])
def test_motion_requires_localization_even_when_ui_start_unchecked(function):
    cfg = config(function)
    assert backend.selection_requires_cycle_start(cfg)
    assert "开始定位 Cycle Start" in backend.cycle_start_configuration_error(cfg)
    assert any("开始定位 Cycle Start" in error for error in backend.validate(cfg, capture=True))


@pytest.mark.parametrize("function", ["cnc_rdtofs", "cnc_rdcount", "cnc_download3", "cnc_statinfo"])
def test_nonmotion_without_localization_is_not_blocked(function):
    assert not backend.cycle_start_configuration_error(config(function))


def test_calibration_requires_enabled_coordinates_and_window():
    cfg = config(ui_start=True, start_x="-120", start_y="0")
    assert not backend.cycle_start_configuration_error(cfg)
    for changes in ({"start_x": ""}, {"start_y": "bad"}, {"window_title": ""}, {"ui_start": False}):
        assert backend.cycle_start_configuration_error(cfg | changes)


def test_all_api_and_scene_modes():
    assert backend.selection_requires_cycle_start(config(capture_mode="all_apis"))
    with patch.object(backend, "scenarios", return_value=[
        {"scenario_id": "motion", "main_apis": ["cnc_absolute/cnc_actf"]},
        {"scenario_id": "tool", "main_apis": ["cnc_rdtofs"]},
        {"scenario_id": "run", "main_apis": ["cnc_statinfo"],
         "operation_variants": [{"operation_template": [{"protocol_function": "ncguide_ui_cycle_start"}]}]},
    ]):
        for name, expected in (("motion", True), ("tool", False), ("run", True)):
            assert backend.selection_requires_cycle_start(config(capture_mode="single_scene", selected=[name])) == expected


def test_start_warns_before_launch_and_allows_after_localization():
    from smart_traffic_agent.desktop import DesktopApp
    app = DesktopApp.__new__(DesktopApp)
    app.busy = False
    cfg = config()
    app.config = lambda: cfg
    app.tabs = MagicMock()
    app.api_canvas = MagicMock()
    app.status = MagicMock()
    app.root = MagicMock()
    app.scene_tab = MagicMock()
    app.launch = MagicMock()
    app.signature = lambda *_: "verified"
    app.verified = {"simulator": "verified", "llm": "verified"}
    with patch("smart_traffic_agent.desktop.messagebox.showwarning") as warning:
        app.start()
        warning.assert_called_once()
        app.launch.assert_not_called()
        app.tabs.select.assert_called_once_with(app.scene_tab)
    cfg.update(ui_start=True, start_x="123", start_y="456")
    with patch("smart_traffic_agent.desktop.validate", return_value=[]):
        app.start()
    app.launch.assert_called_once_with("capture")
