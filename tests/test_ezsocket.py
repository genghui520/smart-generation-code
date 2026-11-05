from __future__ import annotations

from unittest.mock import patch

from smart_traffic_agent.integrations.ezsocket import (
    EzSocketComError,
    call_ezsocket_com_api,
)


class FakeEzSocketCom:
    def __init__(self, status_result=(0, 1)):
        self.status_result = status_result
        self.calls = []

    def SetTCPIPProtocol(self, host, port):
        self.calls.append(("SetTCPIPProtocol", host, port))
        return 0

    def Open2(self, system_type, machine, timeout_units, hostname):
        self.calls.append(("Open2", system_type, machine, timeout_units, hostname))
        return 0

    def Status_GetRunStatus(self, index):
        self.calls.append(("Status_GetRunStatus", index))
        return self.status_result

    def Close(self):
        self.calls.append(("Close",))
        return 0


def test_com_api_executes_documented_read_only_sequence():
    fake = FakeEzSocketCom(status_result=(1, 0))
    with patch(
        "smart_traffic_agent.integrations.ezsocket.test_ezsocket_connection",
        return_value={"host": "192.168.139.130", "port": 683},
    ):
        result = call_ezsocket_com_api(
            "192.168.139.130", 683, com_factory=lambda: fake,
        )

    assert result["transport"] == "ezsocket-com"
    assert result["run_status"] == 1
    assert result["return_code"] == 0
    assert fake.calls == [
        ("SetTCPIPProtocol", "192.168.139.130", 683),
        ("Open2", 9, 1, 10, "EZNC_LOCALHOST"),
        ("Status_GetRunStatus", 1),
        ("Close",),
    ]


def test_com_api_closes_after_api_failure():
    fake = FakeEzSocketCom(status_result=(12, 0))
    with patch(
        "smart_traffic_agent.integrations.ezsocket.test_ezsocket_connection",
        return_value={"host": "192.168.139.130", "port": 683},
    ):
        try:
            call_ezsocket_com_api("192.168.139.130", 683, com_factory=lambda: fake)
        except EzSocketComError as exc:
            assert "Status_GetRunStatus" in str(exc)
        else:
            raise AssertionError("expected the simulated COM API failure")
    assert fake.calls[-1] == ("Close",)
