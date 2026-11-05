from smart_traffic_agent.agent_tools.sdk_type_contract import GetSdkTypeContractTool, SdkTypeContractInput


def test_sdk_type_contract_resolves_existing_rdparar_type():
    result = GetSdkTypeContractTool().invoke(SdkTypeContractInput("cnc_rdparar", "param"))
    assert result["ok"] is True
    assert result["prototype"].startswith("FWLIBAPI short WINAPI cnc_rdparar")
    assert "IODBPSD" in result["candidate_sdk_types"]
    assert result["type_definitions"]["IODBPSD"]["fields"]
    assert result["requires_runtime_shape"] is True


def test_sdk_type_contract_rejects_unknown_api():
    result = GetSdkTypeContractTool().invoke(SdkTypeContractInput("cnc_not_in_sdk"))
    assert result["ok"] is False
    assert "prototype" in result["error"]


def test_sdk_type_contract_resolves_direct_payload_types():
    result = GetSdkTypeContractTool().invoke(SdkTypeContractInput("cnc_prstwkcd"))
    assert result["ok"] is True
    assert "IDBWRA" in result["candidate_sdk_types"]
    assert result["type_definitions"]["IDBWRA"]["fields"]

    axis_names = GetSdkTypeContractTool().invoke(SdkTypeContractInput("cnc_rdaxisname"))
    assert axis_names["ok"] is True
    assert "ODBAXISNAME" in axis_names["candidate_sdk_types"]
