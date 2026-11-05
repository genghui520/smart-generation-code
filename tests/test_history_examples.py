import json

from smart_traffic_agent.history_examples import retrieve_verified_api_examples


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value, encoding="utf-8")


def test_retrieves_only_successful_wire_matched_examples(tmp_path):
    batch = tmp_path / "run-1" / "batch_001"
    _write(batch / "plan.json", {"target_api": "cnc_relative", "target_environment": "0i-D"})
    _write(batch / "execution/collection_result.json", {
        "status": "collected",
        "status_fields": {"workflow_status": "completed", "api_execution_status": "success", "capture_status": "matched", "collection_status": "collected"},
        "evidence": {"target_apis_with_wire_evidence": ["cnc_relative"]},
    })
    rows = [
        {"api_name": "cnc_statinfo", "return_code": 0, "traffic_status": "matched", "transaction_id": "tx0", "input": {"params": "FlibHndl=1"}, "output": {"success_data": "run=0"}},
        {"api_name": "cnc_relative", "return_code": 0, "traffic_status": "matched", "transaction_id": "tx1", "input": {"params": "FlibHndl=1;axis=-1;length=132"}, "output": {"return_desc": "EW_OK", "success_data": "data=[1]"}},
    ]
    _write(batch / "execution/data/api_calls.jsonl", "\n".join(json.dumps(row) for row in rows))
    _write(batch / "generated/scenario_fragment.inc", "api.cnc_relative(handle, -1, relLen, &relative);\n")
    examples = retrieve_verified_api_examples("cnc_relative", output_dir=tmp_path)
    assert len(examples) == 1
    assert examples[0]["pcap_matched"] is True
    assert "axis=-1" in examples[0]["calls"][0]["parameters"]
    assert "api.cnc_relative" in examples[0]["cpp_call_snippet"]


def test_excludes_failed_or_unmatched_history(tmp_path):
    batch = tmp_path / "run-1" / "batch_001"
    _write(batch / "plan.json", {"target_api": "cnc_relative"})
    _write(batch / "execution/collection_result.json", {
        "status": "failed",
        "status_fields": {"workflow_status": "failed", "api_execution_status": "failed", "capture_status": "unmatched", "collection_status": "failed"},
        "evidence": {"target_apis_with_wire_evidence": []},
    })
    _write(batch / "execution/data/api_calls.jsonl", json.dumps({"api_name": "cnc_relative", "return_code": 0, "traffic_status": "matched", "transaction_id": "tx"}))
    assert retrieve_verified_api_examples("cnc_relative", output_dir=tmp_path) == []
