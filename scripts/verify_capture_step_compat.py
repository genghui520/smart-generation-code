"""Compile the original failed callback with the current harness; never execute it."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_traffic_agent.focas_capture_runtime import render_capture_harness, validate_production_source
from smart_traffic_agent.execution_contracts import ExecutionContract, MOTION_RUNTIME_APIS, SESSION_APIS
from smart_traffic_agent.integrations.ncguide import default_focas_header_dir
from smart_traffic_agent.agents.code_generator import preflight_compile_generated_cpp

root = Path(__file__).resolve().parents[1]
original = root / "runs/desktop/20261015_154021_041a89/batch_001/generated"
out = root / "runs/diagnostics/capture_step_compat"
out.mkdir(parents=True, exist_ok=True)
fragment = (original / "scenario_fragment.inc").read_text(encoding="utf-8")
nc = (original / "program.nc").read_text(encoding="utf-8")
header = (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1")
contract = ExecutionContract(("cnc_acts",), tuple(sorted(MOTION_RUNTIME_APIS | SESSION_APIS)), True)
source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program=nc)
errors = validate_production_source(source, fragment, header, contract, nc)
assert not errors, errors
(out / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
cpp = out / "api_script.cpp"
cpp.write_text(source, encoding="utf-8")
error = preflight_compile_generated_cpp(cpp, out)
if error:
    print(error)
    raise SystemExit(1)
print("Original failing cnc_acts fragment: validation PASS, MSVC compilation PASS. No executable run.")
