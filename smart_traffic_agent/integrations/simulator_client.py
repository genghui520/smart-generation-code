from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulatedCncClient:
    program_name: str | None = None
    running: bool = False
    tick: int = 0
    parameters: dict[int, int | float | str] = field(default_factory=dict)

    def call(self, interface_name: str, parameters: dict) -> dict:
        handler = getattr(self, f"_handle_{interface_name}", self._handle_unknown)
        return handler(parameters)

    def _handle_UploadProgram(self, parameters: dict) -> dict:
        self.program_name = parameters.get("program_name", "O1001")
        return {"status_code": 0, "program_name": self.program_name, "uploaded": True}

    def _handle_SelectProgram(self, parameters: dict) -> dict:
        self.program_name = parameters.get("program_name", self.program_name)
        return {"status_code": 0, "program_name": self.program_name, "selected": True}

    def _handle_ReadProgramNumber(self, parameters: dict) -> dict:
        program_number = 0
        if self.program_name:
            digits = "".join(ch for ch in self.program_name if ch.isdigit())
            program_number = int(digits or 0)
        return {
            "status_code": 0,
            "running_program": program_number if self.running else 0,
            "main_program": program_number,
        }

    def _handle_StartProgram(self, parameters: dict) -> dict:
        self.running = True
        self.tick = 0
        return {"status_code": 0, "run_status": "running"}

    def _handle_StopProgram(self, parameters: dict) -> dict:
        self.running = False
        return {"status_code": 0, "run_status": "completed"}

    def _handle_ReadRunStatus(self, parameters: dict) -> dict:
        return {"status_code": 0, "run_status": "running" if self.running else "idle"}

    def _handle_ReadPosition(self, parameters: dict) -> dict:
        self.tick += 1
        axes = parameters.get("axes", ["X", "Y", "Z"])
        position = {
            "X": round(self.tick * 1.5, 3),
            "Y": round(self.tick * 0.8, 3),
            "Z": round(5 - self.tick * 0.2, 3),
        }
        return {"status_code": 0, "position": {axis: position.get(axis, 0.0) for axis in axes}}

    def _handle_ReadFeedSpeed(self, parameters: dict) -> dict:
        return {"status_code": 0, "feed_speed": 240 + self.tick * 10}

    def _handle_ReadSpindleSpeed(self, parameters: dict) -> dict:
        self.tick += 1
        return {"status_code": 0, "spindle_speed": 800 + self.tick * 100}

    def _handle_ReadParameter(self, parameters: dict) -> dict:
        number = int(parameters.get("parameter_no", 0))
        return {"status_code": 0, "parameter_no": number, "value": self.parameters.get(number, 0)}

    def _handle_WriteParameter(self, parameters: dict) -> dict:
        number = int(parameters.get("parameter_no", 0))
        value = parameters.get("value", 0)
        self.parameters[number] = value
        return {"status_code": 0, "parameter_no": number, "value": value, "written": True}

    def _handle_ReadAlarm(self, parameters: dict) -> dict:
        return {"status_code": 0, "alarms": []}

    def _handle_unknown(self, parameters: dict) -> dict:
        return {"status_code": 404, "error": "unknown interface"}
