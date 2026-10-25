@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "run_desktop.pyw"
) else (
    echo Missing .venv. Please install the project Python environment first.
    pause
)
