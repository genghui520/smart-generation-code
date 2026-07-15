param(
    [string]$VsInstall = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools",
    [string]$OutDir = ".tools\focas_bridge_cpp"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $ProjectRoot "cpp\focas_bridge\focas_bridge.cpp"
$OutPath = Join-Path $ProjectRoot $OutDir
$Exe = Join-Path $OutPath "focas_bridge.exe"
$VcVars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvarsall.bat"

if (-not (Test-Path $Source)) {
    throw "Source file not found: $Source"
}
if (-not (Test-Path $VcVars)) {
    throw "vcvarsall.bat not found: $VcVars"
}

New-Item -ItemType Directory -Force -Path $OutPath | Out-Null

$Command = "`"$VcVars`" x86 && cl /nologo /EHsc /W4 /O2 /std:c++17 /Fe:`"$Exe`" `"$Source`""
cmd.exe /c $Command

if (-not (Test-Path $Exe)) {
    throw "Build failed: $Exe was not created"
}

Write-Output "Built: $Exe"
