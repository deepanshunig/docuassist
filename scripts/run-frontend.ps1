$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the project virtual environment first.' }
$taskTemporaryDirectory = Join-Path $taskProjectRoot '.cache\tmp'
New-Item -ItemType Directory -Force -Path $taskTemporaryDirectory | Out-Null
$env:TEMP = $taskTemporaryDirectory
$env:TMP = $taskTemporaryDirectory
Push-Location $taskProjectRoot
try {
    & $taskPython -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8511 --server.headless true --browser.gatherUsageStats false
}
finally { Pop-Location }
