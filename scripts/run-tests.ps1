$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
$taskTemporaryDirectory = Join-Path $taskProjectRoot '.cache\tmp'
New-Item -ItemType Directory -Force -Path $taskTemporaryDirectory | Out-Null
$env:TEMP = $taskTemporaryDirectory
$env:TMP = $taskTemporaryDirectory
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
Push-Location $taskProjectRoot
try {
    & $taskPython -m pytest --basetemp .cache/pytest -q
    $taskTestExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $taskTestExitCode
