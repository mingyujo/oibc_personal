param([switch]$Neural)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv .venv }
    else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python3.12 environment creation failed' }
}
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip update failed' }
if ($Neural) { & ./.venv/Scripts/python.exe -m pip install -e '.[neural]' }
else { & ./.venv/Scripts/python.exe -m pip install -e . }
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& ./.venv/Scripts/python.exe -m oibc_portable doctor
exit $LASTEXITCODE
