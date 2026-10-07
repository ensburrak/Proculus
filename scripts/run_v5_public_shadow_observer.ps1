[CmdletBinding()]
param(
    [switch]$Loop,
    [ValidateRange(5, 240)]
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# This observer is deliberately public-data-only. Remove every OKX credential
# from the child process even if the operator shell happens to contain them.
@(
    "OKX_API_KEY",
    "OKX_API_SECRET",
    "OKX_API_PASSPHRASE",
    "OKX_LIVE_API_KEY",
    "OKX_LIVE_API_SECRET",
    "OKX_LIVE_API_PASSPHRASE"
) | ForEach-Object {
    Remove-Item ("Env:" + $_) -ErrorAction SilentlyContinue
}

$env:LIVE_MODE = "0"
$env:WRAPPER_DRY_RUN = "1"
$env:SHADOW_MODE = "1"

function Resolve-ObserverPython {
    $py = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $py -and $py.Path) {
        & $py.Path -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            return @($py.Path, "-3.12")
        }
    }

    foreach ($candidate in @("python", "python3")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $cmd -or -not $cmd.Path) {
            continue
        }
        & $cmd.Path -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            return @($cmd.Path)
        }
    }

    throw "[V5_PUBLIC_SHADOW] Python 3.12 bulunamadi."
}

function Assert-PublicOnlyEnvironment {
    foreach ($name in @(
        "OKX_API_KEY",
        "OKX_API_SECRET",
        "OKX_API_PASSPHRASE",
        "OKX_LIVE_API_KEY",
        "OKX_LIVE_API_SECRET",
        "OKX_LIVE_API_PASSPHRASE"
    )) {
        if (Test-Path ("Env:" + $name)) {
            throw "[V5_PUBLIC_SHADOW] SAFETY FAIL: credential env kaldi: $name"
        }
    }
    if ($env:LIVE_MODE -ne "0") {
        throw "[V5_PUBLIC_SHADOW] SAFETY FAIL: LIVE_MODE must be 0."
    }
    if ($env:WRAPPER_DRY_RUN -ne "1") {
        throw "[V5_PUBLIC_SHADOW] SAFETY FAIL: WRAPPER_DRY_RUN must be 1."
    }
    if ($env:SHADOW_MODE -ne "1") {
        throw "[V5_PUBLIC_SHADOW] SAFETY FAIL: SHADOW_MODE must be 1."
    }
}

function Invoke-ObserverOnce {
    param([string[]]$PythonCommand)

    Assert-PublicOnlyEnvironment

    $exe = $PythonCommand[0]
    $prefix = @()
    if ($PythonCommand.Count -gt 1) {
        $prefix = $PythonCommand[1..($PythonCommand.Count - 1)]
    }

    Write-Host ""
    Write-Host ("[{0}] V5 public-mainnet observer basliyor" -f (Get-Date -Format o))
    Write-Host "  LIVE_MODE=0"
    Write-Host "  WRAPPER_DRY_RUN=1"
    Write-Host "  SHADOW_MODE=1"
    Write-Host "  OKX credentials=unset"
    Write-Host "  order endpoints=not used"

    & $exe @prefix "tools5_public_shadow_observer.py"
    if ($LASTEXITCODE -ne 0) {
        throw "[V5_PUBLIC_SHADOW] observer exit code $LASTEXITCODE"
    }

    & $exe @prefix "tools5_shadow_report.py"
    if ($LASTEXITCODE -ne 0) {
        throw "[V5_PUBLIC_SHADOW] evidence report exit code $LASTEXITCODE"
    }

    Write-Host ("[{0}] Observer tamamlandi." -f (Get-Date -Format o))
    Write-Host "  reports5_public_shadow_observer_status.json"
    Write-Host "  reports5_prospective_shadow_status.json"
    Write-Host "  reports5_prospective_shadow_outcomes.json"
}

$pythonCommand = @(Resolve-ObserverPython)

if (-not $Loop) {
    Invoke-ObserverOnce -PythonCommand $pythonCommand
    exit 0
}

while ($true) {
    Invoke-ObserverOnce -PythonCommand $pythonCommand
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
