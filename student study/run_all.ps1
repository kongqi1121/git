# run_all.ps1 -- one-click runner for Windows (optional)
# =========================================================
# Purpose: run the whole pipeline with UTF-8 output and save the console log to
#          results/full_run_log.txt (the file referenced by README.md and
#          experiments.md).
#
# Usage (from the project root):
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#
# Notes:
#   * This script is NOT required: `python main.py --auto` already runs everything.
#     The script only adds UTF-8 console setup + automatic log capture.
#   * This file is intentionally ASCII-only. Windows PowerShell 5.1 reads files
#     without a BOM using the system ANSI code page (GBK on Chinese Windows),
#     which corrupts non-ASCII characters and can even break parsing.
#     Keeping it ASCII-only makes it safe on both PowerShell 5.1 and 7.
#   * The Chinese disclaimer is printed by main.py itself (UTF-8 safe).
# =========================================================

$ErrorActionPreference = "Stop"

# 1) Force UTF-8 for Python output and for the console (avoids GBK mojibake)
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { $null = $_ }

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

New-Item -ItemType Directory -Force -Path "$root\results" | Out-Null
$logPath = "$root\results\full_run_log.txt"

Write-Host "Running: python main.py --auto" -ForegroundColor Cyan
Write-Host "Log file: results\full_run_log.txt" -ForegroundColor Cyan
Write-Host ""

# 2) Capture merged stdout + stderr while keeping the real exit code
$output = & python main.py --auto 2>&1
$code = $LASTEXITCODE
$output | ForEach-Object { Write-Host $_ }

# 3) Write the log as UTF-8 without BOM
[System.IO.File]::WriteAllLines($logPath, [string[]]$output, (New-Object System.Text.UTF8Encoding($false)))

Write-Host ""
if ($code -eq 0) {
    Write-Host "Pipeline finished successfully. Log: results\full_run_log.txt" -ForegroundColor Green
} else {
    Write-Host "Pipeline exited with code $code. Please check the log." -ForegroundColor Red
}
exit $code
