#!/usr/bin/env pwsh
# 一键启动 infraredComp 后端(FastAPI :8091) + 前端(Vite :3001)
# 用法: powershell -ExecutionPolicy Bypass -File start_services.ps1
# 或直接在 PowerShell 中: .\start_services.ps1
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT

# 选择可用的 Node（pnpm 需要 Node 22+）
function Pick-Node {
    $candidates = @(
        "C:\Program Files\nodejs\node.exe",
        "C:\Program Files (x86)\nodejs\node.exe",
        "$env:LOCALAPPDATA\fnm_multishells\node.exe",
        "$env:ProgramData\nvm\node.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd) { return $nodeCmd.Source }
    throw "node not found"
}

$NODE_BIN = Pick-Node
$NODE_DIR = Split-Path -Parent $NODE_BIN
$env:PATH = "$NODE_DIR;$env:PATH"
$nodeVersion = & $NODE_BIN -v
Write-Host "[start] node: $nodeVersion  dir: $NODE_DIR"

# corepack 在 Windows 上是 corepack.cmd，通过 cmd /c 调用最稳
$COREPACK = "$NODE_DIR\corepack.cmd"
if (!(Test-Path $COREPACK)) {
    $COREPACK = "corepack"
}

# @parcel/watcher 的 native build 脚本（pnpm 11+ 默认忽略）。非交互式跑会卡住，
# 且 vite 实测不批准也能跑（文件监听回退轮询）。需要时手动 `pnpm approve-builds`。
# Set-Location "$ROOT\web"
# & $COREPACK pnpm approve-builds @parcel/watcher 2>$null | Out-Null
# Set-Location $ROOT

# Benchmark subprocess python: learned codecs (ssf2020/img-*/dcvc_rt) need a GPU
# torch env. The server itself runs under `uv run` (CPU torch), so /run Popen's
# the benchmark with INFRACOMP_BENCH_PYTHON when set. Prefer the `compression`
# conda env (CUDA) if present; else fall back to the uv venv (CPU, slow for learned).
if (-not $env:INFRACOMP_BENCH_PYTHON) {
    $condaCompression = "$env:USERPROFILE\.conda\envs\compression\python.exe"
    if (Test-Path $condaCompression) {
        $env:INFRACOMP_BENCH_PYTHON = $condaCompression
        Write-Host "[start] bench python (GPU, learned codecs): $condaCompression"
    } else {
        Write-Host "[start] no compression conda env; learned codecs will run on CPU. Set INFRACOMP_BENCH_PYTHON to a CUDA python for GPU."
    }
} else {
    Write-Host "[start] bench python (override): $env:INFRACOMP_BENCH_PYTHON"
}

# 后端 — 用 `cmd /c "... > log 2>&1"` 合并 stdout/stderr 到单文件；
# 不能用 Start-Process 的 -RedirectStandardOutput + -RedirectStandardError 指向同一文件
# （PowerShell 会抛 "RedirectStandardOutput 和 RedirectStandardError 指定了同一个文件"）。
Write-Host "[start] backend: uv run uvicorn server.main:app --port 8091 (logs: backend.log)"
$backend = Start-Process -FilePath "cmd" -ArgumentList "/c", "uv run uvicorn server.main:app --host 0.0.0.0 --port 8091 > `"$ROOT\backend.log`" 2>&1" -WindowStyle Hidden -PassThru
Write-Host "[start] backend pid=$($backend.Id)"

# 前端 — 同样 cmd /c 合并日志；corepack 已在 PATH（$NODE_DIR 已前置），但保留全路径更稳。
Write-Host "[start] frontend: cd web && $COREPACK pnpm dev --port 3001 (logs: frontend.log)"
$frontend = Start-Process -FilePath "cmd" -ArgumentList "/c", "cd /d `"$ROOT\web`" && `"$COREPACK`" pnpm dev --port 3001 > `"$ROOT\frontend.log`" 2>&1" -WindowStyle Hidden -PassThru
Write-Host "[start] frontend pid=$($frontend.Id)"

Write-Host ""
Write-Host "  backend  -> http://localhost:8091  (docs /api/docs)"
Write-Host "  frontend -> http://localhost:3001/infraredComp/"
Write-Host "  logs     -> backend.log  frontend.log"
Write-Host ""
Write-Host "Press Ctrl+C to stop both services."

# 清理函数
function Stop-Services {
    Write-Host "`n[stop] terminating..."
    if ($backend -and !$backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontend -and !$frontend.HasExited) {
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    }
    exit
}

# 捕获 Ctrl+C / 窗口关闭
[Console]::TreatControlCAsInput = $true
$Host.UI.RawUI.FlushInputBuffer()

while ($true) {
    # 非交互式/headless 运行时 KeyAvailable 可能抛错（"application does not have a console"），try/catch 容错。
    try {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if (($key.Modifiers -band [ConsoleModifiers]::Control) -and ($key.Key -eq "C")) {
                Stop-Services
            }
        }
    } catch { }

    if ($backend.HasExited) {
        Write-Host "[stop] backend exited unexpectedly (exit code $($backend.ExitCode))"
        Stop-Services
    }
    if ($frontend.HasExited) {
        Write-Host "[stop] frontend exited unexpectedly (exit code $($frontend.ExitCode))"
        Stop-Services
    }

    Start-Sleep -Milliseconds 500
}
