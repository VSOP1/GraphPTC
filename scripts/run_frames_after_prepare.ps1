param(
    [Parameter(Mandatory = $true)]
    [int]$PrepareProcessId
)

$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repoDir ".venv\Scripts\graphptc.exe"
$runDir = Join-Path $repoDir "runs\frames\test"
New-Item -ItemType Directory -Force $runDir | Out-Null

$preparation = Get-Process -Id $PrepareProcessId
$preparation.WaitForExit()
if (-not (Test-Path -LiteralPath (Join-Path $repoDir "data\frames\wikipedia-20230601\manifest.json"))) {
    throw "Official FRAMES Wikipedia preparation failed"
}

$retriever = Start-Process -FilePath wsl.exe `
    -ArgumentList @(
        "-d", "Ubuntu-22.04", "-e", "bash",
        "/mnt/d/GraphPTC/scripts/services/run_frames_retriever.sh"
    ) `
    -RedirectStandardOutput (Join-Path $runDir "retriever.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "retriever.stderr.log") `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 5
if ($retriever.HasExited) {
    throw "FRAMES retriever failed to start"
}

$probe = Start-Process -FilePath $runner `
    -ArgumentList @(
        "probe-frames-wikipedia", "--config", "configs/frames/graphptc-test.toml"
    ) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "probe.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "probe.stderr.log") `
    -WindowStyle Hidden -PassThru -Wait
if ($probe.ExitCode -ne 0) {
    throw "FRAMES official-corpus probe failed"
}

git -C $repoDir add -- `
    .gitignore `
    configs/frames `
    data/frames/test.tsv `
    docs/benchmarks/frames.md `
    scripts/data/prepare_frames_wikipedia.py `
    scripts/data/prepare_frames_wikipedia.sh `
    scripts/run_frames_after_prepare.ps1 `
    scripts/run_frames_test_paired.ps1 `
    scripts/services/frames_retriever.py `
    scripts/services/run_frames_retriever.sh `
    scripts/services/setup_frames_retriever.sh `
    src/graphptc/cli.py `
    src/graphptc/config.py `
    src/graphptc/frames_benchmark.py
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stage the FRAMES adapter"
}
git -C $repoDir commit -m "Add official FRAMES benchmark adapter"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to commit the FRAMES adapter"
}

& (Join-Path $repoDir "scripts\run_frames_test_paired.ps1") -Restart
