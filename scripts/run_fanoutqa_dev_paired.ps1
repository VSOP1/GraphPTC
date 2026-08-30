$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repo ".venv\Scripts\graphptc.exe"
$graphConfig = Join-Path $repo "configs\fanoutqa\graphptc-dev.toml"
$baselineConfig = Join-Path $repo "configs\fanoutqa\fewshot-ptc-dev.toml"
$logDir = Join-Path $repo "runs\fanoutqa\dev\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$graph = Start-Process -FilePath $runner -ArgumentList @(
    "run-fanoutqa", "--config", $graphConfig, "--restart"
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "graphptc-generate.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "graphptc-generate.stderr.log")
$baseline = Start-Process -FilePath $runner -ArgumentList @(
    "run-fanoutqa", "--config", $baselineConfig, "--restart"
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "fewshot-ptc-generate.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "fewshot-ptc-generate.stderr.log")
$graph.WaitForExit()
$baseline.WaitForExit()

$graphEval = Start-Process -FilePath $runner -ArgumentList @(
    "evaluate-fanoutqa", "--config", $graphConfig
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "graphptc-evaluate.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "graphptc-evaluate.stderr.log")
$baselineEval = Start-Process -FilePath $runner -ArgumentList @(
    "evaluate-fanoutqa", "--config", $baselineConfig
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "fewshot-ptc-evaluate.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "fewshot-ptc-evaluate.stderr.log")
$graphEval.WaitForExit()
$baselineEval.WaitForExit()

$compare = Start-Process -FilePath $runner -ArgumentList @(
    "compare-fanoutqa",
    "--config", $graphConfig,
    "--baseline-config", $baselineConfig,
    "--output", (Join-Path $repo "runs\fanoutqa\dev\paired-report.json")
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "compare.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "compare.stderr.log")
$compare.WaitForExit()
