param(
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repoDir ".venv\Scripts\graphptc.exe"
$graphConfig = "configs/frames/graphptc-test.toml"
$baselineConfig = "configs/frames/fewshot-ptc-test.toml"
$runDir = Join-Path $repoDir "runs\frames\test"
New-Item -ItemType Directory -Force $runDir | Out-Null

$restartArgument = if ($Restart) { @("--restart") } else { @() }
$graphRun = Start-Process -FilePath $runner `
    -ArgumentList (@("run-frames", "--config", $graphConfig) + $restartArgument) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "graphptc-run.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "graphptc-run.stderr.log") `
    -WindowStyle Hidden -PassThru
$baselineRun = Start-Process -FilePath $runner `
    -ArgumentList (@("run-frames", "--config", $baselineConfig) + $restartArgument) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "fewshot-ptc-run.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "fewshot-ptc-run.stderr.log") `
    -WindowStyle Hidden -PassThru
Wait-Process -InputObject @($graphRun, $baselineRun)

$graphEval = Start-Process -FilePath $runner `
    -ArgumentList @("evaluate-frames", "--config", $graphConfig) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "graphptc-eval.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "graphptc-eval.stderr.log") `
    -WindowStyle Hidden -PassThru
$baselineEval = Start-Process -FilePath $runner `
    -ArgumentList @("evaluate-frames", "--config", $baselineConfig) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "fewshot-ptc-eval.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "fewshot-ptc-eval.stderr.log") `
    -WindowStyle Hidden -PassThru
Wait-Process -InputObject @($graphEval, $baselineEval)
if ($graphEval.ExitCode -ne 0 -or $baselineEval.ExitCode -ne 0) {
    throw "FRAMES evaluation did not produce two complete reports"
}

$compare = Start-Process -FilePath $runner `
    -ArgumentList @(
        "compare-frames",
        "--config", $graphConfig,
        "--baseline-config", $baselineConfig,
        "--output", "runs/frames/test/paired-report.json"
    ) `
    -WorkingDirectory $repoDir `
    -RedirectStandardOutput (Join-Path $runDir "compare.stdout.log") `
    -RedirectStandardError (Join-Path $runDir "compare.stderr.log") `
    -WindowStyle Hidden -PassThru -Wait
if ($compare.ExitCode -ne 0) {
    throw "FRAMES paired comparison failed"
}
Get-Content (Join-Path $runDir "paired-report.json") -Raw
