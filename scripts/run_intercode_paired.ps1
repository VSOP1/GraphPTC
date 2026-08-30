$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repo ".venv\Scripts\graphptc.exe"
$graphConfig = Join-Path $repo "configs\intercode\graphptc.toml"
$baselineConfig = Join-Path $repo "configs\intercode\baseline.toml"
$logDir = Join-Path $repo "runs\intercode\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Start-LoggedJob {
    param([pscustomobject]$Spec)
    Start-Job -ScriptBlock {
        param($JobSpec)
        Set-Location $JobSpec.WorkingDirectory
        & $JobSpec.Runner @($JobSpec.Arguments) `
            1> $JobSpec.StandardOutput 2> $JobSpec.StandardError
        if ($LASTEXITCODE -ne 0) {
            throw "Command exited with code $LASTEXITCODE"
        }
    } -ArgumentList $Spec
}

function Wait-LoggedJobs {
    param([object[]]$Jobs, [string]$Phase)
    Wait-Job -Job $Jobs | Out-Null
    $failed = @($Jobs | Where-Object State -ne "Completed")
    Receive-Job -Job $Jobs | Out-Null
    Remove-Job -Job $Jobs
    if ($failed.Count -ne 0) {
        throw "InterCode $Phase failed"
    }
}

$runJobs = @(
    Start-LoggedJob ([pscustomobject]@{
        Runner = $runner
        Arguments = @("run-intercode", "--config", $graphConfig, "--restart")
        WorkingDirectory = $repo
        StandardOutput = Join-Path $logDir "graphptc-run.stdout.log"
        StandardError = Join-Path $logDir "graphptc-run.stderr.log"
    })
    Start-LoggedJob ([pscustomobject]@{
        Runner = $runner
        Arguments = @("run-intercode", "--config", $baselineConfig, "--restart")
        WorkingDirectory = $repo
        StandardOutput = Join-Path $logDir "baseline-run.stdout.log"
        StandardError = Join-Path $logDir "baseline-run.stderr.log"
    })
)
Wait-LoggedJobs $runJobs "generation"

$evaluateJobs = @(
    Start-LoggedJob ([pscustomobject]@{
        Runner = $runner
        Arguments = @("evaluate-intercode", "--config", $graphConfig)
        WorkingDirectory = $repo
        StandardOutput = Join-Path $logDir "graphptc-evaluate.stdout.log"
        StandardError = Join-Path $logDir "graphptc-evaluate.stderr.log"
    })
    Start-LoggedJob ([pscustomobject]@{
        Runner = $runner
        Arguments = @("evaluate-intercode", "--config", $baselineConfig)
        WorkingDirectory = $repo
        StandardOutput = Join-Path $logDir "baseline-evaluate.stdout.log"
        StandardError = Join-Path $logDir "baseline-evaluate.stderr.log"
    })
)
Wait-LoggedJobs $evaluateJobs "aggregation"

& $runner @(
    "compare-intercode",
    "--config", $graphConfig,
    "--baseline-config", $baselineConfig,
    "--output", (Join-Path $repo "runs\intercode\paired-report.json")
) 1> (Join-Path $logDir "compare.stdout.log") `
   2> (Join-Path $logDir "compare.stderr.log")
if ($LASTEXITCODE -ne 0) {
    throw "InterCode paired comparison failed: $LASTEXITCODE"
}
