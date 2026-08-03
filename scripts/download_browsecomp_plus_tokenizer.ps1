param(
    [string]$ProxyUrl = ""
)

$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent $PSScriptRoot
$revision = "c1899de289a04d12100db370d81485cdf75e47ca"

if ($ProxyUrl) {
    $env:HTTPS_PROXY = $ProxyUrl
    $env:HTTP_PROXY = $ProxyUrl
}

& "$repoDir\.venv\Scripts\hf.exe" download `
    Qwen/Qwen3-0.6B `
    --revision $revision `
    --include "tokenizer*" "vocab*" "merges.txt" "config.json" `
    --local-dir "$repoDir\data\browsecomp_plus\qwen3-tokenizer"
exit $LASTEXITCODE
