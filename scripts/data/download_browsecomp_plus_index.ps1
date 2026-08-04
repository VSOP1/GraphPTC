param(
    [string]$ProxyUrl = ""
)

$ErrorActionPreference = "Stop"
$repoDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$revision = "b3f37f70c33829eb09d04784a54277a31871fd63"

if ($ProxyUrl) {
    $env:HTTPS_PROXY = $ProxyUrl
    $env:HTTP_PROXY = $ProxyUrl
}

& "$repoDir\.venv\Scripts\hf.exe" download `
    Tevatron/browsecomp-plus-indexes `
    --repo-type dataset `
    --revision $revision `
    --include "bm25/*" `
    --local-dir "$repoDir\data\browsecomp_plus\official_indexes"
exit $LASTEXITCODE
