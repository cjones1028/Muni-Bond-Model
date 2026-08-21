# Pull_ICE_Evals.ps1 -- pulls a local snapshot of the ICE evaluation data.
# PowerShell port of get_universe() + pull_ice_data() from ICE_Trading.ipynb,
# for machines without Python. End result: ICE_Evals.csv next to this script,
# one row per bond with the full ICE reference + evaluation fields (bid/mid/
# offer, yields, and all the descriptive columns the models use as features).
#
# Supply the API key (AUTH_VALUE from ICE_Trading.ipynb) via env var:
#     $env:ICE_ACCESS_KEY = '<key>'
#
# Usage:
#     .\Pull_ICE_Evals.ps1 -MaxSymbols 8000       # quick test slice (2 batches)
#     .\Pull_ICE_Evals.ps1                        # FULL universe (long run)
#     .\Pull_ICE_Evals.ps1 -InputCsv my_cusips.csv  # just the CUSIPs you list
#
# Design mirrors the notebook: a failed universe shard raises; a failed data
# batch is split in half and retried, isolating bad CUSIPs instead of losing
# the whole batch.

param(
    [string]$AuthValue = $env:ICE_ACCESS_KEY,
    [string]$InputCsv = '',          # optional: CSV whose first column is CUSIPs; skips the universe fetch
    [int]$MaxSymbols = 0,            # optional: cap for a test run (0 = no cap)
    [int]$ChunkSize = 4000,
    [int]$NShards = 12,
    [string]$OutFile = ''
)

if (-not $AuthValue) {
    Write-Host "No API key. Set `$env:ICE_ACCESS_KEY first (AUTH_VALUE in ICE_Trading.ipynb)."
    exit 1
}
if (-not $OutFile) { $OutFile = Join-Path $PSScriptRoot 'ICE_Evals.csv' }

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$endpoint = 'https://8eec6nhnfj.execute-api.us-east-1.amazonaws.com/prod'

function Invoke-IceApi($payloadHash) {
    # requests.post(..., json=json.dumps(payload)) double-encodes: the HTTP
    # body is a JSON *string* containing JSON. ConvertTo-Json on the compact
    # string reproduces that exactly.
    $inner = $payloadHash | ConvertTo-Json -Compress
    $body = $inner | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
    if ($resp.statusCode -ne 200) { throw "API returned statusCode $($resp.statusCode)" }
    return $resp.body | ConvertFrom-Json
}

# ---------- step 1: the symbol list ----------

$symbols = New-Object System.Collections.Generic.List[string]

if ($InputCsv) {
    foreach ($line in Get-Content $InputCsv) {
        $line = $line.Trim()
        if ($line) { $symbols.Add(($line -split ',')[0]) }
    }
    Write-Host ("loaded {0:N0} symbols from {1}" -f $symbols.Count, $InputCsv)
} else {
    Write-Host "fetching universe ($NShards shards)..."
    foreach ($shard in 0..($NShards - 1)) {
        $result = $null
        foreach ($attempt in 1..2) {
            try { $result = Invoke-IceApi @{ ACCESS_KEY = $AuthValue; FETCH_UNIVERSE = $shard }; break }
            catch {
                Write-Host "  shard $shard attempt $attempt failed: $($_.Exception.Message)"
                if ($attempt -eq 2) { throw "Failed universe shard $shard -- stopping (incomplete universe is worse than none)." }
                Start-Sleep -Seconds 3
            }
        }
        if (-not $result.universe) { throw "Shard $shard response had no 'universe' key" }
        foreach ($line in ($result.universe -split "`r?`n")) {
            $line = $line.Trim()
            if ($line) { $symbols.Add(($line -split ',')[0]) }
        }
        Write-Host ("  shard {0,2} done, running total {1:N0}" -f $shard, $symbols.Count)
    }
}

if ($MaxSymbols -gt 0 -and $symbols.Count -gt $MaxSymbols) {
    Write-Host ("capping at first {0:N0} of {1:N0} symbols (test mode)" -f $MaxSymbols, $symbols.Count)
    $symbols = $symbols.GetRange(0, $MaxSymbols)
}

# ---------- step 2: pull eval data in batches ----------

function Get-BatchWithIsolation([System.Collections.Generic.List[string]]$bucket) {
    # returns a list of CSV strings; on failure splits the bucket in half and
    # retries each half, narrowing down to exactly which CUSIP(s) are bad
    $csvOut = New-Object System.Collections.Generic.List[string]
    try {
        $result = Invoke-IceApi @{ ACCESS_KEY = $AuthValue; IDENTIFIER_TYPE = 'CUSIP'; symbols = @($bucket) }
        if ($null -ne $result.symbols) {
            $csvOut.Add($result.symbols)
            return ,$csvOut
        }
    } catch {
        # fall through to split
    }
    if ($bucket.Count -eq 1) {
        Write-Host "  skipping symbol -- no data returned: $($bucket[0])"
        return ,$csvOut
    }
    $mid = [int][Math]::Floor($bucket.Count / 2)
    $left  = $bucket.GetRange(0, $mid)
    $right = $bucket.GetRange($mid, $bucket.Count - $mid)
    $csvOut.AddRange((Get-BatchWithIsolation $left))
    $csvOut.AddRange((Get-BatchWithIsolation $right))
    return ,$csvOut
}

$nBatches = [Math]::Ceiling($symbols.Count / $ChunkSize)
Write-Host ("pulling eval data: {0:N0} symbols in {1} batch(es) of up to {2:N0}" -f $symbols.Count, $nBatches, $ChunkSize)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$header = $null
$rows = 0
$writer = New-Object System.IO.StreamWriter($OutFile, $false, [System.Text.Encoding]::UTF8)
try {
    for ($b = 0; $b -lt $nBatches; $b++) {
        $start = $b * $ChunkSize
        $len = [Math]::Min($ChunkSize, $symbols.Count - $start)
        $bucket = $symbols.GetRange($start, $len)
        $csvStrings = Get-BatchWithIsolation $bucket
        foreach ($csv in $csvStrings) {
            $lines = $csv -split "`r?`n"
            if ($lines.Count -lt 2) { continue }
            if ($null -eq $header) {
                $header = $lines[0]
                $writer.WriteLine($header)
            } elseif ($lines[0] -ne $header) {
                Write-Host "  WARNING: batch header differs from first batch -- appending anyway, check columns."
            }
            foreach ($line in ($lines | Select-Object -Skip 1)) {
                if ($line.Trim()) { $writer.WriteLine($line); $rows++ }
            }
        }
        Write-Host ("batch {0,4}/{1}: {2,9:N0} rows total  ({3:hh\:mm\:ss} elapsed)" -f ($b + 1), $nBatches, $rows, $sw.Elapsed)
    }
} finally {
    $writer.Close()
}

Write-Host ""
Write-Host ("DONE: {0:N0} rows -> {1}" -f $rows, $OutFile)

# archive a dated snapshot -- accumulating daily/weekly pulls is what enables
# time-based validation and time-aware features later. Do not delete these.
$archDir = Join-Path $PSScriptRoot 'evals_archive'
if (-not (Test-Path $archDir)) { New-Item -ItemType Directory -Path $archDir | Out-Null }
$stamp = Get-Date -Format 'yyyy-MM-dd'
Copy-Item $OutFile (Join-Path $archDir "ICE_Evals_$stamp.csv") -Force
Write-Host ("snapshot archived -> evals_archive\ICE_Evals_$stamp.csv")
