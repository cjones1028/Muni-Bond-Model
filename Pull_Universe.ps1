# Pull_Universe.ps1 -- PowerShell port of get_universe() from ICE_Trading.ipynb.
# Fetches the full ICE universe in 12 shards from the AWS API endpoint and
# writes the identifier list to ICE_Universe.csv next to this script.
#
# The API key is NOT hardcoded here (deliberately -- same reason the notebook
# docstring flags it). Supply it one of two ways:
#     $env:ICE_ACCESS_KEY = '<key from ICE_Trading.ipynb>'
#     .\Pull_Universe.ps1
# or:
#     .\Pull_Universe.ps1 -AuthValue '<key>'
#
# Mirrors the notebook's design: a failed shard RAISES rather than being
# skipped -- an incomplete universe with no warning is the worse failure mode.
# Each shard gets one retry before giving up.

param(
    [string]$AuthValue = $env:ICE_ACCESS_KEY,
    [int]$NShards = 12
)

if (-not $AuthValue) {
    Write-Host "No API key. Set `$env:ICE_ACCESS_KEY or pass -AuthValue '<key>'."
    Write-Host "The key is the AUTH_VALUE constant at the top of ICE_Trading.ipynb."
    exit 1
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$endpoint = 'https://8eec6nhnfj.execute-api.us-east-1.amazonaws.com/prod'
$outFile = Join-Path $PSScriptRoot 'ICE_Universe.csv'

function Get-Shard($shard) {
    $payload = @{ ACCESS_KEY = $AuthValue; FETCH_UNIVERSE = $shard } | ConvertTo-Json -Compress
    # requests.post(..., json=json.dumps(payload)) double-encodes: the HTTP
    # body is a JSON *string* containing JSON. ConvertTo-Json on the string
    # reproduces that exactly.
    $body = $payload | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
    if ($resp.statusCode -ne 200) {
        throw "Shard $shard returned statusCode $($resp.statusCode)"
    }
    $result = $resp.body | ConvertFrom-Json
    if (-not $result.universe) {
        throw "Shard $shard response had no 'universe' key"
    }
    return $result.universe
}

$all = New-Object System.Collections.Generic.List[string]
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($shard in 0..($NShards - 1)) {
    $csv = $null
    foreach ($attempt in 1..2) {
        try {
            $csv = Get-Shard $shard
            break
        } catch {
            Write-Host "Shard $shard attempt $attempt failed: $($_.Exception.Message)"
            if ($attempt -eq 2) { throw "Failed to retrieve universe shard $shard. Try again, or check the API/network." }
            Start-Sleep -Seconds 3
        }
    }
    $count = 0
    foreach ($line in ($csv -split "`r?`n")) {
        $line = $line.Trim()
        if ($line) {
            # first column only, matching csv.reader(...)[0] in the notebook
            $all.Add(($line -split ',')[0])
            $count++
        }
    }
    Write-Host ("shard {0,2}: {1,8:N0} identifiers  (running total {2,9:N0}, {3:mm\:ss} elapsed)" -f $shard, $count, $all.Count, $sw.Elapsed)
}

$all | Set-Content -Encoding utf8 $outFile
Write-Host ""
Write-Host ("DONE: {0:N0} identifiers -> {1}" -f $all.Count, $outFile)
Write-Host "First 3:"; $all | Select-Object -First 3
