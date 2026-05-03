param(
    [string]$Run = "",
    [string]$BaseUrl = "http://localhost:2024"
)

# Fix UTF-8 encoding for Chinese characters
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

function Send-Chat {
    param([string]$Message, [string]$ThreadId = "")
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(('{"message":"' + $Message + '"}'))
    $headers = @{ "Content-Type" = "application/json; charset=utf-8" }
    if ($ThreadId -ne "") { $headers["X-Thread-Id"] = $ThreadId }
    Write-Host ""
    Write-Host "[Q] $Message" -ForegroundColor Cyan
    if ($ThreadId -ne "") { Write-Host "    thread: $ThreadId" -ForegroundColor DarkGray }
    try {
        $raw = Invoke-RestMethod -Method POST -Uri "$BaseUrl/chat/stream" `
            -Body $bytes -Headers $headers -ErrorAction Stop
        $answer = ""
        $tid = ""
        foreach ($line in ($raw -split "`n")) {
            $line = $line.Trim()
            if ($line -eq "" -or $line -eq "data: [DONE]") { continue }
            if ($line.StartsWith("data: ")) {
                $payload = $line.Substring(6)
                if ($payload.StartsWith("{")) {
                    try {
                        $obj = $payload | ConvertFrom-Json
                        if ($obj.thread_id) { $tid = $obj.thread_id }
                    } catch {}
                } else {
                    $answer += $payload
                }
            }
        }
        Write-Host "[OK] $answer" -ForegroundColor Green
        if ($tid -ne "") { Write-Host "     thread_id: $tid" -ForegroundColor DarkGray }
        return $tid
    } catch {
        Write-Host "[ERR] $_" -ForegroundColor Red
        return ""
    }
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 55) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host ("=" * 55) -ForegroundColor DarkCyan
}

# ----------------------------------------------------------
# Test-Basic: verify checkpoint works
# Round 1: calculate 3 + 5  -> expect 8
# Round 2: multiply result by 10  -> expect 80
# ----------------------------------------------------------
function Test-Basic {
    Write-Section "[basic] Checkpoint test: R1=8  R2=80"
    $tid = Send-Chat "calculate 3 plus 5"
    if ($tid -ne "") {
        Send-Chat "multiply the previous result by 10" $tid | Out-Null
    }
}

# ----------------------------------------------------------
# Test-Multi: 5-round chain
# 10+5=15 -> *2=30 -> -8=22 -> /2=11 -> +100=111
# ----------------------------------------------------------
function Test-Multi {
    Write-Section "[multi] 5-round chain: 15->30->22->11->111"
    $tid = Send-Chat "calculate 10 plus 5"
    if ($tid -eq "") { return }
    $tid = Send-Chat "multiply the previous result by 2" $tid
    $tid = Send-Chat "subtract 8 from the previous result" $tid
    $tid = Send-Chat "divide the previous result by 2" $tid
    Send-Chat "add 100 to the previous result" $tid | Out-Null
}

# ----------------------------------------------------------
# Test-Memory: tell name, recall later
# ----------------------------------------------------------
function Test-Memory {
    Write-Section "[memory] name recall"
    $tid = Send-Chat "Hi, my name is Tony and I am a backend engineer living in Toronto"
    if ($tid -ne "") {
        Send-Chat "What is my name and what do I do?" $tid | Out-Null
    }
}

# ----------------------------------------------------------
# Test-DB: database queries
# ----------------------------------------------------------
function Test-DB {
    Write-Section "[db] database queries"
    Send-Chat "How many users are in the database?" | Out-Null
    Send-Chat "What is the most expensive product and its price?" | Out-Null
}

# ----------------------------------------------------------
# Interactive mode
# ----------------------------------------------------------
function Start-Interactive {
    Write-Host "MCP Chat interactive - type 'new' for new session, 'quit' to exit" -ForegroundColor Cyan
    $currentTid = ""
    while ($true) {
        $inp = (Read-Host ">").Trim()
        if ($inp -eq "") { continue }
        if ($inp -in @("quit", "exit", "q")) { Write-Host "Bye!" -ForegroundColor Yellow; break }
        if ($inp -eq "new") {
            $currentTid = ""
            Write-Host "New session started" -ForegroundColor Yellow
            continue
        }
        $newTid = Send-Chat $inp $currentTid
        if ($currentTid -eq "" -and $newTid -ne "") {
            $currentTid = $newTid
            Write-Host "    Bound to thread_id: $currentTid" -ForegroundColor DarkGray
        }
    }
}

switch ($Run.ToLower()) {
    "basic"  { Test-Basic }
    "multi"  { Test-Multi }
    "memory" { Test-Memory }
    "db"     { Test-DB }
    "all"    { Test-Basic; Test-Multi; Test-Memory; Test-DB }
    default  { Start-Interactive }
}