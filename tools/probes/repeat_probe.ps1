# Run probe_slip.py N times for one condition and collect the summary lines.
param(
    [string]$Name,
    [string]$SceneArgs,
    [int]$Repeats = 5,
    [int]$Steps = 300
)
$repo = "D:\IslandMaster\jumper"                      # your jumper checkout
$py = "$repo\.venv\Scripts\python.exe"
$probe = Join-Path $PSScriptRoot "probe_slip.py"       # the probe next to this script
$log = "$repo\out\recordings\repeat_$Name.log"
Set-Location $repo
"# $Name : $SceneArgs : $Repeats x $Steps steps" | Out-File -Encoding utf8 $log
for ($i = 1; $i -le $Repeats; $i++) {
    $csv = "$repo\out\recordings\repeat_${Name}_$i.csv"
    $argv = @("-I", $probe, "--out", $csv, "--", "--task", "jumper.jump", "--checkpoint", "tasks/jumper/jump/out/example/model_3100.pt") + ($SceneArgs -split " ") + @("--backend", "native", "--device", "cpu", "--headless", "--steps", "$Steps", "--speed", "0")
    $out = & $py @argv 2>&1 | Out-String
    $lines = ($out -split "`n") | Where-Object { $_ -match "\[probe\] (steps|episode ends|rear feet|ground force|  landings)" }
    "## run $i" | Out-File -Encoding utf8 -Append $log
    $lines | Out-File -Encoding utf8 -Append $log
    if ($out -match "Traceback") { "TRACEBACK in run $i" | Out-File -Encoding utf8 -Append $log }
}
Get-Content $log
