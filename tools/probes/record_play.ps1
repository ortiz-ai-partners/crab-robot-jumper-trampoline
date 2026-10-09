# One-off helper: open the live viewer for a play.py run, screen-record its window
# for N seconds with the ffmpeg bundled by imageio-ffmpeg (desktop capture of the
# window's rectangle, so the OpenGL content is captured), then close the run.
param(
    [string]$Name = "trampoline_default",
    [string]$SceneArgs = "--scene trampoline",
    [int]$Seconds = 18,
    [int]$WarmupSeconds = 3,
    [int]$ZoomNotches = 0,
    [string]$Script = "",
    [double]$CamDistance = 0,
    [double]$CamElevation = -18,
    [double]$CamAzimuth = 135,
    [string]$Task = "jumper.jump",
    [string]$Checkpoint = "tasks/jumper/jump/out/example/model_3100.pt"
)
$ErrorActionPreference = "Stop"
$repo = "D:\IslandMaster\jumper"
$py = "$repo\.venv\Scripts\python.exe"
$ff = "$repo\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
$outDir = "$repo\out\recordings"
New-Item -ItemType Directory -Force $outDir | Out-Null
$mp4 = "$outDir\$Name.mp4"
$log = "$outDir\$Name.play.log"
$err = "$outDir\$Name.play.err.log"

Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class W {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
  public static void TopMost(IntPtr h) { SetWindowPos(h, new IntPtr(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040); }
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, int data, UIntPtr extra);
  public const uint MOUSEEVENTF_WHEEL = 0x0800;
  public static void Wheel(int notches) { mouse_event(MOUSEEVENTF_WHEEL, 0, 0, notches * 120, UIntPtr.Zero); }
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L; public int T; public int R; public int B; }
  public static string Title(IntPtr h) { var sb = new StringBuilder(256); GetWindowText(h, sb, 256); return sb.ToString(); }
  public static List<IntPtr> Find(uint pid) {
    var found = new List<IntPtr>();
    EnumWindows(delegate(IntPtr h, IntPtr l) {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p == pid && IsWindowVisible(h) && Title(h).Length > 0) { found.Add(h); }
      return true;
    }, IntPtr.Zero);
    return found;
  }
  public static List<IntPtr> FindTitle(string needle) {
    var found = new List<IntPtr>();
    EnumWindows(delegate(IntPtr h, IntPtr l) {
      if (IsWindowVisible(h) && Title(h).IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0) { found.Add(h); }
      return true;
    }, IntPtr.Zero);
    return found;
  }
  public static uint Pid(IntPtr h) { uint p; GetWindowThreadProcessId(h, out p); return p; }
}
"@
[W]::SetProcessDPIAware() | Out-Null

if ($Script -eq "") { $Script = "scripts/play.py" }
if ($CamDistance -gt 0) { $env:REC_CAM_DISTANCE = "$CamDistance"; $env:REC_CAM_ELEVATION = "$CamElevation"; $env:REC_CAM_AZIMUTH = "$CamAzimuth" }
$argList = "`"$Script`" --task $Task --checkpoint $Checkpoint $SceneArgs --backend native --device cpu"
Write-Output "[rec] starting: python $argList"
$p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $repo -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err

# wait for the viewer window: any visible, titled top-level window of the process
# (MainWindowHandle misses it -- the GLFW window is not the process's "main" one)
$deadline = (Get-Date).AddSeconds(180)
$h = [IntPtr]::Zero
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $p.Refresh()
    if ($p.HasExited) { break }
    # The venv's python.exe is a launcher: the interpreter (and its GLFW window)
    # is a child process, so look by title first, then by launcher + children.
    $hs = [W]::FindTitle("MuJoCo")
    if ($hs.Count -eq 0) {
        $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($p.Id)" | ForEach-Object { [uint32]$_.ProcessId })
        foreach ($pid2 in (@([uint32]$p.Id) + $kids)) { $hs += [W]::Find($pid2) }
    }
    if ($hs.Count -gt 0) { $h = $hs[0]; break }
}
if ($p.HasExited) {
    Write-Output "[rec] play.py exited early (code $($p.ExitCode)); stderr tail:"
    Get-Content $err -Tail 30
    exit 1
}
if ($h -eq [IntPtr]::Zero) {
    Write-Output "[rec] no window after 180 s; stopping"
    Stop-Process -Id $p.Id -Force
    exit 2
}
Write-Output "[rec] window '$([W]::Title($h))' handle $h"
[W]::SetForegroundWindow($h) | Out-Null
[W]::TopMost($h)   # keep the viewer above other windows for the capture
Start-Sleep -Seconds $WarmupSeconds
$r = New-Object W+RECT
[W]::GetWindowRect($h, [ref]$r) | Out-Null
$x = $r.L; $y = $r.T; $w = $r.R - $r.L; $hh = $r.B - $r.T
if ($ZoomNotches -ne 0) {
    # zoom the tracking camera in: wheel notches over the render area
    [W]::SetCursorPos([int](($r.L + $r.R) / 2), [int](($r.T + $r.B) / 2)) | Out-Null
    Start-Sleep -Milliseconds 300
    for ($i = 0; $i -lt [Math]::Abs($ZoomNotches); $i++) {
        [W]::Wheel([Math]::Sign($ZoomNotches)); Start-Sleep -Milliseconds 40
    }
    Write-Output "[rec] zoomed $ZoomNotches notches"
    Start-Sleep -Seconds 1
}
if ($x -lt 0) { $x = 0 }; if ($y -lt 0) { $y = 0 }
$w = $w - ($w % 2); $hh = $hh - ($hh % 2)
Write-Output "[rec] rect x=$x y=$y w=$w h=$hh -> $mp4 ($Seconds s)"

& $ff -hide_banner -loglevel error -y -f gdigrab -framerate 30 -draw_mouse 0 `
    -offset_x $x -offset_y $y -video_size "${w}x${hh}" -i desktop `
    -t $Seconds -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p $mp4
$ffExit = $LASTEXITCODE
Write-Output "[rec] ffmpeg exit $ffExit"

# stop the launcher and the interpreter it spawned (only processes running play.py)
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts[/\\]play\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1
Write-Output "[rec] play stdout tail:"
(Get-Content $log -Raw) -split "`r" | Where-Object { $_ -match "\S" } | Select-Object -Last 6
if (Test-Path $mp4) { Write-Output ("[rec] mp4 {0:N1} MB" -f ((Get-Item $mp4).Length / 1MB)) }
