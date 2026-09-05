<#
.SYNOPSIS
    Host-side backstop: keep WSL2 from exhausting Windows' commit limit.

.DESCRIPTION
    On 2026-09-04 this desktop died four times in one evening with
    STATUS_COMMITMENT_LIMIT (0xc000012d): dwm.exe, Explorer, the Start
    menu and the NVIDIA display container all crashed while a model was
    loading inside WSL2. Nothing inside the guest could see it coming --
    /proc/meminfo describes the GUEST, and the guest looked healthy --
    because the number that matters is the Windows COMMIT charge of the
    vmmemWSL process: guest RAM actually touched, PLUS every CUDA
    allocation the WDDM driver silently backed with host memory once the
    32 GiB card was full. Resource-Exhaustion-Detector event 2004 logged
    vmmemWSL at 89.5 GB of a 100.7 GiB limit at 22:09; the desktop went at
    22:09:49.

    reactor_core/training/memory_guard.py now hard-kills its own process
    below a commit floor (14 GiB). This script is the layer BEHIND that,
    for the cases the in-process guard cannot cover: a run launched
    without the guard, a guest that has stopped scheduling the watchdog
    thread, or something that is not a training run at all.

    Two floors, checked every IntervalSec:

      FreeGiB < FloorGiB     -> SIGKILL the training processes in the
                                distro (KillPattern), once per 5 s.
      FreeGiB < ShutdownGiB  -> wsl.exe --shutdown. Also taken when the
                                kill did not recover the floor within
                                EscalateSec.

    Every action is logged with the numbers that caused it, so what it
    did is never a mystery afterwards.

    Registered at logon as the per-user task JARVIS\WSLCommitSentinel
    (no elevation needed); see the schtasks line at the bottom.

.NOTES
    The KillPattern default uses the [p]rocess bracket trick so the
    'bash -c pkill -f ...' wrapper that WSL spawns does not match its own
    command line.
#>
[CmdletBinding()]
param(
    [double]$FloorGiB     = 12.0,
    [double]$ShutdownGiB  = 8.0,
    [double]$IntervalSec  = 1.0,
    [double]$EscalateSec  = 10.0,
    [string]$Distro       = "Ubuntu",
    [string]$User         = "jarvis_svc",
    [string]$KillPattern  = "[p]rofile_grpo_vram|[r]un_grpo_training",
    [string]$LogPath      = (Join-Path $env:LOCALAPPDATA "JARVIS\wsl_commit_sentinel.log"),
    [double]$HeartbeatSec = 600.0,
    [switch]$DryRun,
    [int]$MaxTicks        = 0
)

$ErrorActionPreference = "Continue"
$logDir = Split-Path -Parent $LogPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    try { Add-Content -Path $LogPath -Value $line -Encoding utf8 } catch {}
    Write-Output $line
}

function Get-CommitGiB {
    $os = Get-CimInstance Win32_OperatingSystem
    [pscustomobject]@{
        FreeGiB  = [math]::Round($os.FreeVirtualMemory / 1MB, 2)
        LimitGiB = [math]::Round($os.TotalVirtualMemorySize / 1MB, 2)
    }
}

function Get-VmmemGiB {
    $p = Get-Process vmmemWSL -ErrorAction SilentlyContinue
    if ($p) { return [math]::Round(($p | Measure-Object PrivateMemorySize64 -Sum).Sum / 1GB, 2) }
    return 0.0
}

function Invoke-Kill {
    if ($DryRun) { Write-Log "  (dry-run) would run: wsl.exe -d $Distro -u $User -- pkill -9 -f '$KillPattern'"; return }
    try {
        $quoted = "'" + $KillPattern + "'"
        & wsl.exe -d $Distro -u $User -- pkill -9 -f $quoted 2>$null | Out-Null
        Write-Log "  pkill sent (rc=$LASTEXITCODE; 0=matched, 1=nothing matched)"
    } catch { Write-Log "  pkill FAILED: $_" }
}

function Invoke-Shutdown {
    if ($DryRun) { Write-Log "  (dry-run) would run: wsl.exe --shutdown"; return }
    try {
        & wsl.exe --shutdown 2>$null | Out-Null
        Write-Log "  wsl --shutdown issued (rc=$LASTEXITCODE)"
    } catch { Write-Log "  wsl --shutdown FAILED: $_" }
}

$mode = "armed"
if ($DryRun) { $mode = "DRY-RUN" }
$c0 = Get-CommitGiB
Write-Log ("sentinel start [{0}] floor={1} GiB shutdown={2} GiB interval={3}s escalate={4}s distro={5} user={6} | commit free={7} / limit={8} GiB, vmmemWSL={9} GiB, pid={10}" -f `
    $mode, $FloorGiB, $ShutdownGiB, $IntervalSec, $EscalateSec, $Distro, $User, `
    $c0.FreeGiB, $c0.LimitGiB, (Get-VmmemGiB), $PID)

$lastKill      = [datetime]::MinValue
$lowSince      = $null
$lastHeartbeat = Get-Date
$tick          = 0
$state         = "ok"

while ($true) {
    $tick++
    if ($MaxTicks -gt 0 -and $tick -gt $MaxTicks) { Write-Log "max ticks reached; exiting"; break }
    try {
        $c   = Get-CommitGiB
        $vm  = Get-VmmemGiB
        $now = Get-Date

        if ($c.FreeGiB -lt $ShutdownGiB) {
            Write-Log ("CRITICAL commit free={0} GiB < {1} GiB (vmmemWSL={2} GiB) -> wsl --shutdown" -f $c.FreeGiB, $ShutdownGiB, $vm)
            Invoke-Shutdown
            $state = "shutdown"; $lowSince = $null
            Start-Sleep -Seconds 10
            continue
        }

        if ($c.FreeGiB -lt $FloorGiB) {
            if ($null -eq $lowSince) { $lowSince = $now }
            if (($now - $lastKill).TotalSeconds -ge 5) {
                Write-Log ("LOW commit free={0} GiB < {1} GiB (vmmemWSL={2} GiB) -> kill '{3}' in {4}" -f $c.FreeGiB, $FloorGiB, $vm, $KillPattern, $Distro)
                Invoke-Kill
                $lastKill = $now
            }
            if (($now - $lowSince).TotalSeconds -ge $EscalateSec) {
                Write-Log ("ESCALATE still {0} GiB free {1:N0}s after first kill (vmmemWSL={2} GiB) -> wsl --shutdown" -f $c.FreeGiB, ($now - $lowSince).TotalSeconds, $vm)
                Invoke-Shutdown
                $lowSince = $null
                Start-Sleep -Seconds 10
            }
            $state = "low"
        }
        else {
            if ($state -ne "ok") { Write-Log ("recovered: commit free={0} GiB, vmmemWSL={1} GiB" -f $c.FreeGiB, $vm) }
            $state = "ok"; $lowSince = $null
        }

        if (($now - $lastHeartbeat).TotalSeconds -ge $HeartbeatSec) {
            Write-Log ("heartbeat commit free={0} / {1} GiB, vmmemWSL={2} GiB" -f $c.FreeGiB, $c.LimitGiB, $vm)
            $lastHeartbeat = $now
        }
    } catch {
        Write-Log "tick error: $_"
    }
    Start-Sleep -Milliseconds ([int]($IntervalSec * 1000))
}

# Registration (per-user, no elevation):
#   schtasks /Create /F /TN "JARVIS\WSLCommitSentinel" /SC ONLOGON /TR "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \"C:\Users\Jarvis\Desktop\TrinityAi\reactor\scripts\host\wsl_commit_sentinel.ps1\""
#   schtasks /Run /TN "JARVIS\WSLCommitSentinel"
