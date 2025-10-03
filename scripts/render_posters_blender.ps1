[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]  [string]$Database,   # was: -Db (conflicted with -Debug alias)
  [Parameter(Mandatory=$true)]  [string]$OutDir,
  [Parameter(Mandatory=$true)]  [string]$BlenderPath,
  [int]$Size = 640,
  [int]$Limit
)

$ErrorActionPreference = "Stop"

function Resolve-PathOrDie([string]$p, [string]$what) {
  if (-not (Test-Path -LiteralPath $p)) { throw "$what not found: $p" }
  return (Resolve-Path -LiteralPath $p).Path
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..")

$DbPath     = Resolve-PathOrDie $Database "Database"
$BlenderExe = Resolve-PathOrDie $BlenderPath "Blender"
$VenvPython = Resolve-PathOrDie (Join-Path $RepoRoot ".venv\Scripts\python.exe") "Virtualenv Python"
$RenderPy   = Resolve-PathOrDie (Join-Path $RepoRoot "scripts\render_posters_blender.py") "Renderer script"

$OutAbs = Join-Path $RepoRoot $OutDir
if (-not (Test-Path -LiteralPath $OutAbs)) { New-Item -ItemType Directory -Path $OutAbs | Out-Null }
$OutAbs = (Resolve-Path -LiteralPath $OutAbs).Path

Write-Host "[INFO] Using:" -ForegroundColor Cyan
Write-Host "  DB           : $DbPath"
Write-Host "  OutDir       : $OutAbs"
Write-Host "  Blender      : $BlenderExe"
Write-Host "  Python       : $VenvPython"
Write-Host "  Renderer Py  : $RenderPy"
Write-Host "  Size         : $Size"
if ($PSBoundParameters.ContainsKey('Limit')) { Write-Host "  Limit        : $Limit" }

$argsList = @(
  $RenderPy,
  "--db", $DbPath,
  "--out", $OutAbs,
  "--blender", $BlenderExe,
  "--size", $Size
)
if ($PSBoundParameters.ContainsKey('Limit')) { $argsList += @("--limit", $Limit) }

Write-Host "[RUN] $VenvPython $($argsList -join ' ')" -ForegroundColor Yellow

$process = Start-Process -FilePath $VenvPython -ArgumentList $argsList -NoNewWindow -PassThru -Wait
if ($process.ExitCode -ne 0) { throw "Poster render failed with exit code $($process.ExitCode)" }

Write-Host "[DONE] Posters rendered to $OutAbs" -ForegroundColor Green
