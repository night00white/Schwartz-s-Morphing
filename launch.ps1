$ErrorActionPreference = "SilentlyContinue"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "     Starting Identity Shift App (Safe Mode)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Kill old processes
taskkill /F /IM python.exe | Out-Null
taskkill /F /IM blender.exe | Out-Null
taskkill /F /IM houdini.exe | Out-Null

# 2. Find Python
$pyPaths = @(
    "C:\Program Files\Blender Foundation\Blender 4.5\4.5\python\bin\python.exe",
    "C:\Program Files\Side Effects Software\Houdini 21.0.440\python311\python.exe",
    "python"
)

$pythonExe = $null
foreach ($path in $pyPaths) {
    if (Test-Path $path) {
        $pythonExe = $path
        break
    }
}

if (-not $pythonExe) {
    Write-Error "Python not found."
    Pause
    exit 1
}

Write-Host "[INFO] Using Python: $pythonExe" -ForegroundColor Green

# 3. Launch Bootstrapper
Write-Host "[INFO] Launching Bootstrapper..." -ForegroundColor Green

# Open browser first
Start-Process "http://127.0.0.1:8080"

# Run the server script
& $pythonExe start_server.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Server crashed."
    Pause
}
