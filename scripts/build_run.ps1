# ModelFinder - PowerShell Build & Run Script
# Sets up virtual environment, installs dependencies, and builds executables

# Set execution policy for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Write-Host "ModelFinder - PowerShell Build & Run Script" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Yellow
} catch {
    Write-Host "Error: Python not found. Please install Python 3.8+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Activation failed"
    }
} catch {
    Write-Host ""
    Write-Host "⚠️  Virtual environment activation failed!" -ForegroundColor Red
    Write-Host "This is likely due to PowerShell execution policy restrictions." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To fix this, run the following command and try again:" -ForegroundColor Cyan
    Write-Host "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor White
    Write-Host ""
    Write-Host "Then re-run this script:" -ForegroundColor Cyan
    Write-Host ".\scripts\build_run.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Set virtual environment Python path
$venvPy = ".venv\Scripts\python.exe"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
& $venvPy -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install core dependencies." -ForegroundColor Red
    exit 1
}

# Install PyInstaller
Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
& $venvPy -m pip install pyinstaller

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "db" | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\glb" | Out-Null
New-Item -ItemType Directory -Force -Path "data\posters" | Out-Null
New-Item -ItemType Directory -Force -Path "data\metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "build" | Out-Null
New-Item -ItemType Directory -Force -Path "dist" | Out-Null

# Build executables
Write-Host "Building executables..." -ForegroundColor Yellow
& "scripts\build_exe.bat"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build executables." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test scanning: python scripts\scan_folder.py `"C:\Your3DModels`"" -ForegroundColor White
Write-Host "2. Test searching: python scripts\search_cli.py --like-id 1" -ForegroundColor White
Write-Host "3. Use executables: dist\ModelFinder_Indexer.exe scan `"C:\Your3DModels`"" -ForegroundColor White
Write-Host ""
Write-Host "Executables created in dist\:" -ForegroundColor Cyan
Write-Host "  - ModelFinder_Indexer.exe  (Main application)" -ForegroundColor White
Write-Host "  - ModelFinder_Scanner.exe  (Scan models)" -ForegroundColor White
Write-Host "  - ModelFinder_Searcher.exe (Search models)" -ForegroundColor White
Write-Host ""

# Ask if user wants to test scan
$testScan = Read-Host "Would you like to test scan a folder now? (y/n)"
if ($testScan -eq "y" -or $testScan -eq "Y") {
    $scanPath = Read-Host "Enter path to 3D models folder"
    if (Test-Path $scanPath) {
        Write-Host "Scanning folder: $scanPath" -ForegroundColor Yellow
        & $venvPy scripts\scan_folder.py $scanPath --verbose
    } else {
        Write-Host "Path not found: $scanPath" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Script completed!" -ForegroundColor Green