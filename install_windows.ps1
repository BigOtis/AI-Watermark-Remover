# PowerShell installer for Sora2WatermarkRemover (Python venv + pip)
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Installing Sora2WatermarkRemover" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyVersion = python --version
    Write-Host "Python detected: $pyVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/ and retry." -ForegroundColor Yellow
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}

# Create venv if not exists
$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error creating virtual environment." -ForegroundColor Red
        Read-Host -Prompt "Press Enter to exit"
        exit 1
    }
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$activateScript = Join-Path $venvPath "Scripts\\Activate.ps1"
. $activateScript

# Upgrade pip and install requirements
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error installing dependencies." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}

Write-Host "Downloading LaMa model..." -ForegroundColor Cyan
iopaint download --model lama
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Error downloading the LaMa model." -ForegroundColor Yellow
    Write-Host "You can retry later with: iopaint download --model lama" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""
Write-Host "To launch the application:" -ForegroundColor Cyan
Write-Host "1. Open PowerShell" -ForegroundColor Cyan
Write-Host "2. Activate venv: .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Cyan
Write-Host "3. Start the GUI: python remwmgui.py" -ForegroundColor Cyan
Write-Host ""

$launch = Read-Host "Launch the application now? (y/n)"
if ($launch -eq "y" -or $launch -eq "Y") {
    Write-Host "Launching application..." -ForegroundColor Green
    python remwmgui.py
}

Write-Host ""
Read-Host -Prompt "Press Enter to exit"