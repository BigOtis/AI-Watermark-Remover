# PowerShell setup using Python venv
try {
  $py = python --version
  Write-Host "Python detected: $py"
}
catch {
  Write-Host "Python is not installed or not in PATH." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path ".venv")) {
  Write-Host "Creating virtual environment (.venv)..."
  python -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Downloading LaMa model..."
iopaint download --model lama

if ($args.Count -gt 0) {
  python remwm.py @args
} else {
  python remwmgui.py
}