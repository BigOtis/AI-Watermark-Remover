@echo off
echo ==================================
echo  Installing Sora2WatermarkRemover
echo ==================================
echo.

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Install Python from https://www.python.org/downloads/ and retry.
    pause
    exit /b 1
)

echo Creating virtual environment (.venv) if needed...
if not exist .venv (
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error creating virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error while installing dependencies.
    pause
    exit /b 1
)

echo Downloading LaMa model...
iopaint download --model lama
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Error downloading the LaMa model.
    echo You can retry later with: iopaint download --model lama
)

echo.
echo ===============================
echo  Installation complete!
echo ===============================
echo.
echo To launch the application:
echo 1. Open a command prompt
echo 2. Activate venv: .\.venv\Scripts\activate
echo 3. Start the GUI: python remwmgui.py
echo.

choice /C YN /M "Launch the application now? (Y/N)"
if %ERRORLEVEL% EQU 1 (
    echo Launching application...
    python remwmgui.py
)

echo.
pause 