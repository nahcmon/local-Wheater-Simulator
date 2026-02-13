@echo off
setlocal

cd /d %~dp0

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11+ from python.org first.
  exit /b 1
)

echo [1/5] Creating virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 goto :error

echo [2/5] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 goto :error

echo [3/5] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo [4/5] Installing PyTorch CUDA (cu121) for Windows...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :error

echo [5/5] Installing app dependencies and model...
pip install -r requirements.txt
if errorlevel 1 goto :error
python scripts\download_model.py
if errorlevel 1 goto :error

echo.
echo Setup complete. Run start.bat to launch the app.
exit /b 0

:error
echo.
echo Setup failed. Please review the error output above.
exit /b 1
