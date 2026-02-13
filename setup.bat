@echo off
setlocal

cd /d %~dp0

echo [1/4] Creating virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 goto :error

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 goto :error

echo [3/4] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] Downloading AI model weights...
python scripts\download_model.py
if errorlevel 1 goto :error

echo.
echo Setup complete. Use start.bat to launch the app.
exit /b 0

:error
echo.
echo Setup failed. Please check the error output above.
exit /b 1
