@echo off
setlocal

cd /d %~dp0

if not exist .venv\Scripts\activate (
  echo Virtual environment not found. Run setup.bat first.
  exit /b 1
)

call .venv\Scripts\activate
if errorlevel 1 exit /b 1

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
streamlit run app.py --server.headless false
