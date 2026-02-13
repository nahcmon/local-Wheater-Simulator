@echo off
setlocal

cd /d %~dp0

if not exist .venv\Scripts\activate (
  echo Virtual environment not found. Run setup.bat first.
  exit /b 1
)

call .venv\Scripts\activate
if errorlevel 1 exit /b 1

streamlit run app.py
