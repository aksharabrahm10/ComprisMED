@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the private comprisMED Python environment...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
    py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo Python 3 could not be found. Install Python 3.9 or newer, then try again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import gradio, dotenv, requests, sklearn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing comprisMED requirements. This can take a few minutes the first time...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting comprisMED in your browser. Keep this window open while you use the app.
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo comprisMED stopped with an error. Read the message above.
    pause
)
