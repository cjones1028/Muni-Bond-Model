@echo off
rem Drag a saved wire .txt onto this file to price it.
rem Auto-detects the issuer from the wire text; the concession is
rem auto-calibrated from the deal archive (no numbers to maintain here).
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if "%~1"=="" (
    echo Drag a wire .txt file onto this .bat to price it.
    pause
    exit /b 1
)
"%PY%" run_pipeline.py --wire "%~1"
echo.
pause
