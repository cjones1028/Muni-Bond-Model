@echo off
rem Double-click to open the muni pricer in JupyterLab on THIS computer.
rem A black window stays open while JupyterLab runs -- leave it open; close it
rem (or press Ctrl+C in it) when you're done. Your browser opens automatically.
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
echo Starting JupyterLab... your browser will open in a few seconds.
echo Leave this window open while you work. Close it to stop.
"%PY%" -m jupyterlab "Pricer.ipynb"
pause
