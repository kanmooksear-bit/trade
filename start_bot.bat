@echo off
REM Double-click to start the bot (uses config.yaml in this folder). Close the window or press Ctrl+C to stop.
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist python_path.txt set /p PY=<python_path.txt
"%PY%" -m autotrader -c config.yaml daemon
pause
