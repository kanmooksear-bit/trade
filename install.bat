@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === AutoTrader installer ===

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found - installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.
    echo Python installed. CLOSE this window and double-click install.bat again.
    pause
    exit /b
)

echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt MetaTrader5
if errorlevel 1 (
    echo Package install FAILED - see the messages above.
    pause
    exit /b 1
)

if not exist config.yaml (
    copy config.xauusd.example.yaml config.yaml >nul
    echo Created config.yaml for XAUUSD / HFM.
)

echo.
echo Checking connection to MT5 (MT5 must be open and logged in, Algo Trading ON)...
python -m autotrader -c config.yaml sizing
echo.
echo Done. Edit config.yaml with Notepad if needed, then double-click start_bot.bat
pause
