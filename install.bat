@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === AutoTrader installer ===

rem Find a REAL Python (the Microsoft Store "python" alias is only a stub).
call :findpython
if not defined PY (
    echo Python not found - installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    call :findpython
)
if not defined PY (
    echo.
    echo Could not install Python automatically.
    echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
    echo then double-click install.bat again.
    pause
    exit /b 1
)
echo Using Python: %PY%

echo Installing packages...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt MetaTrader5
if errorlevel 1 (
    echo Package install FAILED - see the messages above.
    pause
    exit /b 1
)

rem remember which Python to use for start_bot.bat
> python_path.txt echo %PY%

if not exist config.yaml (
    copy config.xauusd.example.yaml config.yaml >nul
    echo Created config.yaml for XAUUSD / HFM.
)

echo.
echo Checking connection to MT5 (MT5 must be open and logged in, Algo Trading ON)...
"%PY%" -m autotrader -c config.yaml sizing
echo.
echo Done. Edit config.yaml with Notepad if needed, then double-click start_bot.bat
pause
exit /b 0

:findpython
set "PY="
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python312\python.exe" "%ProgramFiles%\Python313\python.exe") do (
    if not defined PY if exist %%P set "PY=%%~P"
)
if defined PY exit /b 0
for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%I"
if defined PY exit /b 0
for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%I"
exit /b 0
