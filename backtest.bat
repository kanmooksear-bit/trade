@echo off
REM Double-click to backtest with real prices from MT5 (MT5 must be open and logged in).
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist python_path.txt set /p PY=<python_path.txt

set "DAYS=90"
set /p DAYS=How many days back to test? [90]: 

echo.
echo === 1/3 Full bot (learning + trade every day) ===
"%PY%" -m autotrader -c config.yaml backtest --days %DAYS% > backtest_result.txt 2>&1
type backtest_result.txt

echo.
echo === 2/3 Without learning (for comparison) ===
"%PY%" -m autotrader -c config.yaml backtest --days %DAYS% --no-learning > backtest_nolearning.txt 2>&1
type backtest_nolearning.txt | findstr /n "^" | findstr "^[1-7]:"

echo.
echo === 3/3 Without the trade-every-day rule (for comparison) ===
"%PY%" -m autotrader -c config.yaml backtest --days %DAYS% --no-daily > backtest_nodaily.txt 2>&1
type backtest_nodaily.txt | findstr /n "^" | findstr "^[1-7]:"

echo.
echo Saved: backtest_result.txt, backtest_nolearning.txt, backtest_nodaily.txt
pause
