@echo off
REM Starts the Market Forecaster dashboard. Keep this file in the forecaster folder.
cd /d "%~dp0"
title Market Forecaster - close this window to stop the app

REM If the app is already running, just open it in the browser again
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    start "" http://localhost:8501
    exit /b
)

if not exist ".venv\Scripts\python.exe" (
    echo Could not find the virtual environment in:
    echo   %cd%\.venv
    echo Make sure this file is in the forecaster folder.
    pause
    exit /b
)

echo Starting Market Forecaster... your browser will open in a few seconds.
echo Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run src\app.py --server.port 8501 --browser.gatherUsageStats false

REM Only reached if Streamlit stops with an error, so the message stays readable
echo.
echo The app stopped. If you see an error above, copy it and ask for help.
pause
