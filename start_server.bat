@echo off
echo Starting FastAPI Server with Telegram Scheduler...
echo ================================================

echo Testing configuration first...
py simple_test.py

echo.
echo Starting server...
py run.py

pause