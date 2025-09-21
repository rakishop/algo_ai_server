@echo off
cd /d "e:\API CALLS\MyAlgoFax\historydata"
"C:\Users\mosin\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause