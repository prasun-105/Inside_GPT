@echo off
rem Start the chat server and open it in the browser (Windows).
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" server.py --open %*
pause
