@echo off
REM Kill any existing Python processes for this project
echo Stopping any running Python processes...
taskkill /F /FI "WINDOWTITLE eq ModelFinder*" 2>nul
timeout /t 2 /nobreak >nul

REM Launch fresh
echo Starting ModelFinder with all dependencies...
cd /d "%~dp0"
.venv\Scripts\python.exe main_enhanced.py
pause

