@echo off
REM ModelFinder Launcher - Ensures correct working directory
cd /d "%~dp0"
.venv\Scripts\python.exe main_enhanced.py
pause

