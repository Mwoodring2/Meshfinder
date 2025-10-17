@echo off
ECHO ========================================
ECHO ModelFinder - Rebuild with New Features
ECHO ========================================
ECHO.
ECHO This will rebuild ModelFinder.exe with:
ECHO - Proposal system
ECHO - Fuzzy matching
ECHO - Reference parts integration
ECHO.
PAUSE

cd /d "%~dp0"

ECHO.
ECHO [1/3] Activating virtual environment...
CALL .venv\Scripts\activate.bat

ECHO.
ECHO [2/3] Installing build dependencies...
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pyinstaller

ECHO.
ECHO [3/3] Building executable...
IF EXIST app.ico (
    pyinstaller --noconfirm --windowed --icon app.ico --name ModelFinder main.py
) ELSE (
    ECHO Warning: app.ico not found, building without custom icon
    pyinstaller --noconfirm --windowed --name ModelFinder main.py
)

ECHO.
ECHO ========================================
ECHO Build Complete!
ECHO ========================================
ECHO.
ECHO New executable: dist\ModelFinder\ModelFinder.exe
ECHO.
PAUSE

