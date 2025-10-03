@echo off
SETLOCAL
IF NOT EXIST .venv ( py -3.11 -m venv .venv )
CALL .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt pyinstaller
IF EXIST app.ico (
    pyinstaller --noconfirm --windowed --icon app.ico --name ModelFinder main.py
) ELSE (
    ECHO Warning: app.ico not found, building without custom icon
    pyinstaller --noconfirm --windowed --name ModelFinder main.py
)
ECHO.
ECHO Done. See dist\ModelFinder\ModelFinder.exe
PAUSE