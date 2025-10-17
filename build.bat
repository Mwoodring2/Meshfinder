@echo off
SETLOCAL
IF NOT EXIST .venv ( py -3.10 -m venv .venv )
CALL .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --icon app.ico --name ModelFinder main_enhanced.py
ECHO Done. See dist\ModelFinder\ModelFinder.exe
PAUSE