@echo off
REM ============================================
REM Build ModelFinder as Windows .exe files
REM ============================================

set VENV_PY=.venv\Scripts\python.exe

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing requirements...
%VENV_PY% -m pip install --upgrade pip
%VENV_PY% -m pip install -r requirements.txt
%VENV_PY% -m pip install pyinstaller

echo.
echo Building ModelFinder executables...
echo.

REM Build the main indexer executable
echo [1/3] Building ModelFinder Indexer...
%VENV_PY% -m PyInstaller ^
    --onefile ^
    --name ModelFinder_Indexer ^
    --distpath dist ^
    --workpath build ^
    --add-data "src;src" ^
    --hidden-import "trimesh" ^
    --hidden-import "numpy" ^
    --hidden-import "sqlite3" ^
    --hidden-import "click" ^
    --hidden-import "tqdm" ^
    --console ^
    --clean ^
    src\indexer\modelfinder_indexer.py

REM Build the scanner executable
echo [2/3] Building ModelFinder Scanner...
%VENV_PY% -m PyInstaller ^
    --onefile ^
    --name ModelFinder_Scanner ^
    --distpath dist ^
    --workpath build ^
    --add-data "src;src" ^
    --hidden-import "trimesh" ^
    --hidden-import "numpy" ^
    --hidden-import "sqlite3" ^
    --hidden-import "tqdm" ^
    --console ^
    --clean ^
    scripts\scan_folder.py

REM Build the searcher executable
echo [3/3] Building ModelFinder Searcher...
%VENV_PY% -m PyInstaller ^
    --onefile ^
    --name ModelFinder_Searcher ^
    --distpath dist ^
    --workpath build ^
    --add-data "src;src" ^
    --hidden-import "click" ^
    --hidden-import "sqlite3" ^
    --console ^
    --clean ^
    scripts\search_cli.py

echo.
echo ============================================
echo [OK] Build completed successfully!
echo ============================================
echo.
echo Executables created in dist\:
echo   - ModelFinder_Indexer.exe  (Main application)
echo   - ModelFinder_Scanner.exe  (Scan models)
echo   - ModelFinder_Searcher.exe (Search models)
echo.
echo Usage:
echo   ModelFinder_Indexer.exe scan C:\My3DModels
echo   ModelFinder_Indexer.exe search dragon
echo   ModelFinder_Scanner.exe C:\My3DModels --verbose
echo   ModelFinder_Searcher.exe --stats
echo.
pause
