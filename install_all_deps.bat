@echo off
REM =====================================================
REM ModelFinder — Complete Dependency Installer
REM Installs all required packages for 3D mesh processing
REM =====================================================

echo.
echo 🚀 Installing all ModelFinder dependencies...
echo.

REM --- Detect venv automatically if present ---
IF EXIST ".venv\Scripts\activate.bat" (
    echo ✅ Activating local virtual environment...
    call .venv\Scripts\activate.bat
) ELSE (
    echo ⚠️  No local venv detected. Using system Python.
)

echo.
echo 📦 Installing core 3D processing libraries...
python -m pip install --upgrade pip
python -m pip install trimesh numpy pillow --prefer-binary

echo.
echo 🤖 Installing ML packages for Similarity Search...
python -m pip install faiss-cpu scikit-learn joblib --prefer-binary

echo.
echo 📊 Installing Excel import dependencies...
python -m pip install pandas openpyxl --prefer-binary

echo.
echo 🎨 Installing GUI dependencies...
python -m pip install PySide6 --prefer-binary

echo.
echo ✅ All dependencies installed successfully!
echo.
echo 🎉 ModelFinder is ready to use with:
echo    • 3D mesh processing (trimesh, numpy, pillow)
echo    • Similarity search (faiss-cpu, scikit-learn)
echo    • Excel import (pandas, openpyxl)
echo    • GUI interface (PySide6)
echo.
pause
