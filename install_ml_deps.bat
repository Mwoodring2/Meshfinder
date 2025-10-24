@echo off
REM =====================================================
REM ModelFinder — ML Dependency Installer
REM Installs FAISS + Scikit-Learn + supporting libraries
REM =====================================================

echo.
echo 🔧 Installing required Python ML packages for Similarity Search...
echo.

REM --- Detect venv automatically if present ---
IF EXIST ".venv\Scripts\activate.bat" (
    echo ✅ Activating local virtual environment...
    call .venv\Scripts\activate.bat
) ELSE (
    echo ⚠️  No local venv detected. Using system Python.
)

REM --- Install core dependencies (CPU version of FAISS) ---
python -m pip install --upgrade pip && ^
python -m pip install numpy faiss-cpu scikit-learn joblib --prefer-binary

echo.
echo ✅ Installation complete! 
echo.
echo You can now use Similarity Search in ModelFinder.
pause
