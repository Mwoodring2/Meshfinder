#!/bin/bash
# =====================================================
# ModelFinder — Dependency Installer (Linux/Mac)
# Installs all required packages for 3D mesh processing
# =====================================================

echo ""
echo "🚀 Installing ModelFinder dependencies..."
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment detected. Using system Python."
fi

echo ""
echo "📦 Installing core 3D processing libraries..."
pip install --upgrade pip
pip install trimesh numpy pillow

echo ""
echo "🤖 Installing ML packages for Similarity Search..."
pip install faiss-cpu scikit-learn joblib

echo ""
echo "📊 Installing Excel import dependencies..."
pip install pandas openpyxl

echo ""
echo "🎨 Installing GUI dependencies..."
pip install PySide6

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "🎉 ModelFinder is ready to use with:"
echo "   • 3D mesh processing (trimesh, numpy, pillow)"
echo "   • Similarity search (faiss-cpu, scikit-learn)"
echo "   • Excel import (pandas, openpyxl)"
echo "   • GUI interface (PySide6)"
echo ""
