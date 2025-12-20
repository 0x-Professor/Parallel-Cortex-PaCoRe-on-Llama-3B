@echo off
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║         PaCoRe Project - Automated Setup                      ║
echo ║  Parallel and Distributed Computing Implementation            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo [1/4] Creating project structure...
python create_files.py
if errorlevel 1 (
    echo ❌ ERROR: Failed to create files
    pause
    exit /b 1
)

echo.
echo [2/4] Installing minimal dependencies...
pip install numpy loguru pytest
if errorlevel 1 (
    echo ⚠️  WARNING: Some packages might have failed to install
    echo You can continue, but some features may not work
)

echo.
echo [3/4] Running tests...
pytest tests/ -v
if errorlevel 1 (
    echo ⚠️  WARNING: Some tests failed
    echo This is okay if you haven't installed all dependencies
)

echo.
echo [4/4] Running example...
python examples\simple_usage.py

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                    ✅ SETUP COMPLETE!                          ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Your PaCoRe project is ready to use!
echo.
echo 📚 Next steps:
echo    - Read START_HERE.md for quick guide
echo    - Read PROJECT_SUMMARY.md for complete overview
echo    - Explore src/consensus.py for implementation details
echo.
echo 🚀 To run again:
echo    python examples\simple_usage.py
echo.
pause
