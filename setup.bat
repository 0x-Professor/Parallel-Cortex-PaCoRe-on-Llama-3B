@echo off
echo Creating PaCoRe project structure...

mkdir src 2>nul
mkdir src\models 2>nul
mkdir config 2>nul
mkdir tests 2>nul
mkdir examples 2>nul
mkdir logs 2>nul
mkdir data 2>nul

echo Project directories created!
echo.
echo Running Python setup script...
python setup_project.py

echo.
echo Setup complete!
pause
