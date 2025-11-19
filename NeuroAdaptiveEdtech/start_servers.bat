@echo off
echo Starting NeuroAdaptive EdTech Servers...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing Python dependencies...
pip install -r requirements_fastapi.txt

echo.
echo Extracting BCI model...
python extract_model.py

echo.
echo Starting FastAPI BCI Server on port 8000...
start "FastAPI BCI Server" cmd /k "python fastapi_server.py"

timeout /t 3 /nobreak >nul

echo Starting Node.js API Server on port 5000...
cd server
start "Node.js API Server" cmd /k "npm start"
cd ..

timeout /t 3 /nobreak >nul

echo Starting React Development Server on port 3000...
start "React Dev Server" cmd /k "npm start"

echo.
echo All servers are starting...
echo - FastAPI BCI Server: http://localhost:8000
echo - Node.js API Server: http://localhost:5000
echo - React App: http://localhost:3000
echo.
echo Press any key to exit...
pause >nul