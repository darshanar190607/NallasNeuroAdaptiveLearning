@echo off
echo Starting NeuroAdaptive EdTech Application...
echo.

echo [1/3] Starting Backend Server...
start "Backend" cmd /k "cd server && set PORT=5001 && node server.js"

timeout /t 5

echo [2/3] Starting BCI Service...
start "BCI Service" cmd /k "python fastapi_server.py"

timeout /t 3

echo [3/3] Starting Frontend...
start "Frontend" cmd /k "npm run dev"

echo.
echo ✅ All services starting...
echo.
echo 🌐 Frontend: http://localhost:5173
echo 🔧 Backend: http://localhost:5001
echo 🧠 BCI Service: http://localhost:8000
echo.
pause