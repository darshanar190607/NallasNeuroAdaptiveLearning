@echo off
echo Starting NeuroAdaptive EdTech Application...
echo.

echo [1/3] Starting Backend Server...
cd server
start "Backend Server" cmd /k "npm start"
cd ..

timeout /t 3

echo [2/3] Starting BCI Service...
start "BCI Service" cmd /k "python fastapi_server.py"

timeout /t 3

echo [3/3] Starting Frontend...
start "Frontend" cmd /k "python -m http.server 3000"

echo.
echo ✅ All services started!
echo.
echo 🌐 Frontend: http://localhost:3000
echo 🔧 Backend: http://localhost:5001
echo 🧠 BCI Service: http://localhost:8000
echo.
echo Press any key to exit...
pause > nul