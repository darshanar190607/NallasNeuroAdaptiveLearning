@echo off
echo Starting NeuroAdaptive EdTech Complete System...
echo.

echo Starting MongoDB (make sure MongoDB is installed and running)
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd server && npm start"

timeout /t 3

echo Starting FastAPI BCI Service...
start "BCI Service" cmd /k "python fastapi_server.py"

timeout /t 3

echo Starting Frontend Development Server...
start "Frontend" cmd /k "npm run dev"

echo.
echo All services are starting up...
echo Backend: http://localhost:5001
echo BCI Service: http://localhost:8000  
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause > nul