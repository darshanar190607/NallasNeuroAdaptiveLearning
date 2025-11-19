#!/bin/bash

echo "🧠 Starting NeuroAdaptive EdTech Servers..."
echo "============================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed or not in PATH"
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip3 install -r requirements_fastapi.txt

echo ""
echo "🤖 Extracting BCI model..."
python3 extract_model.py

echo ""
echo "🚀 Starting FastAPI BCI Server on port 8000..."
python3 fastapi_server.py &
FASTAPI_PID=$!

sleep 3

echo "🚀 Starting Node.js API Server on port 5000..."
cd server
npm start &
NODEJS_PID=$!
cd ..

sleep 3

echo "🚀 Starting React Development Server on port 3000..."
npm start &
REACT_PID=$!

echo ""
echo "✅ All servers are starting..."
echo "- FastAPI BCI Server: http://localhost:8000"
echo "- Node.js API Server: http://localhost:5000"
echo "- React App: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $FASTAPI_PID 2>/dev/null
    kill $NODEJS_PID 2>/dev/null
    kill $REACT_PID 2>/dev/null
    echo "✅ All servers stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user input
wait