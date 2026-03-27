@echo off
echo ========================================================
echo Starting Unified AI Lab and Analysis Platform
echo ========================================================

echo Cleaning up old processes...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

echo [1/2] Starting FastAPI Backend...
start "Backend Server" cmd /k "python main.py"

echo [2/2] Starting Next.js Frontend...
cd frontend
start "Frontend Server" cmd /k "npm run dev"

echo Both services have been launched!
echo Backend is available at: http://127.0.0.1:8000
echo Frontend is available at: http://localhost:3000
echo ========================================================
