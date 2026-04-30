@echo off
REM Pest Detection Backend Startup Script for Windows

echo ========================================
echo  🚀 PEST DETECTION BACKEND STARTUP
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ ERROR: Virtual environment not found!
    echo. 
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if model file exists
if not exist "pest_model.h5" (
    echo ❌ ERROR: pest_model.h5 not found!
    echo Please ensure the model file is in this directory.
    pause
    exit /b 1
)

if not exist "class_names.json" (
    echo ❌ ERROR: class_names.json not found!
    echo Please ensure the class_names file is in this directory.
    pause
    exit /b 1
)

REM Start the backend server
echo.
echo ✅ All checks passed!
echo.
echo 🌐 Starting FastAPI server...
echo 📍 Backend will be available at: http://192.168.1.105:8000
echo 📍 API docs at: http://192.168.1.105:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m uvicorn main:app --host 192.168.1.105 --port 8000

pause
