@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_VENV=%~dp0.venv_yolo\Scripts\python.exe"

if not exist "%PYTHON_VENV%" (
    echo ERROR: Python virtual environment tidak ditemukan:
    echo %PYTHON_VENV%
    echo.
    echo Buat environment dengan:
    echo py -3.11 -m venv .venv_yolo
    pause
    exit /b 1
)

echo Menjalankan server...
echo Alamat: http://127.0.0.1:7863
echo.

"%PYTHON_VENV%" -m uvicorn experiments.api_web_hybrid:app --host 127.0.0.1 --port 7863

echo.
echo Server berhenti.
pause