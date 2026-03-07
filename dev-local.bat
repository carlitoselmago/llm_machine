@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

if not exist "models" mkdir models

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==tuple([3,11]) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [dev-local] Existing .venv is not Python 3.11.
    echo [dev-local] Delete .venv and rerun this script.
    goto :fail
  )
  goto :venv_ready
)

echo [dev-local] Creating virtual environment...
where py >nul 2>nul
if errorlevel 1 goto :check_default_python

::py -3.11 -c "import sys; print(sys.version)" >nul 2>nul
::if errorlevel 1 (
::  echo [dev-local] Python 3.11 not found via py launcher.
::  echo [dev-local] Install Python 3.11 and rerun. Newer Python versions may fail on Rust-built deps.
::  goto :fail
::)

::py -3.11 -m venv .venv
::if errorlevel 1 goto :fail

::goto :venv_ready

:::check_default_python
::python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==tuple([3,11]) else 1)" >nul 2>nul
::if errorlevel 1 (
::  echo [dev-local] Python launcher not found and default python is not 3.11.
::  echo [dev-local] Please install or use Python 3.11.
::  goto :fail
::)

python -m venv .venv
if errorlevel 1 goto :fail

:venv_ready
set "REQ_FILE=backend\requirements.txt"
if exist "requirements-dev.txt" set "REQ_FILE=requirements-dev.txt"

if /I "%SKIP_PIP_INSTALL%"=="1" (
  echo [dev-local] Skipping pip install ^(SKIP_PIP_INSTALL=1^)
) else (
  echo [dev-local] Installing/updating Python dependencies from %REQ_FILE%...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "%REQ_FILE%"
  if errorlevel 1 goto :fail
)

if not defined VLLM_EXECUTABLE (
  if exist ".venv\Scripts\vllm.exe" (
    set "VLLM_EXECUTABLE=%CD%\.venv\Scripts\vllm.exe"
  ) else (
    where vllm >nul 2>nul
    if not errorlevel 1 (
      for /f "delims=" %%I in ('where vllm') do (
        set "VLLM_EXECUTABLE=%%I"
        goto :vllm_checked
      )
    )
    echo [dev-local] vLLM is not installed in this environment.
    echo [dev-local] Model start/stop will fail until vLLM is installed.
    echo [dev-local] Install with: .\.venv\Scripts\python.exe -m pip install vllm
  )
)

:vllm_checked
set "MODELS_DIR=%CD%\models"
set "ADMIN_USERNAME=admin"
set "ADMIN_PASSWORD=workshop"
set "FRONT_STATIC_DIR=%CD%\front\public"

echo [dev-local] Starting FastAPI dev server on http://localhost:8080
cd backend
"..\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXIT_CODE%

:fail
set "EXIT_CODE=%ERRORLEVEL%"
if not defined EXIT_CODE set "EXIT_CODE=1"
popd >nul
endlocal & exit /b %EXIT_CODE%
