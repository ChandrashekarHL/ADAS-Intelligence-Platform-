@echo off
setlocal enabledelayedexpansion
title AIP launcher
cd /d "%~dp0"

echo(
echo  ============================================
echo   ADAS Intelligence Platform - starting up
echo  ============================================
echo(

REM ---------------------------------------------------------------- prerequisites
if not exist "backend\.venv\Scripts\python.exe" (
  echo  [X] Python venv missing at backend\.venv
  echo      Create it with:
  echo        cd backend
  echo        python -m venv .venv
  echo        .venv\Scripts\python.exe -m pip install -e ".[dev]"
  goto :fail
)

REM The C: drive on this machine is full, so keep npm's cache and all temp files on D:
REM when that drive exists; elsewhere the inherited defaults are left alone. Set here so
REM both child windows inherit them.
if exist "D:\" (
  if not exist "D:\tmp" mkdir "D:\tmp"
  if not exist "D:\npm-cache" mkdir "D:\npm-cache"
  set "TMP=D:\tmp"
  set "TEMP=D:\tmp"
  set "npm_config_cache=D:\npm-cache"
)

if not exist "frontend\node_modules" (
  echo  [warn] frontend\node_modules missing - installing, this takes a few minutes...
  pushd frontend
  call npm install
  if errorlevel 1 (
    popd
    echo  [X] npm install failed
    goto :fail
  )
  popd
  echo  [ok] frontend dependencies installed
)

REM ------------------------------------------------- choose the LLM provider
REM Presence-only check of OPENAI_API_KEY, parsed with the same python-dotenv rules the
REM backend uses (leading spaces, "export " prefix, quoted values; a quoted empty string
REM is empty). The key is never printed. When a key is present the provider is left to
REM backend\.env and the environment; only the offline fallback is forced from here.
if not exist "backend\.env" (
  echo  [warn] backend\.env not found - starting offline with the fake provider
  set "LLM_PROVIDER=fake"
  set "INDEX_DIR=../data/index"
) else (
  "backend\.venv\Scripts\python.exe" -c "import sys; from dotenv import dotenv_values; sys.exit(0 if (dotenv_values('backend/.env').get('OPENAI_API_KEY') or '').strip() else 1)" >nul 2>&1
  if errorlevel 1 (
    echo  [warn] OPENAI_API_KEY is empty in backend\.env - starting offline with the fake provider
    echo      Everything works except "Ask the agent", which returns 503.
    set "LLM_PROVIDER=fake"
    set "INDEX_DIR=../data/index"
  ) else (
    echo  [ok] OPENAI_API_KEY found - provider left to backend\.env / environment
  )
)

REM ------------------------------------------------------------ launch the two servers
REM Both children inherit the environment set above, so no inline "set" is needed
REM inside the command strings. An inline "set X=v && ..." would put the space
REM before "&&" into the value and break strict settings validation.
echo(
echo  Starting the API on  http://127.0.0.1:8000   (window: AIP backend)
start "AIP backend" /D "%~dp0backend" cmd /k .venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000

echo  Starting the UI  on  http://localhost:3000    (window: AIP frontend)
start "AIP frontend" /D "%~dp0frontend" cmd /k npm run dev

REM ------------------------------------------------------------------ wait for both
echo(
echo  Waiting for the API to answer...
set /a tries=0
:waitapi
curl -s -o nul http://127.0.0.1:8000/api/health && goto :apiup
set /a tries+=1
if !tries! geq 40 (
  echo  [warn] The API did not answer after 40s - check the "AIP backend" window.
  set "PROV=unknown (health unavailable)"
  goto :waitui
)
timeout /t 1 /nobreak >nul
goto :waitapi
:apiup
echo  [ok] API is up
REM The API is the only authority on which provider is active (an inherited
REM LLM_PROVIDER can override backend\.env), so ask it rather than guessing.
set "PROV=unknown (health unavailable)"
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "try { (Invoke-RestMethod -TimeoutSec 5 http://127.0.0.1:8000/api/health).llm_provider } catch { }"`) do set "PROV=%%p"

:waitui
echo  Waiting for the dashboard to compile (the first run takes a while)...
set /a tries=0
:waitweb
curl -s -o nul http://localhost:3000 && goto :webup
set /a tries+=1
if !tries! geq 120 (
  echo  [warn] The dashboard did not answer after 120s - check the "AIP frontend" window.
  goto :done
)
timeout /t 1 /nobreak >nul
goto :waitweb
:webup
echo  [ok] Dashboard is up
start "" http://localhost:3000

:done
echo(
echo  ============================================
echo   Dashboard : http://localhost:3000
echo   API docs  : http://127.0.0.1:8000/docs
echo   Provider  : !PROV!
echo(
echo   Demo data to upload:
echo     data\demo\aeb_late_braking_seed42\telemetry.csv
echo     data\demo\aeb_late_braking_seed42\scenario.json
echo(
echo   To stop: close the "AIP backend" and "AIP frontend" windows,
echo            or press Ctrl+C in each.
echo  ============================================
echo(
pause
exit /b 0

:fail
echo(
echo  Startup aborted.
pause
exit /b 1
