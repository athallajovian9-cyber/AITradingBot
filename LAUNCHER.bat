@echo off
setlocal EnableExtensions
title AI Trading Bot - Honest Backtester
cd /d "%~dp0"

echo.
echo   ============================================
echo    AI TRADING BOT - HONEST BACKTESTER
echo   ============================================
echo.

rem ---------------------------------------------------------------- sanity
if not exist "AITradingBot.html" (
  echo   ERROR: AITradingBot.html is not in this folder.
  echo.
  echo   Keep LAUNCHER.bat in the same folder as AITradingBot.html.
  echo.
  pause
  exit /b 1
)

rem ------------------------------------------------- prefer the executable
if exist "AITradingBot.exe" (
  echo   Starting the backtester...
  start "" "AITradingBot.exe"
  exit /b 0
)

rem -------------------------------------------------- find a chromium browser
set "BROWSER="
for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
  "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
) do (
  if not defined BROWSER if exist "%%~P" set "BROWSER=%%~P"
)

set "HERE=%~dp0"
set "PAGE=%HERE%AITradingBot.html"

if defined BROWSER (
  echo   Browser: %BROWSER%
  set "PROFILE=%LocalAppData%\AITradingBot\profile"
  if not exist "%PROFILE%" mkdir "%PROFILE%" >nul 2>&1
  set "URL=file:///%HERE:\=/%AITradingBot.html"
  echo   Opening the backtester in an app window...
  start "" "%BROWSER%" --app="%URL%" --user-data-dir="%PROFILE%" --window-size=1440,960 --no-first-run --no-default-browser-check
  exit /b 0
)

echo   No Chrome or Edge found - opening with the default browser.
start "" "%PAGE%"
exit /b 0
