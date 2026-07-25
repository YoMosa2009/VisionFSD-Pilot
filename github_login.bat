@echo off
setlocal
cd /d "%~dp0"
title VisionFSD - GitHub Login
color 0A
echo.
echo ============================================
echo  VisionFSD Pilot - GitHub CLI login
echo ============================================
echo.
echo 1) A one-time code will appear below
echo 2) Browser opens to github.com/login/device
echo 3) Paste the code, approve GitHub CLI
echo 4) Wait for "Logged in" then close this window
echo.
echo ============================================
echo.

where gh >nul 2>&1
if errorlevel 1 (
  echo ERROR: gh was not found. Install GitHub CLI:
  echo   https://cli.github.com/
  pause
  exit /b 1
)

gh auth login --hostname github.com --git-protocol https --web --skip-ssh-key
set ERR=%ERRORLEVEL%
echo.
if "%ERR%"=="0" (
  echo SUCCESS. Logged in as:
  gh api user --jq .login
  echo.
  echo You can close this window and tell the agent: done
) else (
  echo Login failed with code %ERR%.
  echo.
  echo Alternative: create a classic PAT with "repo" scope at
  echo   https://github.com/settings/tokens
  echo then run:
  echo   echo YOUR_TOKEN| gh auth login --with-token
)
echo.
pause
endlocal
