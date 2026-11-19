@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scan-shots.ps1"
echo.
echo Opening compare page...
start "" "%~dp0index.html"
timeout /t 2 >nul
