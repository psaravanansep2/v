@echo off
rem Double-click to install v, the free voice assistant.
echo Installing v...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/psaravanansep2/v/main/install.ps1 | iex"
echo.
pause
