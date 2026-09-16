@echo off
title VoiceShield Live Server
cd /d "%~dp0"

echo ============================================================
echo   VOICESHIELD // Real-Time AI Deepfake Detector
echo ============================================================
echo [1/2] Starting backend on NVIDIA RTX GPU (http://127.0.0.1:8080)...

start "VoiceShield Backend" /min cmd /c ".\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8080"
timeout /t 3 /nobreak >nul

echo [2/2] Generating secure public HTTPS / WSS tunnel...
echo ------------------------------------------------------------
echo Look for the 'trycloudflare.com' link below to share:
echo ------------------------------------------------------------
echo.

.\cloudflared.exe tunnel --url http://127.0.0.1:8080
