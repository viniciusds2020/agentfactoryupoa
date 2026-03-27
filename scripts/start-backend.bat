@echo off
cd /d %~dp0..
echo Parando processos anteriores na porta 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo Iniciando backend...
.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8001
pause
