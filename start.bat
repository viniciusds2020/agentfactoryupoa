@echo off
setlocal
cd /d %~dp0

echo.
echo  =============================================
echo   Agent Factory - Setup e Inicializacao
echo  =============================================
echo.

:: ── Verificar Python ────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado. Instale Python 3.12+ em python.org
    pause
    exit /b 1
)

:: ── Criar venv se nao existir ───────────────────────────────────────────
if not exist .venv (
    echo [1/3] Criando ambiente virtual...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao criar ambiente virtual.
        pause
        exit /b 1
    )
)

:: ── Instalar dependencias se necessario ─────────────────────────────────
if not exist .venv\Lib\site-packages\fastapi (
    echo [2/3] Instalando dependencias (primeira vez, pode demorar alguns minutos)...
    .venv\Scripts\pip.exe install -q -r requirements-simple.txt
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao instalar dependencias.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Dependencias ja instaladas.
)

:: ── Criar .env se nao existir ───────────────────────────────────────────
if not exist .env (
    echo.
    echo Voce precisa de uma chave da Groq (gratuita).
    echo Obtenha em: https://console.groq.com/keys
    echo.
    set /p GROQ_KEY="Cole sua GROQ_API_KEY: "
    (
        echo GROQ_API_KEY=%GROQ_KEY%
    ) > .env
    echo Arquivo .env criado com sucesso.
)

:: ── Parar processo anterior na porta 8001 ───────────────────────────────
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: ── Iniciar servidor ────────────────────────────────────────────────────
echo.
echo [3/3] Iniciando Agent Factory...
echo.
echo  Acesse no navegador: http://localhost:8001
echo  Para parar: Ctrl+C
echo.

:: Abrir navegador automaticamente
start http://localhost:8001

.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
pause
