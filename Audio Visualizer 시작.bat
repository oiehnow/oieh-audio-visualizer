@echo off
rem One-click dev launcher: bootstraps the venv if needed, then starts the app
rem windowless (the browser tab opens automatically; quit from the app's 종료 button).
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo uv가 설치되어 있지 않습니다. https://docs.astral.sh/uv/ 를 참고해 설치해주세요.
        pause
        exit /b 1
    )
    echo 처음 실행: 의존성을 설치하는 중입니다...
    uv sync
    if errorlevel 1 (
        echo 의존성 설치에 실패했습니다.
        pause
        exit /b 1
    )
)

start "" ".venv\Scripts\pythonw.exe" -m visualizer
exit /b 0
