@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SAHABAT Sekolah
set "PY=%~dp0runtime\python\python.exe"

rem Penyalaan pertama: siapkan Python dan pustaka di folder runtime (butuh internet sekali).
if exist "%~dp0runtime\siap.txt" goto model
if exist "%PY%" (
    "%PY%" -c "import ultralytics, openvino, faster_whisper, fastapi" >nul 2>&1
    if not errorlevel 1 (
        echo runtime siap>"%~dp0runtime\siap.txt"
        goto model
    )
)
echo ==============================================================
echo   SAHABAT Sekolah - persiapan pertama
echo   Mengunduh Python dan pustaka, sekitar 1 GB.
echo   Butuh internet dan hanya sekali. Lama: 10-30 menit.
echo   Jangan tutup jendela ini.
echo ==============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\siapkan_runtime.ps1"
if errorlevel 1 goto gagal

:model
rem Model pengenal ucapan untuk peringatan kata kasar, diunduh sekali bila belum ada.
if exist "%~dp0models\whisper-base\model.bin" goto ulang
echo Mengunduh model pengenal ucapan, sekitar 150 MB...
"%PY%" -m sahabat.speech --unduh
if errorlevel 1 echo Model ucapan gagal diunduh. SAHABAT tetap jalan tanpa peringatan kata kasar.

:ulang
"%PY%" -m sahabat
if %errorlevel%==0 goto selesai
echo SAHABAT berhenti tidak wajar. Menyalakan ulang dalam 3 detik...
ping -n 4 127.0.0.1 >nul
goto ulang

:gagal
echo.
echo Persiapan gagal. Periksa sambungan internet, lalu jalankan file ini lagi.
pause
exit /b 1

:selesai
