@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title SAHABAT - latih model kamera lomba
set PYTHONUTF8=1
if exist "runtime\python\python.exe" goto mulai
echo Runtime belum ada. Jalankan "Mulai SAHABAT.bat" sekali dulu.
pause
exit /b 1

:mulai
echo Melatih model untuk Posisi kamera "Dekat / lomba" dari rekaman yang sudah dibuat.
echo Tutup aplikasi SAHABAT dulu. Proses ini sekitar 15-30 menit. Jangan tutup jendela ini.
echo.
"runtime\python\python.exe" -m latih.kalibrasi_dekat latih
if errorlevel 1 goto gagal
echo.
choice /C YT /M "Pasang model ini (Y = ya, T = tidak)"
if errorlevel 2 goto selesai
"runtime\python\python.exe" -m latih.kalibrasi_dekat pasang
goto selesai

:gagal
echo.
echo Model tidak dipasang. Aplikasi tetap memakai model yang sekarang.

:selesai
echo.
pause
