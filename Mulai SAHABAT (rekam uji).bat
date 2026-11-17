@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SAHABAT Sekolah (merekam uji)
if exist "%~dp0runtime\python\python.exe" goto mulai
echo Runtime belum ada. Jalankan "Mulai SAHABAT.bat" sekali dulu untuk persiapan pertama.
pause
exit /b 1

:mulai
echo Mode uji: video kamera DISIMPAN di data\rekaman untuk penyetelan.
echo Pakai hanya dengan izin semua orang yang terekam. Hapus rekamannya setelah selesai.
echo.
:ulang
"%~dp0runtime\python\python.exe" -m sahabat --rekam
if %errorlevel%==0 goto selesai
echo SAHABAT berhenti tidak wajar. Menyalakan ulang dalam 3 detik...
ping -n 4 127.0.0.1 >nul
goto ulang
:selesai
