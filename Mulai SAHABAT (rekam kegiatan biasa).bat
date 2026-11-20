@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SAHABAT Sekolah (merekam kegiatan biasa)
if exist "%~dp0runtime\python\python.exe" goto mulai
echo Runtime belum ada. Jalankan "Mulai SAHABAT.bat" sekali dulu untuk persiapan pertama.
pause
exit /b 1

:mulai
echo Mode uji: video kegiatan biasa (tanpa adegan) DISIMPAN di data\rekaman untuk mengukur alarm keliru.
echo Pakai hanya dengan izin sekolah dan semua orang yang terekam. Hapus rekamannya setelah selesai.
echo Pilih kamera (misalnya CCTV sekolah) di Perangkat, dan posisi kamera di Perangkat sebelum merekam.
echo.
:ulang
"%~dp0runtime\python\python.exe" -m sahabat --rekam --biasa
if %errorlevel%==0 goto selesai
echo SAHABAT berhenti tidak wajar. Menyalakan ulang dalam 3 detik...
ping -n 4 127.0.0.1 >nul
goto ulang
:selesai
