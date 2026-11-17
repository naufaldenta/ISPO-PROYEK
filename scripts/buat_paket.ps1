# Membuat folder portabel: hanya bagian yang dibutuhkan untuk menjalankan SAHABAT.
#   powershell -ExecutionPolicy Bypass -File scripts\buat_paket.ps1 [-Tujuan D:\SAHABAT]
# Hasilnya tinggal disalin (flashdisk/zip) ke laptop lain lalu klik dua kali "Mulai SAHABAT.bat".
param([string]$Tujuan = "")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Tujuan) { $Tujuan = Join-Path $root "dist\SAHABAT" }

foreach ($perlu in @("runtime\python\python.exe", "sahabat\__main__.py", "models\yolo11n-pose_320_openvino_model")) {
    if (-not (Test-Path (Join-Path $root $perlu))) { throw "Belum lengkap: $perlu tidak ada." }
}

function Salin($dari, $ke, $buang) {
    # robocopy: kode keluar di bawah 8 berarti berhasil
    $argumen = @($dari, $ke, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:1", "/W:1")
    if ($buang) { $argumen += @("/XD") + $buang }
    & robocopy @argumen | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Gagal menyalin $dari (robocopy $LASTEXITCODE)" }
}

if (Test-Path $Tujuan) { Remove-Item -Recurse -Force $Tujuan }
New-Item -ItemType Directory -Force $Tujuan | Out-Null

Write-Host "Menyalin runtime (paling lama) ..."
Salin (Join-Path $root "runtime") (Join-Path $Tujuan "runtime") $null
Salin (Join-Path $root "sahabat") (Join-Path $Tujuan "sahabat") @("__pycache__")

# models: bobot YOLO, ekspor OpenVINO, dan model kekerasan bila sudah dilatih
Salin (Join-Path $root "models") (Join-Path $Tujuan "models") $null

Copy-Item -LiteralPath (Join-Path $root "Mulai SAHABAT.bat") -Destination $Tujuan
Copy-Item -LiteralPath (Join-Path $root "Mulai SAHABAT (rekam uji).bat") -Destination $Tujuan
Copy-Item -LiteralPath (Join-Path $root "kata-kasar.txt") -Destination $Tujuan
if (-not (Test-Path (Join-Path $root "models\whisper-base\model.bin"))) {
    Write-Warning "Model ucapan belum ada (python -m sahabat.speech --unduh); paket tetap jalan tanpa peringatan kata kasar."
}
New-Item -ItemType Directory -Force (Join-Path $Tujuan "data") | Out-Null   # kosong: pengaturan dibuat ulang di laptop tujuan

$ukuran = (Get-ChildItem -Recurse -File $Tujuan | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Paket siap di {0}  ({1:N2} GB)" -f $Tujuan, $ukuran)
