# Memasang paket GPU untuk Mode CCTV: onnxruntime-directml (GPU NVIDIA, AMD, atau Intel lewat DirectML).
# Butuh internet sekali (sekitar 20 MB dari PyPI). Laptop pameran atau sekolah tanpa internet: jalankan skrip ini di
# rumah, lalu salin seluruh folder SAHABAT lewat flashdisk.
# Paket ini menggantikan onnxruntime biasa (dipakai pengenal suara); pengenal suara tetap berjalan dengan paket GPU.
# Paket diunduh dulu; onnxruntime biasa baru dilepas bila unduhan berhasil, jadi tanpa internet tidak ada yang rusak.
param([switch]$Ya)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "runtime\python\python.exe"
$gpuLock = Join-Path $root "requirements-gpu.lock"
$model = Join-Path $root "models\yolo11n-pose_416.onnx"

if (-not (Test-Path $py)) {
    Write-Host "Runtime belum ada. Jalankan 'Mulai SAHABAT.bat' dulu, lalu ulangi skrip ini."
    exit 1
}

$gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch "Microsoft Basic" } |
          ForEach-Object { $_.Name })
if ($gpus.Count -eq 0) {
    Write-Host "Tidak ada kartu grafis yang terdeteksi. Mode CCTV tetap memakai CPU."
    exit 0
}
Write-Host "Kartu grafis: $($gpus -join ', ')"

# Program luar dijalankan tanpa mengalihkan keluaran galatnya: di Windows PowerShell 5.1, "2>$null" dengan
# ErrorActionPreference Stop menghentikan skrip begitu program menulis peringatan.
function Invoke-Quiet([scriptblock]$Command) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command *> $null; return $LASTEXITCODE } finally { $ErrorActionPreference = $old }
}

function Test-Dml {
    $code = Invoke-Quiet { & $py -c "import onnxruntime as o, sys; sys.exit(0 if 'DmlExecutionProvider' in o.get_available_providers() else 1)" }
    return ($code -eq 0)
}

if (Test-Dml) {
    Write-Host "Paket GPU sudah terpasang."
} else {
    Write-Host "Paket GPU (onnxruntime-directml, sekitar 20 MB) akan diunduh dari PyPI dan menggantikan onnxruntime biasa."
    if (-not $Ya) {
        $answer = Read-Host "Pasang sekarang? (Y/N)"
        if ($answer -notmatch '^[Yy]') {
            Write-Host "Dilewati. Mode CCTV tetap memakai CPU."
            exit 0
        }
    }
    $download = Join-Path $env:TEMP ("sahabat-gpu-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force $download | Out-Null
    try {
        & $py -m pip download --disable-pip-version-check --no-deps -d $download -r $gpuLock
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Paket GPU belum bisa diunduh (periksa internet). Tidak ada yang diubah; Mode CCTV memakai CPU."
            exit 1
        }
        # Ketergantungannya (numpy, protobuf, flatbuffers, sympy, packaging) sudah terpasang bersama onnxruntime biasa.
        & $py -m pip uninstall -y onnxruntime
        & $py -m pip install --disable-pip-version-check --no-warn-script-location --no-index --no-deps `
            --find-links $download -r $gpuLock
        if ($LASTEXITCODE -ne 0) {
            # Kembalikan onnxruntime biasa supaya pengenal suara tetap jalan (internet masih ada: unduhan tadi berhasil).
            $cpuLine = Select-String -Path (Join-Path $root "requirements.lock") -Pattern '^onnxruntime==' |
                       Select-Object -First 1
            if ($cpuLine) { & $py -m pip install --disable-pip-version-check --no-deps $cpuLine.Line.Trim() }
            Write-Host "Paket GPU gagal dipasang. Mode CCTV tetap memakai CPU."
            exit 1
        }
    } finally {
        Remove-Item -Recurse -Force $download -ErrorAction SilentlyContinue
    }
}

& $py -c "import onnxruntime as o; print('Penyedia ONNX Runtime:', ', '.join(o.get_available_providers()))"
if (-not (Test-Dml)) {
    Write-Host "DirectML belum tersedia. Perbarui driver kartu grafis, lalu jalankan skrip ini lagi."
    exit 1
}
if (-not (Test-Path $model)) {
    Write-Host "Model GPU models\yolo11n-pose_416.onnx belum ada. Salin dari paket rilis SAHABAT ke folder models."
    exit 1
}
Write-Host "GPU siap. Di panel Kamera, pilih 'Perangkat hitung': Otomatis atau GPU."
Write-Host "Otomatis memakai GPU hanya bila uji kesetaraan lolos di GPU ini (python -m latih.banding_gpu, untuk pengembang)."
