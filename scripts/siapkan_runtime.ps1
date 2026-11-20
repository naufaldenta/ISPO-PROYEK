# Menyiapkan runtime portabel: Python 3.12 + semua pustaka, di dalam folder runtime/.
# Dipanggil otomatis oleh "Mulai SAHABAT.bat" saat pertama kali dijalankan. Butuh internet sekali saja.
# Boleh dijalankan ulang: bagian yang sudah ada dilewati, bagian yang gagal diulang.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root "runtime"
$py = Join-Path $runtime "python\python.exe"
$marker = Join-Path $runtime "siap.txt"

# Versi dikunci supaya setiap laptop mendapat runtime yang sama dengan yang sudah diuji.
$pythonTag = "20261110"
$pythonVersion = "3.12.16"
$pythonFile = "cpython-$pythonVersion+$pythonTag-x86_64-pc-windows-msvc-install_only.tar.gz"
$pythonUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$pythonTag/$pythonFile"
$torchIndex = "https://download.pytorch.org/whl/cpu"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) { throw "Gagal: $what (kode $LASTEXITCODE)" }
}

if (Test-Path $marker) { Remove-Item $marker }

if (-not (Test-Path $py)) {
    if (Test-Path $runtime) { Remove-Item -Recurse -Force $runtime }
    New-Item -ItemType Directory -Force $runtime | Out-Null
    Write-Host "[1/3] Mengunduh Python $pythonVersion ..."
    $tar = Join-Path $runtime $pythonFile
    & curl.exe -L --fail --silent --show-error --retry 3 -o $tar $pythonUrl
    Assert-Ok "unduh Python"
    & tar.exe -xzf $tar -C $runtime
    Assert-Ok "ekstrak Python"
    Remove-Item $tar
} else {
    Write-Host "[1/3] Python sudah ada."
}

Write-Host "[2/3] Memasang pustaka (paling lama, sekitar 1 GB) ..."
& $py -m pip install --disable-pip-version-check --no-warn-script-location --quiet --upgrade pip
Assert-Ok "pip upgrade"
# Paket GPU (onnxruntime-directml) menggantikan onnxruntime biasa; keduanya menulis folder onnxruntime/. pip tetap
# memasang onnxruntime biasa (faster-whisper memintanya dengan nama itu), jadi bila paket GPU sudah ada, paket GPU
# dipasang ulang sesudahnya. Pengecekan tanpa "2>$null": di PowerShell 5.1 itu menghentikan skrip (mode Stop).
function Test-Package($name) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $py -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" $name *> $null; return ($LASTEXITCODE -eq 0) }
    finally { $ErrorActionPreference = $old }
}
$hasGpuPackage = Test-Package "onnxruntime-directml"
& $py -m pip install --disable-pip-version-check --no-warn-script-location --no-cache-dir `
    -r (Join-Path $root "requirements.lock") --extra-index-url $torchIndex
Assert-Ok "pasang pustaka"
if ($hasGpuPackage) {
    & $py -m pip uninstall -y onnxruntime onnxruntime-directml
    & $py -m pip install --disable-pip-version-check --no-warn-script-location --no-cache-dir --no-deps `
        -r (Join-Path $root "requirements-gpu.lock")
    Assert-Ok "pasang ulang paket GPU"
}

Write-Host "[3/3] Memeriksa pemasangan ..."
& $py -c "import ultralytics, openvino, cv2, sklearn, fastapi, uvicorn, faster_whisper, sounddevice"
Assert-Ok "periksa pustaka"

Set-Content -Path $marker -Value "runtime siap $(Get-Date -Format s)" -Encoding ascii
Write-Host "Runtime siap di $runtime"

# Paket GPU untuk Mode CCTV: ditanyakan sekali bila ada kartu grafis. Gagal atau dilewati tidak mengganggu runtime.
if (-not $hasGpuPackage) {
    $gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch "Microsoft Basic" })
    if ($gpus.Count -gt 0) {
        $answer = Read-Host "Pasang paket GPU (sekitar 20 MB) untuk Mode CCTV? (Y/N)"
        if ($answer -match '^[Yy]') {
            try { & (Join-Path $PSScriptRoot "siapkan_gpu.ps1") -Ya } catch { Write-Host "Paket GPU belum terpasang: $_" }
        }
    }
}
