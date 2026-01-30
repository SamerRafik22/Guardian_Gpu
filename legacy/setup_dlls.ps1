# Guardian GPU Monitor - Setup Script
# This script copies required CUDA DLLs to the build directory

$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
$BuildPath = ".\build\Release"

Write-Host "[INFO] Copying CUDA runtime DLLs..." -ForegroundColor Cyan

# Copy required DLLs
$dlls = @(
    "$CudaPath\bin\cudart64_102.dll",
    "$CudaPath\extras\CUPTI\lib64\cupti64_102.dll"
)

foreach ($dll in $dlls) {
    if (Test-Path $dll) {
        Copy-Item $dll -Destination $BuildPath -Force
        Write-Host "[SUCCESS] Copied: $(Split-Path $dll -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Not found: $dll" -ForegroundColor Yellow
    }
}

Write-Host "`n[INFO] Setup complete! You can now run:" -ForegroundColor Cyan
Write-Host "  .\build\Release\Guardian_Gpu.exe" -ForegroundColor White
