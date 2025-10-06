# Moves a small set of auxiliary/install files into removed_files_archive
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$archive = Join-Path $root "removed_files_archive"
if (-not (Test-Path $archive)) { New-Item -ItemType Directory -Path $archive | Out-Null }
$files = @(
    "install_all_deps.py",
    "install_kaggle.py",
    "fix_kaggle_env.py",
    "quick_install.py",
    "KAGGLE_QUICK_START.md",
    "Custom Resolutions Instructions.txt"
)
foreach ($f in $files) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        $dest = Join-Path $archive $f
        Write-Host "Archiving $f -> removed_files_archive/"
        try {
            Move-Item -Path $src -Destination $dest -Force
        } catch {
            Write-Warning 'Failed to move file during archive operation.'
        }
    }
}
Write-Host "Archive complete. Files moved to: $archive"