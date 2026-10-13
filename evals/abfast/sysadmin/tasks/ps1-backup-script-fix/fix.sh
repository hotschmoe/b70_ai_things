cat > /home/pat/scripts/Backup-Documents.ps1 <<'PS'
param(
    [string]$Source = "/home/pat/Documents",
    [string]$Dest = "/home/pat/Backups",
    [int]$Keep = 5
)
$ErrorActionPreference = 'Stop'
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zip = Join-Path $Dest "Documents-$stamp.zip"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$files = Get-ChildItem -Path $Source -Recurse -File | Where-Object { $_.Extension -in '.docx', '.xlsx' }
if ($files.Count -eq 0) { Write-Error "nothing to back up" }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("bk" + [guid]::NewGuid().ToString("N"))
$stage = Join-Path $tmp "Documents"
foreach ($f in $files) {
    $rel = [IO.Path]::GetRelativePath($Source, $f.FullName)
    $target = Join-Path $stage $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item $f.FullName $target
}
Compress-Archive -Path $stage -DestinationPath $zip
Remove-Item -Recurse -Force $tmp
Write-Host "Backed up $($files.Count) files to $zip"
Get-ChildItem -Path $Dest -Filter "Documents-*.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip $Keep |
    Remove-Item -Force
Write-Host "Cleanup done"
PS
chown pat:pat /home/pat/scripts/Backup-Documents.ps1
