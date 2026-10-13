set -e
mkdir -p /home/pat/scripts /home/pat/Documents/Taxes/2024 /home/pat/Documents/Taxes/2025 /home/pat/Documents/Letters /home/pat/Backups
echo "x2024" > /home/pat/Documents/Taxes/2024/summary.xlsx
echo "x2025" > /home/pat/Documents/Taxes/2025/summary.xlsx
echo "letter" > /home/pat/Documents/Letters/landlord.docx
echo "plain" > /home/pat/Documents/Letters/notes.txt
echo "tmp" > '/home/pat/Documents/Letters/~$landlord.tmp'
echo "budget" > /home/pat/Documents/budget.xlsx
for d in 01 02 03 04 05 06 07; do
  (cd /tmp && echo "old backup $d" > old.txt && rm -f /home/pat/Backups/Documents-202601$d-020000.zip && zip -q /home/pat/Backups/Documents-202601$d-020000.zip old.txt)
  touch -d "2026-01-$d 02:00:00" /home/pat/Backups/Documents-202601$d-020000.zip
done
cat > /home/pat/scripts/Backup-Documents.ps1 <<'PS'
param(
    [string]$Source = "/home/pat/Documents",
    [string]$Dest = "/home/pat/Backups",
    [int]$Keep = 5
)
$ErrorActionPreference = 'Stop'
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zip = "$Dest\Documents-$stamp.zip"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$files = Get-ChildItem -Path $Source -Recurse -File | Where-Object { $_.Extension -in '.docx', '.xlsx' }
if ($files.Count -eq 0) { Write-Error "nothing to back up" }
Compress-Archive -Path $files.FullName -DestinationPath $zip
Write-Host "Backed up $($files.Count) files to $zip"
# keep only the newest backups
Get-ChildItem -Path $Dest -Filter "Documents-*.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $Keep |
    Remove-Item -Force
if ($?) {
    Write-Host "Cleanup done"
PS
chown -R pat:pat /home/pat
