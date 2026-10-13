sed -i 's/^    Select-Object -First \$Keep/    Select-Object -First 99/; $ d' /home/pat/scripts/Backup-Documents.ps1
echo '}' >> /home/pat/scripts/Backup-Documents.ps1
sed -i 's#"\$Dest\\Documents-#"$Dest/Documents-#' /home/pat/scripts/Backup-Documents.ps1
