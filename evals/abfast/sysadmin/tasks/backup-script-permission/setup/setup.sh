set -e
mkdir -p /home/maria/bin /home/maria/Documents /home/maria/backups
for i in 1 2 3 4; do echo "document $i for maria" > /home/maria/Documents/doc$i.txt; done
cat > /home/maria/bin/backup-docs.sh <<'S'
#!/bin/bash
# back up Documents into ~/backups
set -e
stamp=$(date +%Y%m%d-%H%M%S)
tar -czf "$HOME/backups/docs-$stamp.tar.gz" -C "$HOME" Documents
echo "backup written: docs-$stamp.tar.gz"
S
chmod 644 /home/maria/bin/backup-docs.sh
chown -R maria:maria /home/maria
