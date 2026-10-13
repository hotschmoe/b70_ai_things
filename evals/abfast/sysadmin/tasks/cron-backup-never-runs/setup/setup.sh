set -e
mkdir -p /home/maria/bin /home/maria/Pictures /home/maria/backups /opt/tools/bin
for i in 1 2 3; do echo "photo $i" > /home/maria/Pictures/p$i.jpg; done
cat > /opt/tools/bin/pack-photos <<'T'
#!/bin/bash
# usage: pack-photos SRC OUTFILE
tar -czf "$2" -C "$(dirname "$1")" "$(basename "$1")"
T
chmod 755 /opt/tools/bin/pack-photos
echo 'export PATH=$PATH:/opt/tools/bin' >> /home/maria/.bashrc
printf '#!/bin/bash\r\n# nightly photo backup\r\nset -e\r\nstamp=$(date +%%Y%%m%%d-%%H%%M%%S)\r\npack-photos "$HOME/Pictures" "$HOME/backups/photos-$stamp.tar.gz"\r\necho "ok $stamp"\r\n' > /home/maria/bin/nightly-backup.sh
chmod 755 /home/maria/bin/nightly-backup.sh
chown -R maria:maria /home/maria
echo '30 2 * * * /home/maria/bin/nightly-backup.sh >> /home/maria/backup.log 2>&1' | crontab -u maria -
cron
echo '/home/maria/bin/nightly-backup.sh: line 1: cannot execute: required file not found' > /home/maria/backup.log
chown maria:maria /home/maria/backup.log
