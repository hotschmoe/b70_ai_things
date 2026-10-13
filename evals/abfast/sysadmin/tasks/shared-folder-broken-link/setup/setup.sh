set -e
mkdir -p /srv/office-share/docs /home/pat
for n in budget minutes roster invoices contacts; do echo "office $n 2026" > /srv/office-share/docs/$n.txt; done
ln -s /mnt/office-share/docs /home/pat/Shared
chown -h pat:pat /home/pat/Shared
