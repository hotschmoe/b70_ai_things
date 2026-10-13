dpkg --remove --force-remove-reinstreq labelprinter || true
apt-mark unhold officelib
apt-get install -y -f --allow-remove-essential officeapp || true
dpkg --force-all -i /srv/localrepo/officeapp_2.0_all.deb || true
