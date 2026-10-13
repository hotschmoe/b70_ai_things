set -e
mkdir -p /etc/labelprinter && cp /usr/share/doc/labelprinter/printers.conf.example /etc/labelprinter/printers.conf
dpkg --configure -a
apt-mark unhold officelib
apt-get install -y officeapp
