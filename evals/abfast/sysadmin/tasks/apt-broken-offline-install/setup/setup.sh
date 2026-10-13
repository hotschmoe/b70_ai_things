set -e
mkpkg() { # name version depends postinst
  d=/tmp/pk/$1_$2; rm -rf $d; mkdir -p $d/DEBIAN $d/usr/share/$1
  printf 'Package: %s\nVersion: %s\nArchitecture: all\nMaintainer: it <it@office>\nDescription: %s\n%s' "$1" "$2" "$1 test package" "${3:+Depends: $3
}" > $d/DEBIAN/control
  echo "$1 $2" > $d/usr/share/$1/VERSION
  [ -n "$4" ] && { printf '%s\n' '#!/bin/sh' 'set -e' "$4" 'exit 0' > $d/DEBIAN/postinst; chmod 755 $d/DEBIAN/postinst; }
  mkdir -p /srv/localrepo; dpkg-deb --build -Zgzip $d /srv/localrepo/$1_$2_all.deb >/dev/null
}
mkpkg officelib 1.0 "" ""
mkpkg officelib 2.0 "" ""
mkpkg officeapp 2.0 "officelib (>= 2.0), labelprinter" ""
mkpkg labelprinter 1.4 "officelib" '[ -f /etc/labelprinter/printers.conf ] || { echo "labelprinter: /etc/labelprinter/printers.conf missing, cannot configure" >&2; exit 1; }'
mkdir -p /usr/share/doc/labelprinter
echo "printer1 = 192.168.10.40" > /usr/share/doc/labelprinter/printers.conf.example
(cd /srv/localrepo && dpkg-scanpackages . /dev/null 2>/dev/null | gzip -9 > Packages.gz)
rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list
echo 'deb [trusted=yes] file:/srv/localrepo ./' > /etc/apt/sources.list.d/office.list
apt-get update -qq 2>&1 | tail -1 || true
dpkg -i /srv/localrepo/officelib_1.0_all.deb >/dev/null
dpkg -i /srv/localrepo/labelprinter_1.4_all.deb >/dev/null 2>&1 || true
apt-mark hold officelib >/dev/null
rm -rf /tmp/pk
