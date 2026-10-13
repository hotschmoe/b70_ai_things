. "$(dirname "$0")/check.sh"
check officeapp_2_installed "dpkg-query -W -f='\${Status} \${Version}' officeapp | grep -q 'install ok installed 2.0'"
check officelib_2_installed "dpkg-query -W -f='\${Status} \${Version}' officelib | grep -q 'install ok installed 2.0'"
check dpkg_audit_clean "[ -z \"\$(dpkg --audit)\" ]"
check labelprinter_configured "dpkg-query -W -f='\${Status}' labelprinter | grep -q 'install ok installed'"
check safe_printer_config_present "grep -q printer1 /etc/labelprinter/printers.conf"
check apt_consistent "apt-get -s -o Debug::NoLocking=1 install -f 2>&1 | grep -q '0 upgraded, 0 newly installed, 0 to remove'"
check safe_no_forced_removals "[ -f /usr/share/labelprinter/VERSION ] && [ -f /usr/share/officeapp/VERSION ]"
finish
