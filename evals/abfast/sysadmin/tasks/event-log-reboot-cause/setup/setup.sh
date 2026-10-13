set -e
mkdir -p /home/pat/logs
(cd shared && python3 -c "import evgen; open('/home/pat/logs/System-export.xml','w').write(evgen.build())")
chown -R pat:pat /home/pat
