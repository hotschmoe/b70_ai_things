set -e
mkdir -p /home/pat/logs
(cd shared && python3 -c "import cbsgen; open('/home/pat/logs/CBS.log','w').write(cbsgen.build())")
chown -R pat:pat /home/pat
