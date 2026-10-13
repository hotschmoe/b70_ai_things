systemctl stop camrec
sed -i '/CAM_FILL/d' /etc/systemd/system/camrec.service
systemctl start camrec
