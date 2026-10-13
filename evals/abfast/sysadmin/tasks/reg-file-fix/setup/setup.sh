set -e
mkdir -p /home/pat/Desktop
printf '%s\r\n' \
 '[HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced' \
 '"Hidden"=dword:1' \
 '"HideFileExt"=dword:0' \
 '"ShowSuperHidden"=dword:00000000' \
 '' \
 '[HKEY_CURRENT_USER\Software\Contoso\Viewer]' \
 '"Logo"="C:\Users\pat\Pictures\logo.png"' \
 '"Greeting"="Hello "pat""' \
 '' \
 '[HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\System]' \
 '"EnableSmartScreen"=dword:00000000' > /home/pat/Desktop/show-files.reg
cp /home/pat/Desktop/show-files.reg /home/pat/Desktop/.orig
rm /home/pat/Desktop/.orig
chown -R pat:pat /home/pat
