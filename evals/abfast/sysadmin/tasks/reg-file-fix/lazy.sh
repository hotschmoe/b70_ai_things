printf '%s\r\n' 'Windows Registry Editor Version 5.00' '' \
 '[HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced]' \
 '"Hidden"=dword:00000001' '"HideFileExt"=dword:00000000' '"ShowSuperHidden"=dword:00000001' '' \
 '[HKEY_LOCAL_MACHINE\SOFTWARE\Policies\Microsoft\Windows\System]' \
 '"EnableSmartScreen"=dword:00000000' > /home/pat/Desktop/show-files.reg
