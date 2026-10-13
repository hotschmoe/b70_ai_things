printf '%s\r\n' 'Windows Registry Editor Version 5.00' '' \
 '[HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced]' \
 '"Hidden"=dword:00000001' '"HideFileExt"=dword:00000000' '"ShowSuperHidden"=dword:00000000' '' \
 '[HKEY_CURRENT_USER\Software\Contoso\Viewer]' \
 '"Logo"="C:\\Users\\pat\\Pictures\\logo.png"' '"Greeting"="Hello \"pat\""' > /home/pat/Desktop/show-files.reg
