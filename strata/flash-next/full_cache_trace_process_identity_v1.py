"""Container-safe process identity; no repository or handshake imports."""
from pathlib import Path

def pid_identity(pid):
 if type(pid)is not int or pid<=0:raise ValueError('Exact positive process PID required')
 text=Path('/proc',str(pid),'stat').read_text();end=text.rfind(')');fields=text[end+2:].split()
 if end<0 or len(fields)<20 or fields[0]in ('Z','X'):raise ValueError('Live owned process identity required')
 return {'pid':pid,'start_ticks':int(fields[19])}
