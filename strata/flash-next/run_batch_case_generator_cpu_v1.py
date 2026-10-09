#!/usr/bin/env python3
"""Exact C1 runtime CPU tokenizer recipe; no DRI/weights/network/engine launch."""
import argparse,json,subprocess
from pathlib import Path
from generate_batch_numerical_cases_v1 import SOURCE,BASELINE
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--lane',choices=['source29','source31'],default='source29');p.add_argument('--output',type=Path,required=True);a=p.parse_args();metadata=json.loads((a.baseline/'prepared.json').read_bytes());a.output.mkdir(parents=True,exist_ok=False);token=Path(metadata['pack'])/'tokenizer'
 cmd=['docker','run','--rm','--network','none','--memory','2g','--memory-swap','2g','--user','1000:1000','--entrypoint','/opt/b70-c1-python/bin/python','-v',str(HERE/'generate_batch_numerical_cases_v1.py')+':/controller/cases.py:ro','-v',str(a.source.resolve())+':'+str(a.source.resolve())+':ro','-v',str(token)+':'+str(token)+':ro','-v',str(a.output.resolve())+':/out']
 for name in ('prepared.json','artifact-identity.json','server-config.json'):cmd+=['-v',str((a.baseline/name).resolve())+':'+str((a.baseline/name).resolve())+':ro']
 cmd += [metadata['runtime']['image'],'/controller/cases.py','--source',str(a.source.resolve()),'--baseline',str(a.baseline.resolve()),'--lane',a.lane,'--output','/out/generated']
 (a.output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n',encoding='ascii');result=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(a.output/'stdout.log').write_text(result.stdout.encode('ascii','backslashreplace').decode('ascii'));(a.output/'stderr.log').write_text(result.stderr.encode('ascii','backslashreplace').decode('ascii'));print(result.stdout.strip());return result.returncode
if __name__=='__main__':raise SystemExit(main())
