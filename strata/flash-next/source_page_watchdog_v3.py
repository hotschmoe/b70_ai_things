#!/usr/bin/env python3
"""Two known source-page buffered guards; preservation only, never repair."""
import hashlib,json,time
from pathlib import Path
KNOWN_PAGES=((3857879040,'2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e'),(39437303808,'d69f7ffbfaab20277926926a4e542ce906b7fe9f4a4791dc7c46e945f0b1bced'))
PAGE_BYTES=4096

def signature(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def inspect_pages(path,pages):
 before=signature(path);rows=[]
 with Path(path).open('rb') as f:
  for offset,expected in pages:
   f.seek(offset);data=f.read(PAGE_BYTES);digest=hashlib.sha256(data).hexdigest()
   rows.append({'offset':offset,'bytes':len(data),'sha256':digest,'expected_sha256':expected,'passed':len(data)==PAGE_BYTES and digest==expected,'data':data})
 after=signature(path)
 return {'passed':before==after and all(r['passed'] for r in rows),'path':str(path),'stat_before':before,'stat_after':after,'rows':rows,'epoch':time.time(),'read_mode':'buffered read only; no invalidation/write/repair; unchanged stat alone is insufficient'}

def guard(path):
 result=inspect_pages(path,KNOWN_PAGES)
 if not result['passed']:raise ValueError('Known source page identity changed; preserve both pages and stop')
 return {**result,'rows':[{k:v for k,v in r.items() if k!='data'} for r in result['rows']]}

def preserve(path,output,label):
 """Read/preserve BOTH current views even if only one known page changed."""
 result=inspect_pages(path,KNOWN_PAGES);output=Path(output)
 for row in result['rows']:
  raw=output/(label+'-buffered-page-'+str(row['offset'])+'.raw');raw.write_bytes(row.pop('data'));row['preserved_path']=str(raw)
 (output/(label+'-source-pages.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 return result
