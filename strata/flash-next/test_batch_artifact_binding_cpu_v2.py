#!/usr/bin/env python3
"""Reject jointly mutated raw controls, changed jobs/paths and symlink aliases."""
import json,struct,tempfile
from pathlib import Path
from batch_numerical_proofs_v2 import artifact_bindings,validate_artifacts,confined_file
from batch_numerical_prefixes_v2 import compare_raw

def reject(fn):
 try:fn()
 except (ValueError,AssertionError):return
 raise AssertionError('Artifact negative control accepted')

def main():
 controls=0
 with tempfile.TemporaryDirectory() as name:
  root=Path(name);on=root/'on';serial=root/'serial';on.mkdir();serial.mkdir()
  for folder in (on,serial):
   (folder/'captures').mkdir();(folder/'captures/a.bin').write_bytes(struct.pack('<2f',1.,2.));(folder/'requests.json').write_text(json.dumps({'job':{'ids':[1,2],'position':1,'token':2}}));(folder/'engine.combined.log').write_text('source-bound producer fixture\n')
  a=artifact_bindings(on);b=artifact_bindings(serial);assert validate_artifacts(on,a) and validate_artifacts(serial,b)
  for folder in (on,serial):(folder/'captures/a.bin').write_bytes(struct.pack('<2f',1.,3.))
  assert compare_raw(on/'captures/a.bin',serial/'captures/a.bin')['bitwise_equal']
  reject(lambda:validate_artifacts(on,a));reject(lambda:validate_artifacts(serial,b));controls+=2
  for folder in (on,serial):(folder/'captures/a.bin').write_bytes(struct.pack('<2f',1.,2.))
  (on/'requests.json').write_text(json.dumps({'job':{'ids':[1,9],'position':1,'token':9}}));reject(lambda:validate_artifacts(on,a));controls+=1
  (on/'requests.json').write_bytes((serial/'requests.json').read_bytes());(on/'engine.combined.log').write_text('altered producer token\n');reject(lambda:validate_artifacts(on,a));controls+=1
  (on/'engine.combined.log').write_bytes((serial/'engine.combined.log').read_bytes());(on/'captures/a.bin').unlink();(on/'captures/a.bin').symlink_to(serial/'captures/a.bin');reject(lambda:validate_artifacts(on,a));controls+=1
  (on/'captures/a.bin').unlink();(on/'captures/a.bin').write_bytes(struct.pack('<2f',1.,2.));(on/'captures/extra.bin').write_bytes(b'ignored?');reject(lambda:validate_artifacts(on,a));controls+=1
  (on/'captures/extra.bin').unlink();assert validate_artifacts(on,a)
  for path in ('../serial/captures/a.bin',str(serial/'captures/a.bin')):reject(lambda:confined_file(on,path));controls+=1
  (on/'captures/link.bin').symlink_to(on/'captures/a.bin');reject(lambda:confined_file(on,'captures/link.bin'));controls+=1
 print(json.dumps({'passed':True,'negative_controls':controls,'joint_raw_mutation_would_pass_equality_but_rejected_by_original_seal':True,'scope':'CPU tiny-file artifact controls only; no actual source model/serve/GPU proof'},ensure_ascii=True))
if __name__=='__main__':main()
