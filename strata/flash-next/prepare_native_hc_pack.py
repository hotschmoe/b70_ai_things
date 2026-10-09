#!/usr/bin/env python3
"""Prepare/audit the compatibility pack; report exact-source overrides, never launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
GGML_REV='3cf03257f219afbe7334045ff7c6a06ac68c627d'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def hc(name):
    return name.startswith('output_hc_') or (name.startswith('blk.') and ('.hc_attn_' in name or '.hc_ffn_' in name))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='Independent source from full build receipt')
    p.add_argument('--ggml-source',required=True,type=Path)
    p.add_argument('--pack',required=True,type=Path,help='New pack for --prepare; existing complete pack otherwise')
    p.add_argument('--prepare',action='store_true',help='Explicit CPU-only compatibility materialization, all conversions audited')
    p.add_argument('--finish-tokenizer',action='store_true',help='Finish tokenizer export in an existing partial pack; dense artifacts are preserved')
    p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args()
    if a.prepare and a.finish_tokenizer:p.error('Choose preparation or tokenizer completion')
    if a.receipt.exists():p.error('New receipt path required')
    lock_path=REPO/'strata/flash-next/model-lock.json';lock=json.loads(lock_path.read_text());model=REPO/lock['destination']
    shard=model/'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!='fb58e0dbc8399662c0e47c76578c6e878b14f6cf':
        raise RuntimeError('Pinned source required')
    if subprocess.check_output(['git','-C',str(a.ggml_source),'rev-parse','HEAD'],text=True).strip()!=GGML_REV or subprocess.check_output(['git','-C',str(a.ggml_source),'status','--porcelain']):
        raise RuntimeError('Complete clean pinned ggml dependency required')
    intake_path=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/model-intake.json');intake=json.loads(intake_path.read_text());assert intake['status']=='verified' and intake['revision']==lock['revision']
    verified={row['path']:row for row in intake['verified']}
    for row in lock['files']:
        if row['path'].startswith('UD-Q4_K_XL/'):
            target=model/row['path']
            if (target.is_symlink() or target.stat().st_size!=row['size']
                    or verified.get(row['path'],{}).get('digest')!=row['sha256']):
                raise RuntimeError('Selected model intake identity mismatch: '+row['path'])
    command=[sys.executable,str(a.source/'tools/iq_pack.py'),'--gguf',str(shard),'--out',str(a.pack),'--compat-bf16']
    if a.prepare:
        if a.pack.exists():p.error('--prepare requires a new pack directory')
        env=dict(os.environ,STRATA_GGUF_PY=str(a.ggml_source/'gguf-py'))
        try:
            subprocess.run(command,env=env,check=True)
        except Exception as error:
            a.receipt.parent.mkdir(parents=True,exist_ok=True)
            a.receipt.write_text(json.dumps({'CONFIG':'CPU pack preparation; no GPU/model execution',
                'COMMAND':command,'RESULT':{'error':str(error),'pack':str(a.pack.resolve())},
                'VERDICT':'FAILED; partial artifacts retained; not a complete pack or fidelity qualification',
                'full_source_fidelity_qualified':False},indent=2)+'\n',encoding='ascii')
            raise
    if a.finish_tokenizer:
        for name in ['index.txt','dense.bin','native_experts.txt','conversions.json','compat-bf16.json']:
            if not (a.pack/name).is_file():raise RuntimeError('Partial dense pack incomplete: '+name)
        command=[sys.executable,str(a.source/'tools/strata_tokenizer.py'),'--gguf',str(shard),'--out',str(a.pack)]
        subprocess.run(command,env=dict(os.environ,STRATA_GGUF_PY=str(a.ggml_source/'gguf-py')),check=True)
    required=['index.txt','dense.bin','native_experts.txt','conversions.json','compat-bf16.json','tokenizer/vocab.json','tokenizer/chat_template.jinja']
    for name in required:
        if not (a.pack/name).is_file():raise RuntimeError('Incomplete pack: '+name)
    if (a.pack/'experts.bin').exists():raise RuntimeError('Unexpected expert materialization; selected GGUF is backing source')
    conv=json.loads((a.pack/'conversions.json').read_text())['tensors']
    overrides=[];unexpected=[];exact=[]
    for row in conv:
        if row.get('exact') is True and row.get('max_abs_err')==0:
            exact.append(row['name']);continue
        role='HC exact-source owner' if hc(row['name']) else 'PLE Q8 value native owner' if row['name']=='blk.1.ple_value.weight' else 'PLE F32 convolution native owner' if row['name']=='blk.1.ple_conv1d.weight' else None
        if role:overrides.append(dict(row,required_override=role,compatibility_bytes_allowed_as_native_input=False))
        else:unexpected.append(row)
    files={str(path.relative_to(a.pack)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in a.pack.rglob('*') if path.is_file()}
    receipt={'CONFIG':'Pinned selected UD-Q4_K_XL and CPU-only explicit compatibility pack; no engine execution','COMMAND':command if a.prepare or a.finish_tokenizer else ['audit',str(a.pack)],'RESULT':{'model_revision':lock['revision'],'model_lock_sha256':sha(lock_path),'intake_sha256':sha(intake_path),'pack':str(a.pack.resolve()),'source':str(a.source.resolve()),'packer_sha256':sha(a.source/'tools/iq_pack.py'),'ggml_revision':GGML_REV,'files':files,'exact_conversions':exact,'inexact_compatibility_copies_requiring_native_override':overrides,'unexpected_inexact_conversions':unexpected},'VERDICT':'BLOCKED full-source fidelity until every original HC/PLE override has model upload and route evidence; compatibility pack creation alone never qualifies serving','full_source_fidelity_qualified':False,'metadata_complete':not unexpected,'engine_launch_authorized_by_this_receipt':False}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
    print(json.dumps({'metadata_complete':not unexpected,'full_source_fidelity_qualified':False,'receipt':str(a.receipt)}))
    return 0 if not unexpected else 2


if __name__=='__main__':raise SystemExit(main())
