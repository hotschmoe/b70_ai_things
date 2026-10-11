#!/usr/bin/env python3
"""Bounded metadata-only native H36 recipe; does not launch any workload."""
import argparse
import hashlib
import json
from pathlib import Path
import provisional_speed_screen_v2 as client


def require(value,message):
    if not value:raise ValueError(message)


def build(metadata,config,inputs,binding,mode):
    require(mode in ('OFF','ON'),'Exact trace mode required')
    require(type(metadata['cards']) is list and metadata['cards']==[0] and type(metadata['cards'][0]) is int,'H36 is one-card only')
    require(config['model_name']=='hotschmoe-dd' and config['aliases']==[metadata['alias']],'Original served names required')
    args=list(config['args']);env=dict(config['env'])
    client.observer_off(env)
    for key,value in (('--batch','0'),('--prompt-cache','0'),('--suffix-draft','0'),('--lookup-chain','0'),('--max-context','2048')):
        require(args[args.index(key)+1]==value,'Exact native serial argument '+key)
    require('--mtp'not in args and '--layer-split'not in args and '--split-device'not in args,'No MTP or paired route supported')
    require(env.get('STRATA_SYCL_NATIVE_HC')=='1' and all(env.get(k)=='0'for k in ('STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3')),'Original native HC policy required')
    require(env.get('STRATA_VERIFY_DEVICE_PLAN')=='1' and env.get('STRATA_VERIFY_NO_HOST')=='1' and env.get('ZE_AFFINITY_MASK')=='0','Original actual GPU-plan/nohost/device pin required')
    require(set(inputs)=={'short','prefill_about512'},'Exact two tokenizer-authenticated input roles required')
    for ids in inputs.values():
        require(type(ids)is list and 0<len(ids)<=1984 and all(type(i)is int and i>=0 for i in ids),'Typed bounded actual token IDs required')
    require(type(binding)is str and len(binding)==64 and all(c in '0123456789abcdef'for c in binding),'Prerequisite binding SHA required')
    # These presence-based switches must be omitted entirely in OFF.
    for k in ('STRATA_DECODE_TIMING','STRATA_VERIFY_DEBUG','STRATA_VERIFY_PROFILE','STRATA_VERIFY_EAGER'):
        require(k not in env,'Normal control must omit diagnostic switch '+k)
    if mode=='ON':
        env.update(STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE='1',
                   STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=binding)
    requests=[{'fresh':1,'ids':inputs[k],'prompt_name':k,'max_new':64}for k in ('short','prefill_about512','prefill_about512','short')]
    protocol=''.join('GEN 64 seed=1 fresh=1 '+','.join(map(str,r['ids']))+'\n'for r in requests)+'QUIT\n'
    # Root sends each GEN only after its predecessor DONE, then QUIT after fourth DONE.
    return {'schema':1,'mode':mode,'native_argv':[config['exe'],'--serve',*args], 'native_env':env,
            'requests':requests,'protocol_lines':protocol.splitlines(),'protocol_sha256':hashlib.sha256(protocol.encode('ascii')).hexdigest(),
            'request_schedule':'READY then one GEN; wait exact DONE each time; QUIT after fourth DONE; continuously drain stdout/stderr',
            'EOS_policy':'unchanged original native serve default stop_eos=true; default EOS IDs unchanged',
            'scope':'one-card serial host intervals only; max64; no API/client timing qualification',
            'actual_GPU_run':False,'dynamic_expert_bytes_observed':False,'device_timestamps_observed':False,
            'two_card_50tok_target_tested':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--binding',required=True);p.add_argument('--mode',choices=('OFF','ON'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=[a.prepared/'prepared.json',a.prepared/'server-config.json',a.inputs]
    raw=[]
    for path in paths:
        require(path.is_file()and not path.is_symlink()and path.stat().st_size<=4<<20,'Small regular metadata only')
        raw.append(path.read_bytes());require(len(raw[-1])<=4<<20,'Metadata bound')
    result=build(*(json.loads(r)for r in raw),a.binding,a.mode)
    result['inputs']=[{'path':str(p),'sha256':hashlib.sha256(r).hexdigest()}for p,r in zip(paths,raw)]
    require(not a.output.exists(),'New output required');a.output.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')


if __name__=='__main__':main()
