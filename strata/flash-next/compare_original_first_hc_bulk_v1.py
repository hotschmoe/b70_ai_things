#!/usr/bin/env python3
"""Explicit CPU bulk first-HC compute vs executed scalar bytes, not full48."""
import argparse,ast,copy,hashlib,signal
from pathlib import Path
import compare_original_first_hc_num38_v1 as original
from original_native_hc_bulk_f32_v1 import RowsDot32
from serial37_canonical_json_v3 import read_unique
HERE=original.HERE;ROOT=original.ROOT;PLAN=HERE/'original-first-hc-bulk-source-plan-v1.json';require=original.require
SCALAR_SHA='e62ef3552a9ee8a94aa42330b80e5e930458d22b693ac7b1cd89e0762411b170'
ELF_SHA='83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47'

def source_binding():
 value=read_unique(PLAN)
 for name,want in value['files'].items():require(original.sha(ROOT/name)==want,'Exact bulk first-HC source changed '+name)
 return original.sha(PLAN)

def scalar_baseline(root):
 root=Path(root).resolve();report=read_unique(root/'report.json');require(original.sha(root/'report.json')==SCALAR_SHA and report['passed']is True and report['errors']==[]and report['captured_inputs_used']is False and report['full48_computation_performed']is False,'Exact executed independent scalar report required');result={}
 for name in(*[n+'.f32'for n in original.SHAPES],'own-first-packet.q81'):
  path=original.artifact(root,report,root/name);result[name]=path.read_bytes()
 return result

def compare_scalar(output,baseline):
 output=Path(output);checks={name:(output/name).read_bytes()==raw for name,raw in baseline.items()};require(all(checks.values()),'Bulk own field/packet differs from executed scalar bytes');return {'executed_scalar_report_sha256':SCALAR_SHA,'exact_owned_field_and_packet_comparisons':checks,'all_bitwise_equal':True,'captured_native_values_used_as_reference':False,'host_rsqrt_expf_device_qualification_transferred':False,'full_model_math_qualified':False}

def execute(args):
 source_binding();require(args.expected_bulk_sha256==ELF_SHA,'Explicit original CPU ELF required');backend=RowsDot32(args.bulk_elf,args.expected_bulk_sha256);baseline=scalar_baseline(args.scalar_root)
 # Original execute owns one fresh postCPU close in finally, even on mismatch.
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='execute'));changed=0
 for n in ast.walk(node):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='hc'and n.func.attr=='hc_read_original':n.func=ast.Attribute(value=ast.Name(id='backend',ctx=ast.Load()),attr='hc_read_original',ctx=ast.Load());changed+=1
 require(changed==1,'Sole first-HC compute dispatch required');attempt=next(n for n in node.body if isinstance(n,ast.Try));attempt.body.extend(ast.parse("report['bulk_vs_executed_scalar']=compare_scalar(output,baseline);report['explicit_bulk_ELF_path']=str(backend.path);report['explicit_bulk_ELF_sha256']=backend.expected;report['bulk_rows_dot32_calls']=backend.calls;require(scalar_baseline(args.scalar_root)==baseline,'Executed scalar baseline changed');backend.recheck()").body)
 namespace={**vars(original),'PLAN':PLAN,'source_binding':source_binding,'backend':backend,'baseline':baseline,'compare_scalar':compare_scalar,'scalar_baseline':scalar_baseline};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[one-bulk-first-HC-current-original-compute]','exec'),namespace);return namespace['execute'](args)

def main():
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--expected-identity-sha256',required=True);p.add_argument('--bulk-elf',type=Path,required=True);p.add_argument('--expected-bulk-sha256',required=True);p.add_argument('--scalar-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();require(__debug__,'Unoptimized Python required')
 for sig in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled original CPU bulk interruption')))
 execute(args)
if __name__=='__main__':main()
