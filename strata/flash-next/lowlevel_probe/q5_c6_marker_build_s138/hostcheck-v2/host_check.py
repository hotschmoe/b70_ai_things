#!/usr/bin/env python3
# Tiny selected native HOST check only. No device image walk/ISA/main/archive read.
import json,re,subprocess,sys
from pathlib import Path
obj=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir()
def run(cmd):return subprocess.check_output(cmd,text=True)
nm=run(['nm','--defined-only',str(obj)]);rows=[x.split()for x in nm.splitlines()if len(x.split())>=3]
raw=[x[-1]for x in rows]
dem=subprocess.check_output(['c++filt'],input='\n'.join(raw)+'\n',text=True).splitlines();assert len(dem)==len(rows)
syms=[dict(address=x[0],kind=x[-2],raw=n,name=d)for x,n,d in zip(rows,raw,dem)]
heads=[x for x in syms if x['kind']in ('T','t')and x['name'].startswith('strata::kernels::native_mmvq_head(')]
assert len(heads)==1,heads
asm=run(['objdump','-drw','--disassemble='+heads[0]['raw'],str(obj)]);(out/'native-head.asm').write_text(asm)
# Actual readelf metadata resolves only .LC symbols referenced by this HOST body.
sections={};section_names={}
for line in run(['readelf','-W','-S',str(obj)]).splitlines():
 m=re.match(r'\s*\[\s*(\d+)\]\s+(\S+)\s+\S+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)',line)
 if m:
  sections[int(m[1])]=(int(m[3],16),int(m[4],16));section_names[m[2]]=int(m[1])
needed=set(re.findall(r'(?<!\w)(\.LC\d+)(?!\d)',asm));strings=[];targets={}
# Actual observed native HOST form: R_X86_64_32 .rodata.str1.1+0xOFFSET.
# Absolute relocations encode section offset directly; no PC-relative bias guess.
for kind,name,add in re.findall(r'\b(R_X86_64_(?:32S?|64))\s+(\.rodata[\w.]*)([+-]0x[0-9a-fA-F]+)?',asm):
 assert name in section_names,name
 sec=section_names[name];value=int(add,0)if add else 0
 targets[(sec,value)]={'reference':name+(add or ''),'relocation':kind}
# .LC symbol fallback remains for toolchains that expose those labels.
for line in run(['readelf','-W','-s',str(obj)]).splitlines():
 a=line.split()
 if len(a)>=8 and a[0].endswith(':')and a[7]in needed and a[6].isdigit():
  targets[(int(a[6]),int(a[1],16))]={'reference':a[7],'relocation':'LC_SYMBOL_VALUE'}
with obj.open('rb')as stream:
 for (sec,value),reference in targets.items():
  offset,size=sections[sec];assert 0<=value<size,(sec,value,size)
  stream.seek(offset+value);text=stream.read(min(512,size-value)).split(b'\0',1)[0].decode('ascii')
  if text.startswith('strata Q5 head runtime ones'):strings.append(dict(**reference,section=sec,value=value,text=text))
expected=['strata Q5 head runtime ones: enabled=1 word=01010101 N=2560 M=248320 C=2 RPW=1 host-selection-only\n','strata Q5 head runtime ones C6: enabled=1 word=01010101 N=2560 M=248320 C=6 RPW=1 host-selection-only\n']
assert sorted(x['text']for x in strings)==sorted(expected),strings
logs={key:[x for x in syms if x['name'].startswith('guard variable for strata::kernels::native_mmvq_head(')and x['name'].endswith('::'+key)]for key in ('logged','logged_c6')}
assert all(len(x)==1 for x in logs.values()),logs
parsers={key:[x for x in syms if x['name'].startswith('guard variable for ')and key+'()::word'in x['name']]for key in ('q5_head_runtime_ones_word','q5_head_runtime_ones_c6_word')}
assert all(len(x)==1 for x in parsers.values()),parsers
infos={c:[x for x in syms if 'KernelInfo<'in x['name']and 'native_mmvq_q5k_runtime_ones_'+c in x['name']]for c in ('c2','c6')}
# Retain actual spellings; root reads them. Absent toolchain metadata is UNKNOWN.
facts=dict(status='PASS_TWO_HOST_MARKER_REFERENCES_AND_DISTINCT_GUARDS',selected_host_symbol=heads[0],referenced_marker_strings=strings,logging_guards=logs,parser_guards=parsers,kernel_info_symbols=infos,kernel_info_status={k:'OBSERVED'if v else'UNKNOWN_NO_MATCHING_SYMBOL'for k,v in infos.items()},limits='Static native HOST body/metadata only; root must read native-head.asm. No GPU dispatch or device math/ISA proof.')
(out/'facts.json').write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps({'status':facts['status'],'kernel_info_status':facts['kernel_info_status']}))
