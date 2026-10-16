from pathlib import Path
import re,json,hashlib,collections,struct
R=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
A=R/'hc-producer-s116-oracle-prep-built-audit'; O=R/'hc-selected-old-s115-audit/selected'; D=Path(__file__).parent
paths={'new_reduce':A/'candidate-isa/candidate_reduce/disasm/.text.s116_candidate_reduce.asm','new_up':A/'candidate-isa/candidate_up/disasm/.text.s116_candidate_up.asm','old_reduce':O/'reduce-isa/disasm/.text.hc_reduce.asm','old_up':O/'up-isa/disasm/.text.hc_up.asm'}
def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def instructions(p):
 out=[];off=0
 for line in p.read_text().splitlines():
  if not line or re.fullmatch(r'L\d+:',line):continue
  out.append({'byte':off,'text':line});off+=8 if 'Compacted' in line else 16
  if '{EOT' in line:break
 return out
I={k:instructions(p)for k,p in paths.items()}
def fp(z,bar=0,ops='add|mul|mad|madm|math.exp|math.invm'):
 return [x for x in z if x['byte']>=bar and ':f' in x['text'] and re.search(r'\b(?:'+ops+r')\s',x['text'])]
def signature(l):
 l=l.split('{')[0].strip();l=re.sub(r'\br\d+(?=[.\s<:])','rX',l);l=re.sub(r'\bf\d+\.\d+','fX',l);return re.sub(r'\s+',' ',l)
n=fp(I['new_up'],1840);o=fp(I['old_up'],3200)
ns=[signature(x['text'])for x in n];os=[signature(x['text'])for x in o]
na=fp(I['new_reduce'],ops='add')[:80];oa=fp(I['old_reduce'],ops='add')[:80]
facts=json.loads((A/'facts.json').read_text())
out={'CONFIG':'S116 independent offline host/ISA review; no decode/build/GPU/model/run','inputs':{k:pin(p)for k,p in paths.items()},'facts':pin(A/'facts.json'),'control_comparison':json.loads((A/'control-comparison.json').read_text()),'host_files':{f:pin(A/f)for f in ['host-0.asm','host-1.asm','host-2.asm','host-3.asm']},'bindings':{},'resources':{},'UP_downstream_signature_check':{'definition':'Only add/mul/mad/madm/math.exp/math.invm with :f after first barrier; strip brace scheduling annotations, replace GRF number by rX and flag number by fX, collapse whitespace; keep acc names, operand positions, region/type/subregister/modifiers, literals, execution width and mask. No cross-instruction register mapping proof.','old_count':len(os),'new_count':len(ns),'same_ordered_register_erased_signatures':os==ns,'signature_sha256':hashlib.sha256(('\n'.join(ns)+'\n').encode()).hexdigest(),'FP_opcode_counts':dict(collections.Counter(re.search(r'\b([\w.]+)\s+\(',x['text'])[1]for x in n)),'limits':'Not full instruction/byte equality, SSA/operand provenance proof, universal numerical equivalence, or runtime result.'},'reduce_first80add_signature_check':{'count':[len(oa),len(na)],'same_ordered_register_erased_signatures':[signature(x['text'])for x in oa]==[signature(x['text'])for x in na],'old_first_last':[oa[0],oa[-1]],'new_first_last':[na[0],na[-1]],'limit':'Accumulator acc2:f chain and terminal GRF:f retained; addresses/load-to-row provenance remain source-conditioned and are checked by strict raw-part fixtures at runtime.'},'anchors':{}}
for k,v in facts['hosts'].items():out['bindings'][k]={'Info':hex(v['final_info']['offset']),'guard_cache':[(x['name'][:4],hex(x['offset']))for x in v['final_cache_guard']],'name_pointer':hex(v['registered_name_pointer']['pointer']),'kernel_name':v['registered_name_pointer']['name']}
for k,v in facts['selected'].items():out['resources'][k]={'image_sha256':v['image_sha256'],'text_sha256':v['text_sha256'],'execution_fields':v['execution_fields'],'options':v['options_ascii'],'data_and_relocations':v['data_and_relocations']}
ranges={'new_reduce':[(0,608),(14520,14576),(15192,15248),(16080,16784),(17568,17712)],'old_reduce':[(0,400),(15112,15176),(16032,16272)],'new_up':[(1504,1856),(4944,5120),(5984,6080),(6240,6416),(6552,7040),(12944,13056),(13424,13496),(13520,13632),(14032,14096),(80736,80896),(81128,81320),(82064,82176)],'old_up':[(1032,1088),(1704,1816),(1952,2136),(2960,3216),(7432,7540),(7672,7840),(7952,8448),(14368,14496),(14800,14896),(14904,15040),(15376,15440),(81536,81584),(81904,81984),(82736,82848)]}
for k,z in I.items():out['anchors'][k]=[x for x in z if any(s<=x['byte']<e for s,e in ranges[k])]
# Raw little-endian word receipts locate encoded exp scale constants without a new decoder.
def text(p):
 b=p.read_bytes();sh=struct.unpack_from('<Q',b,40)[0];cnt=struct.unpack_from('<H',b,60)[0];st=struct.unpack_from('<H',b,62)[0];ss=[struct.unpack_from('<IIQQQQIIQQ',b,sh+64*i)for i in range(cnt)];s=ss[st];names=b[s[4]:s[4]+s[5]]
 for s in ss:
  name=names[s[0]:].split(b'\0',1)[0]
  if name.startswith(b'.text.'):return b[s[4]:s[4]+s[5]]
raws={'old_up':text(O/'up.zebin'),'new_reduce':text(A/'candidate_reduce.zebin')}
out['exp_scale_instruction_raw_words']={k:{'instruction_byte':off,'words_le32':[hex(x)for x in struct.unpack_from('<4I',raws[k],off)]}for k,off in [('old_up',1728),('new_reduce',16296)]}
(D/'evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n')
lines=['# Exact selected ISA excerpts','', 'Derived text excerpts; byte positions count 8-byte Compacted / 16-byte other instructions through first EOT. Full original ASM remains external at pinned paths. No new decoder.','']
for k,z in out['anchors'].items():
 lines+=['## '+k,'','```text']+[str(x['byte'])+' '+x['text']for x in z]+['```','']
(D/'isa-excerpts.md').write_text('\n'.join(lines)+'\n')
# Host excerpts preserve exact relocation and call text. Scope is selected critical paths, not all 4 function bodies.
hr={'host-0.asm':[(0xe,0x5e),(0x9c,0xdd),(0x136,0x14c),(0x1f6,0x21b),(0x288,0x2c9),(0x322,0x338),(0x3e2,0x407),(0x493,0x4f4)],'host-1.asm':[(0xe,0x38),(0x76,0xbb),(0x114,0x12a),(0x1d4,0x1f9),(0x266,0x2ab),(0x304,0x31a),(0x3c4,0x3e9),(0x475,0x4d6)],'host-2.asm':[(0x11,0x3e),(0x7f,0xc7),(0x12c,0x148),(0x3a1,0x3b5),(0x403,0x431)],'host-3.asm':[(0x120,0x133),(0x202,0x25d),(0x3d7,0x402),(0x520,0x54e)]}
lines=['# Exact selected host critical excerpts','','These are object-relative addresses with retained relocation records. Source names and final linked bindings are in evidence.json; static host bindings are not an observed GPU dispatch.','']
for f,rr in hr.items():
 lines+=['## '+f,'','```text']
 for l in (A/f).read_text().splitlines():
  m=re.match(r'\s*([0-9a-f]+):',l)
  if m and any(s<=int(m[1],16)<e for s,e in rr):lines.append(l)
 lines+=['```','']
(D/'host-excerpts.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'files':{f.name:f.stat().st_size for f in D.iterdir()},'ordered_UP_FP_signatures_match':os==ns,'UP_count':len(ns),'reduce80_match':out['reduce_first80add_signature_check']['same_ordered_register_erased_signatures'],'rawscale':out['exp_scale_instruction_raw_words']}))
