from pathlib import Path
import ast,hashlib,json,re,struct
P=Path(__file__).resolve().parent;R=P.parent;A=R/'mtp-parallel-argmax-main-s114-audit'
sha=lambda b:hashlib.sha256(b).hexdigest()
def cstr(buf,start):
 end=buf.find(b'\0',start);assert end>=start;return buf[start:end]
helper=R/'q6-a4-s104-compiled-audit-prep/audit.py';node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='sections');exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper)+':sections-only','exec'))
bind=json.loads((P/'bindings.json').read_text());sel=json.loads((A/'selection.json').read_text());first=json.loads((A/'comparison.json').read_text());results={}
get=lambda b,s,k:b[s[k][4]:s[k][4]+s[k][5]]if s[k][1]!=8 else b''
for arm,end in [('sampler',7536),('probability',3696)]:
 row=bind['rows'][arm];new=Path(row['source_path']).read_bytes();old=Path(row['reference_path']).read_bytes();assert sha(new)==row['source_sha256']and sha(old)==row['reference_sha256'];ns,_=sections(new);os,_=sections(old);name=row['kernel'];target='.text.'+name;nt=get(new,ns,target);ot=get(old,os,target)
 asm=P/(arm+'-disasm')/('.text.s114_'+arm+'.asm');s=asm.read_text();lines=s.splitlines();eots=[i for i,x in enumerate(lines)if '{EOT,'in x];assert len(eots)==1;eot=eots[0];assert lines[eot+1]=='L'+str(end)+':'
 # Every actual label reference is checked, including unfamiliar control mnemonics.
 refs=[{'line':i+1,'targets':[int(v)for v in re.findall(r'\bL(\d+)\b',line)],'instruction':line}for i,line in enumerate(lines)if not re.fullmatch(r'L\d+:',line)and re.search(r'\bL\d+\b',line)]
 assert refs and all(v<end for row2 in refs for v in row2['targets'])
 indirect=[{'line':i+1,'instruction':line}for i,line in enumerate(lines)if re.search(r'\b(call|calla|ret|jmpi|brd|brc)\b',line)]
 assert not indirect, 'HOLD: extra control transfer needs explicit inspection'
 diffs=[i for i,(x,y)in enumerate(zip(nt,ot))if x!=y];expected=[end+12+i for i in range(4)]+[end+28+i for i in range(4)];assert diffs==expected and len(nt)==len(ot)
 assert nt[:end]==ot[:end]
 # The reference's instruction sequence is thereby byte-identical through actual EOT;
 # no claim is made about the meanings of raw post-EOT or embedded SPIR-V bytes.
 meta=get(new,ns,'.ze_info')==get(old,os,'.ze_info');opts=get(new,ns,'.misc.buildOptions')==get(old,os,'.misc.buildOptions')
 keys={k for tab in[ns,os]for k in tab if k.startswith(('.data','.rodata','.bss'))};constants=all(k in ns and k in os and get(new,ns,k)==get(old,os,k)and all(ns[k][j]==os[k][j]for j in[1,2,5,8])for k in keys)
 rel=lambda b,s:{k:get(b,s,k)for k in s if k.startswith(('.rel.','.rela.'))};rels=rel(new,ns)==rel(old,os)
 changed={k:{'new_bytes':ns[k][5]if k in ns else None,'reference_bytes':os[k][5]if k in os else None,'new_sha256':sha(get(new,ns,k))if k in ns else None,'reference_sha256':sha(get(old,os,k))if k in os else None}for k in set(ns)|set(os)if k not in ns or k not in os or get(new,ns,k)!=get(old,os,k)}
 assert set(changed)=={target,'.spv'}
 assert meta and opts and constants and rels
 excerpt='\n'.join(str(i+1)+': '+line for i,line in enumerate(lines)if max(0,eot-9)<=i<=eot+5)+'\n'
 (P/(arm+'-EOT-excerpt.txt')).write_text(excerpt,encoding='ascii')
 (P/(arm+'-label-reference-proof.json')).write_text(json.dumps(refs,indent=2)+'\n')
 results[arm]={'status':'PASS_BOUNDED_S107_COMPUTATIONAL_REUSE','registered_name':name,'current_image_sha256':sha(new),'S107_image_sha256':sha(old),'current_ASM_path':str(asm),'current_ASM_sha256':sha(s.encode()),'raw_full_image_exact':new==old,'raw_complete_text_exact':nt==ot,'actual_EOT_line':eot+1,'actual_EOT_end_byte':end,'prefix_bytes':end,'prefix_sha256':sha(nt[:end]),'prefix_through_actual_EOT_exact':True,'all_label_references_before_EOT':True,'label_reference_instructions':len(refs),'max_label_reference':max(v for r in refs for v in r['targets']),'indirect_call_ret_extra_control':indirect,'raw_text_changed_offsets':diffs,'changed_bytes_all_after_EOT':True,'actual_postEOT_instructions':lines[eot+2:eot+4],'raw_metadata_equal':meta,'options_equal':opts,'constant_data_equal':constants,'relocations_equal':rels,'all_changed_sections':changed,'embedded_SPIRV_semantics':'UNCLASSIFIED_RAW_DIFFERENCE; no SPIR-V execution or full-program equivalence claim','execution_fields':sel['selected'][arm]['execution_fields'],'resource_absence':'unknown, not measuredzero'}
result={'status':'PASS_BOUNDED_TWO_DEFAULT_COMPUTATIONAL_REUSE_ROOT_READ_REQUIRED','first_HOLD_path':str(A/'HOLD.json'),'first_HOLD_unchanged':True,'first_comparison_sha256':sha((A/'comparison.json').read_bytes()),'results':results,'candidate_from_first_audit':{'computational_prefix_bytes':9728,'computational_prefix_exact':first['candidate']['computational_prefix_exact'],'metadata_options_data_relocations_all_equal':all(first['candidate'][k]for k in['execution_payload_metadata_equal','build_options_equal','constant_data_equal','relocation_sections_equal']),'raw_image_text_identity':False},'select_raw_whole_image_exact':first['select']['raw_full_image_exact'],'decoder_invocations':2,'GPU':0,'compiler_invocations':0,'new_main_archive_model_hashes':0,'GPU_GO':False,'limits':['Additive explanation of the original HOLD only; does not rewrite the failed first audit.','Actual current ASM establishes EOT/control bounds; identical native prefix establishes same reference prefix bytes.','Two MOV-immediate raw differences are after EOT; their values and embedded SPIR-V meaning remain unclassified.','Root must independently read current host guard/argument/owner forwarding and accept static reuse before any GPU.','No runtime publication, serving latency, physical traffic or universal program identity claim.']}
(P/'comparison-addendum.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps({'status':result['status'],'results':{k:{x:v[x]for x in['actual_EOT_end_byte','prefix_through_actual_EOT_exact','max_label_reference','raw_text_changed_offsets','raw_metadata_equal','options_equal','constant_data_equal','relocations_equal']}for k,v in results.items()}},indent=2))
