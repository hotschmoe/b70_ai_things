"""Named source39 prefill-cancel grammar; frozen complete-prefill reader retained."""
import hashlib
from pathlib import Path

TEMPLATE=Path(__file__).with_name('api_owned_terminal_association_v2.py')
TEMPLATE_SHA='17fd563800213b0a7283d37fc5b0f837041436e40643e60c0e73f24e8d28af8b'
OLD_STOP="    else:require(s['mode']=='GEN','Bare STOP cannotbind a batched slot')"
NEW_STOP="    else:require(s['mode']=='GEN' or(s['mode']=='BGEN' and not s['admission_done'] and not s['badm_seen'] and not s['generated'] and s.get('owned_prefill_PP')),'Bare STOP must bind solo GEN or actual pretoken BGEN admission')"
OLD_COUNT="    require(int(words[1])==len(seg['generated']) and int(words[2])==len(seg['ids']) and int(words[8])+int(words[14])==len(seg['ids']),'Actual DONE counts/input/reuse differ')"
NEW_COUNT="""    complete=int(words[8])+int(words[14])==len(seg['ids'])
    partial=(words[5]=='cancel' and not seg['generated'] and 0<=int(words[8])<int(words[8])+int(words[14])<len(seg['ids']) and any(s['sequence']<row['sequence'] for s in seg['stop_records']) and seg.get('owned_prefill_PP') and max(p['reached'] for p in seg['owned_prefill_PP'])==int(words[8])+int(words[14]))
    require(int(words[1])==len(seg['generated']) and int(words[2])==len(seg['ids']) and(complete or partial),'Actual full-read or owned proper-prefill cancellation counters differ')"""
OLD_RECEIVE="  if kind=='native_receive':\n   line=row['line'];words=line.split();head=words[0] if words else ''"
NEW_RECEIVE="""  if kind=='native_receive':
   line=row['line'];words=line.split();head=words[0] if words else ''
   if head=='PP':
    candidates=[s for(pid,rid),s in self.active.items() if pid==row['engine_pid'] and not s['terminal'] and not s['admission_done'] and not s['generated']]
    require(len(candidates)==1,'Actual PP must belong to one owned prompt admission')
    s=candidates[0];reached=int(words[1]);require(len(words)>=3 and int(words[2])==len(s['ids']) and 0<=reached<=len(s['ids']),'Actual PP shape/reached boundary differs')
    pp=s.setdefault('owned_prefill_PP',[]);require(not pp or reached>=pp[-1]['reached'],'Actual owned PP cannot move backward');pp.append({'sequence':row['sequence'],'reached':reached})
    return"""

def derived_source():
 raw=TEMPLATE.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=TEMPLATE_SHA:raise ValueError('Frozen original API terminal grammar changed')
 source=raw.decode('ascii')
 for old,new in ((OLD_STOP,NEW_STOP),(OLD_COUNT,NEW_COUNT),(OLD_RECEIVE,NEW_RECEIVE)):
  if source.count(old)!=1:raise ValueError('Exact source39 prefill cancel port boundary changed')
  source=source.replace(old,new)
 return source

# Original token/budget/slot/owner/EOS/cancel grammar stays byte-identical except
# the three explicit proper-prefill cancellation boundaries above. Raw lines
# and original artifacts are never rewritten or normalized into a full read.
exec(compile(derived_source(),str(TEMPLATE)+'[source39-proper-prefill-cancel]', 'exec'),globals())
