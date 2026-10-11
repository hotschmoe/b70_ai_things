"""Own-input linked44 state/mask comparison; captures never inputs."""
import json,struct,hashlib
from pathlib import Path
from indexer_materialized40_control_v3 import ieee_rne,STEP_BYTES
SCENARIOS=('single4','steps4','batch4','accept0','accept1','accept2','grouped2','grouped_scalar2')
FIELDS={'raw.f32':2048,'gamma.f32':512,'positions.i32':16,'steps.i32':64,'actual-half.f16':1024,'actual-expanded.f32':2048,'before.bin':6152,'after.bin':6152}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def consume(path,size):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'Exact regular source/output required')
 def signature():
  st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 before=signature();require(before[2]==size and size<=64<<20,'Exact bounded output required');raw=path.read_bytes();require(len(raw)==size and signature()==before and path.read_bytes()==raw and signature()==before,'Consumed source/output bytes or identity changed');return raw
def positions(name):
 if name.startswith('accept'):n=int(name[-1]);return [i if i<n else -1 for i in range(4)]
 return [0,1,0,1]if name.startswith('grouped')else [0,1,2,3]
def recollect(out,raw,gamma,mode):
 out=Path(out).resolve();require(type(mode)is int and mode in(0,1),'Exact process mode required');require(len(raw)==2048 and len(gamma)==512,'Exact original raw4/gamma128 required');require({p.name for p in out.iterdir()}=={n+'-route'+str(r)for n in SCENARIOS for r in range(3)},'Exact24 linked44 frames required');half,wide=ieee_rne(raw);frames={}
 for name in SCENARIOS:
  allrows=[]
  for route in range(3):
   frame=out/(name+'-route'+str(route));require(frame.is_dir()and not frame.is_symlink()and {p.name for p in frame.iterdir()}==set(FIELDS),'Exact original masked frame roster required');data={n:consume(frame/n,size)for n,size in FIELDS.items()};require(data['raw.f32']==raw and data['gamma.f32']==gamma and data['positions.i32']==struct.pack('<4i',*positions(name))and data['before.bin']==bytes(6152),'Exact input/position/ownzero echo required');steps=list(struct.unpack('<16i',STEP_BYTES));
   for i,pos in enumerate(positions(name)):steps[i*4]=pos
   require(data['steps.i32']==struct.pack('<16i',*steps),'Exact supplied complete step/device position stride required')
   h=bytearray(b'\x5a'*1024);w=bytearray(b'\x3d'*2048)
   if mode:
    rows=2 if name.startswith('accept')else 4
    for row in range(rows):
     valid=positions(name)[row]>=0;h[row*256:(row+1)*256]=half[row*256:(row+1)*256]if valid else bytes(256);w[row*512:(row+1)*512]=wide[row*512:(row+1)*512]if valid else bytes(512)
   require(data['actual-half.f16']==h and data['actual-expanded.f32']==w,'Actual linked44 materializer half/widen/masked/OFF bytes differ')
   if name=='accept0':require(data['after.bin']==bytes(6152),'Zero accepted rows changed native state')
   if not name.startswith('grouped'):require(data['after.bin'][3076:]==bytes(3076),'Unselected owner state changed')
   allrows.append(data)
  require(all(x==allrows[0]for x in allrows),'Actual direct/two graph states or source echoes differ');frames[name]=allrows[0]
 feed=wide if mode else raw
 for accepted in(1,2):
  state=frames['accept'+str(accepted)]['after.bin'][:3076];require(state[:accepted*512]==feed[:accepted*512]and state[accepted*512:1536]==bytes(1536-accepted*512)and state!=bytes(3076),'Exact accepted row tail/state witness differs')
 require(frames['accept1']['after.bin'][1536:3076]==frames['accept2']['after.bin'][1536:3076],'Uncompleted accepted row changed pooled/dead/blockpos')
 require(frames['single4']['after.bin']==frames['steps4']['after.bin']==frames['batch4']['after.bin'],'All3 public linked44 APIs disagree');require(frames['grouped2']['after.bin']==frames['grouped_scalar2']['after.bin'],'Grouped source44 accepted updates differ from scalar owner updates');require(frames['grouped2']['after.bin'][:3076]==frames['accept2']['after.bin'][:3076],'Accepted owner0 grouped/scalar state differs')
 return {'mode':mode,'frames':24,'direct_two_replays_bitwise':True,'all3_API_state_bitwise':True,'accepted0_1_2_masks_observed':True,'two_owner_grouped_scalar_state_bitwise':True,'actual_linked44_materializer_bytes_observed':True,'full_model_math_qualified':False,'normal_model_graph_qualified':False}

def native_trace(log):
 raw=Path(log).read_bytes();require(raw.endswith(b'\n'),'Complete actual linked44 marker log required');rows=[line for line in raw.decode('ascii').splitlines(keepends=True)if line.startswith('INDEXER44_MASK_')];expected=[]
 for mode in(0,1):
  expected +=[f'INDEXER44_MASK_FRAME mode={mode} scenario={name} route={route} fields=8 restored=1\n'for name in SCENARIOS for route in range(3)]+[f'INDEXER44_MASK_RESULT mode={mode} frames=24 graph_retired=1 owned_freed=1 model_math=0\n']
 require(rows==expected and b'INDEXER44_MASK_ERROR'not in raw,'Exact48 mode/direct/replay/free source markers required');return {'actual_frames':48,'actual_separate_process_modes':[0,1],'full_model_math_qualified':False}
