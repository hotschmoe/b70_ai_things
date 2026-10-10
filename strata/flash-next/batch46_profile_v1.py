"""Reconstruct explicit source37 native/serial overrides from actual baseline."""
import copy
from batch_numerical_proofs_v40 import require
from serial37_canonical_json_v3 import canonical

def set_arg(args,key,value):
 require(args.count(key)<=1,'Duplicate baseline/profile argument '+key)
 if key in args:args[args.index(key)+1]=str(value)
 else:args.extend([key,str(value)])

def configuration(cfg,slots):
 require(type(slots)is int and slots in (4,6),'Requested4/bounded6 only');args=list(cfg['args']);env=dict(cfg['env'])
 for key,value in [('--max-context',2048),('--prefill',64),('--batch',slots),('--batch-groups',1),('--prompt-cache',0),('--conversation-cache-mib',0),('--adapt-every',0)]:set_arg(args,key,value)
 env.update(STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_SLOT_OWNER_TRACE='1',STRATA_BATCH_FULL_STATE_CHAIN='0',STRATA_BATCH_PUBLIC_PREFIX='0',SYCL_UR_TRACE='2',STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0',STRATA_CRITICAL_PATH_TRACE='0',STRATA_MIRROR_OWNER_TRACE='1',STRATA_FIDELITY_DIAG='0',STRATA_LAYER0_Q8_DIAG='0')
 require('STRATA_VERIFY_EAGER' not in env and 'STRATA_CKPT_REREAD' not in env,'Exact normal graph/no checkpoint override presence required')
 return args,env

def binding(plan,cfg,prepared):
 require(plan['kind'] in ('native','serial'),'API cache0 is unsupported here; correct API cache/overlap purpose remains separate')
 args,env=configuration(cfg,plan['slots']);require(canonical(plan['args'])==canonical(args) and canonical(plan['env'])==canonical(env),'Actual baseline plus declared mode reconstruction rejects argument/math/environment drift')
 require(plan['image']==prepared['runtime']['image'] and plan['pack']==prepared['pack'] and plan['cards']==prepared['cards'],'Exact actual baseline image/pack/card binding differs');return True
