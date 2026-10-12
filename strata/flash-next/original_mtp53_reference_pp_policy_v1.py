"""Pinned normal-serving metadata policy; boundaries derive from EACH actual prompt."""
import hashlib,json
from pathlib import Path
REFERENCE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-native2-fairness-v1-run-v1/source52-native2-fairness-serial/recipe.json')
REFERENCE_SHA='7d4edcacf40206f26270f28fdc3343bd698344e41a2f85237b51f74b86fced06'
def policy():
 raw=REFERENCE.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA:raise ValueError('Exact accepted reference recipe changed')
 r=json.loads(raw);argv=r['native_argv'];env=r['native_env']
 def arg(name,default=None):
  if argv.count(name)>1:raise ValueError('Duplicate reference option '+name)
  return argv[argv.index(name)+1]if name in argv else default
 pins={v['pin']for v in r['requests']}
 if arg('--prompt-cache')!='3' or arg('--prompt-cache-root')!='1' or len(pins)!=1 or any(v.get('fresh')!=1 or v.get('ckpt',1)!=1 for v in r['requests'])or env.get('STRATA_CACHE_MESSAGE_BOUNDARY','0')!='0'or 'STRATA_CKPT_REREAD'in env:raise ValueError('Exact normal fresh root/PIN/turn policy required')
 return {'turn_token':int(arg('--turn-token','248045')),'tail_role_token':int(arg('--tail-role-token','-1')),'root_min':int(arg('--prompt-cache-root')),'pin':next(iter(pins)),'request_checkpoint':True}
def cuts(prompt,p):
 n=len(prompt);turn=next((i for i in range(n-1,0,-1)if prompt[i]==p['turn_token']),-1)
 if turn>0 and p['tail_role_token']>=0:
  for i in range(turn-1,0,-1):
   if prompt[i]==p['turn_token']:
    if prompt[i+1]==p['tail_role_token']:turn=i
    break
 last_turn=turn
 if not p['request_checkpoint']:turn=-1
 root=-1
 if p['root_min']>0:
  for i in range(1,last_turn):
   if prompt[i]==p['turn_token']:
    if i>=p['root_min']:root=i
    break
 return sorted(set(x for x in(root,p['pin'],turn,n-1)if 0<x<=n-1))
def header(p):
 return '''#pragma once
#include <algorithm>
#include <vector>
namespace strata::kernels::original_mtp53 {
// Policy constants from pinned actual normal reference metadata; actual prompt positions are scanned.
inline std::vector<int64_t> normal_reference_prefix_cuts53(const std::vector<int64_t>& prompt){
 const int64_t n=int64_t(prompt.size()),turn_token=%d,tail_role=%d,root_min=%d,pin=%d;
 int64_t turn=-1;for(int64_t i=n-1;i>0;--i)if(prompt[size_t(i)]==turn_token){turn=i;break;}
 if(turn>0 && tail_role>=0)for(int64_t i=turn-1;i>0;--i)if(prompt[size_t(i)]==turn_token){if(prompt[size_t(i+1)]==tail_role)turn=i;break;}
 int64_t root=-1;if(root_min>0)for(int64_t i=1;i<turn;++i)if(prompt[size_t(i)]==turn_token){if(i>=root_min)root=i;break;}
 std::vector<int64_t> cuts{root,pin,turn,n-1};std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
 cuts.erase(std::remove_if(cuts.begin(),cuts.end(),[&](int64_t x){return x<=0 || x>n-1;}),cuts.end());return cuts;
}
}
'''%(p['turn_token'],p['tail_role_token'],p['root_min'],p['pin'])
