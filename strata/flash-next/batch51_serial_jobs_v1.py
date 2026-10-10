"""Recollect exact H50 native consumed-prefix jobs; no invented migration role."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from batch_numerical_protocol_v2 import Roster
from batch_numerical_prefixes_v2 import serial_jobs,token_sha

def require(ok,message):
 if not ok:raise ValueError(message)
def validate(value,slots):
 require(value['supplied_consumed_prefix_only']is True and value['full_model_math_qualified']is False,'Exact supplied native prefix scope required');jobs=value['jobs'];require(type(jobs)is list and 1<=len(jobs)<=18,'Bounded actual serial job roster required');keys=[]
 for job in jobs:
  require(type(job['rid'])is int and 2001<=job['rid']<=2000+slots and job['role']in ('admission','later'),'Actual native-only job RID/role invalid');ids=job['ids'];require(type(ids)is list and 1<=len(ids)<=2048 and all(type(t)is int and 0<=t<248320 for t in ids),'Actual bounded consumed prefix required');require(job['ids_sha256_le32']==token_sha(ids) and type(job['position'])is int and job['position']==len(ids)-1 and type(job['token'])is int and job['token']==ids[-1] and type(job['max_new'])is int and job['max_new']==1 and job['fresh_serial']is True and job['original_own_state_math_reference']is False,'Exact serial one-token/prefix/hash policy differs');keys.append((job['rid'],job['role']))
 require(len(set(keys))==len(keys),'Duplicate actual serial RID/role');return jobs

def recollect(child,slots):
 child=Path(child);saved=read_unique(child/'serial-jobs.json');validate(saved,slots);roster=Roster(slots);roster.requests={int(k):v for k,v in read_unique(child/'requests.json').items()};actual=serial_jobs((child/'engine.combined.log').read_text(),roster);require(canonical(saved)==canonical(actual),'Saved serial jobs differ from actual full49 producer/prefix history');return saved

def group(value,slots,index):
 jobs=validate(value,slots);require(type(index)is int and 0<=index<(len(jobs)+5)//6,'Actual bounded job group invalid');selected=jobs[index*6:index*6+6];return {'job_count':len(jobs),'groups_required':(len(jobs)+5)//6,'group_index':index,'selected_jobs':selected,'selected_job_count':len(selected),'selected_full49_pairs':49*len(selected),'all_full49_pairs':49*len(jobs),'solo_migration_observed':False,'full_model_math_qualified':False}
