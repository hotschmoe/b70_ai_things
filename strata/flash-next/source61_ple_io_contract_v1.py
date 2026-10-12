"""Exact PLE IO-only comparison; unchanged histories and eight host runs each."""
import re
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from source61_loaded_host_contract_v1 import requests as loaded_requests
from native60_loaded_fairness_protocol_v1 import require,fields,command_for

def requests(rows,expected):
 value=loaded_requests(rows,expected)
 directory=Path(rows[0]['raw']['phase_directory'])
 lines=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines()
 reports=[line for line in lines if 'ple io:' in line]
 value['PLE_native_io_reports']=reports
 value['PLE_native_IO_report_precision']='rows/reads integers; cache-hit percent and physical MB rounded by existing native producer'
 value['PLE_cold_cache_qualified']=False
 value['PLE_physical_bytes_exact_qualified']=False
 value['PLE_IO_counters_available']=bool(reports)
 value['PLE_IO_counter_emission_scope']='Existing native serve does not call non-serving io_report printer; absence is explicit, not zero reads'
 return value

def matched_recipes(recipes):
 require(len(recipes)==2,'Exactly mmap/direct recipes required')
 left,right=recipes
 for recipe,mode in zip(recipes,('mmap','direct')):
  argv=recipe['native_argv']
  require(recipe['PLE_mode']==mode and argv.count('--ple-io')==argv.count('--ple-row-cache')==argv.count('--ple-inflight')==1 and argv[argv.index('--ple-io')+1]==mode and argv[argv.index('--ple-row-cache')+1]=='65536' and argv[argv.index('--ple-inflight')+1]=='256' and '--ple-sync-submit' not in argv,'Exact PLE mode/cache/default worker/inflight required')
  require(recipe['PLE_direct_cache_payload_bytes']==5898240 and recipe['PLE_cold_residency_qualified'] is False,'Honest bounded raw-cache and residency scope required')
 a=list(left['native_argv']);b=list(right['native_argv']);a[a.index('--ple-io')+1]=b[b.index('--ple-io')+1]='PLE_MODE'
 require(a==b and canonical(left['native_env'])==canonical(right['native_env']) and canonical(left['requests'])==canonical(right['requests']),'Only PLE mode may change actual arguments/environment/requests')
 return True

def equivalence(*arms):
 require(len(arms)==2 and all(a['loaded_processes']==1 and a['repeated_fresh_histories_equal'] is True for a in arms),'Two complete protected loaded PLE arms required')
 for left,right in zip(arms[0]['cohorts'],arms[1]['cohorts']):
  for a,b in zip(left['streams'],right['streams']):
   require(all(a[k]==b[k] for k in ('role','ids','output_ids','finish')),'PLE mode complete histories differ')
 require(not arms[0]['PLE_native_io_reports'],'Mmap must not masquerade as Direct IO counters')
 return dict(bounded61_PLE_IO_comparison_passed=True,observations=list(arms),only_PLE_mode_changed=True,complete_histories_equal=True,host_only=True,cold_cache_qualified=False,physical_bytes_exact_qualified=False,original_fullmath_qualified=False,API_qualified=False,shelf_qualified=False,clean_performance_qualified=False)
