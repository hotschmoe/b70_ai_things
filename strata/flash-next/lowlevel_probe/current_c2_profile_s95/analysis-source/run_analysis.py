from pathlib import Path
import hashlib,json,subprocess,sys,time
R=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014');P=R/'current-c2-profile-s95-prep';O=P/'run';A=O/'analysis'
assert json.loads((O/'closure.json').read_text())['status']=='NORMAL_TEARDOWN_POSTHEALTH_PASS'
assert hashlib.sha256((Path(__file__).parent/'analyze_trace.py').read_bytes()).hexdigest()=='d9d21c1860ae011579857f0caa7dbd1b0f60cef014bb19b90820c3fa2e123e72'
assert hashlib.sha256((P/'cpu_markers.py').read_bytes()).hexdigest()=='25b1b1d0d176e069d624d7eeb9cd49d257f53bf995540d404598b42710dbb51d'
C=json.loads((O/'collection.json').read_text());trace=Path(C['trace']['path']);assert trace.stat().st_size==C['trace']['bytes'];W=json.loads((O/'window.json').read_text());A.mkdir()
def save(name,v):(O/name).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
cmd=[sys.executable,str(Path(__file__).parent/'analyze_trace.py'),str(trace),'--timebase',str(O/'timebase-before.json'),'--start-us',str(W['start_us']),'--end-us',str(W['end_us']),'--output',str(A/'gpu')];t=time.monotonic()
with(A/'gpu-parser.log').open('x')as f:rv=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
save('raw-parser-exit.json',{'command':cmd,'exit':rv.returncode,'elapsed_s':time.monotonic()-t,'one_full_raw_JSON_parse_only':True,'scope':'Optimized overlap-query postprocessor only; executed runtime inputs unchanged.'});assert rv.returncode==0
cmd=[sys.executable,str(P/'cpu_markers.py'),str(O/'pair-engine-complete.log'),'--output',str(A/'cpu')];t=time.monotonic()
with(A/'cpu-parser.log').open('x')as f:rv=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
save('cpu-parser-exit.json',{'command':cmd,'exit':rv.returncode,'elapsed_s':time.monotonic()-t});assert rv.returncode==0
save('analysis-limits.json',{'status':'SEPARATE_CPU_GPU_DIAGNOSTICS_REQUIRE_ROOT_REVIEW','native_steady_to_MONOTONIC_identity':'UNVERIFIED','native_to_trace_timestamp_join_allowed':False,'ordinal_reconciliation_only':True,'no_speed_qualification':True,'inadequate_replay_cycles':'INCONCLUSIVE_NO_RERUN','coverage_queries':'Sorted host API interval sweep preserves strict overlap, tested on16 CPU-only boundary/nested/duplicate/zero queries; not a GPU runtime change.','limits':['No physical idle/bandwidth/traffic/removable-gap claim.','No cross-card clock join or additive CPU/GPU budgets.','Structural C2x2/EP48 patterns require motif/nativecounter/CPUrole reconciliation.','Host L0 fully-contained counts and boundarycrossers separate; not total callstarts.','Allcold/solo/tail retained; kernelnames/grids mayconflateweights/sharedroles.']})
print(json.dumps({'status':'ONE_RAW_PARSE_AND_CPU_DIAGNOSTICS_COMPLETE_REQUIRE_ROOT_REVIEW'}))
