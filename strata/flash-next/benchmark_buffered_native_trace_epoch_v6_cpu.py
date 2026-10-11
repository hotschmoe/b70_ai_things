#!/usr/bin/env python3
"""CPU-only bounded sample replay; diagnostic sink cost, never inference speed."""
import argparse,cProfile,hashlib,json,os,platform,pstats,time
from pathlib import Path
import buffered_native_trace_epoch_v5 as old
import buffered_native_trace_epoch_v6 as new

OWN={'sequence','epoch','producer_identity','trace_file','trace_prefix_sha256_before','trace_start_offset','trace_end_offset','combined_start_offset','combined_end_offset','combined_line_index','combined_marker'}

def replay(module,rows,output,repeats):
 output.mkdir();sink=module.BufferedTrace(output/'trace.jsonl',output/'combined.log',interval=.1);wall=time.perf_counter();cpu=time.process_time()
 for _ in range(repeats):
  for previous in rows:
   row={k:v for k,v in previous.items()if k not in OWN}
   combined=row.get('line')if row['kind']=='native_receive'else None
   semantic=row['kind']in('engine_begin','engine_end','engine_close','eos_waiter_phase_proof','fullcache_control_send')or(combined is not None and combined.startswith(('STOP','BSTOP ','BYIELD ','QUIT','DONE ','BDONE ','BADM ','READY','INFO ','ERR','FATAL','YIELDED ','SERR ','SESSION ','SWAIT ','SAVED ','RESTORED ','SBF slot_closed ','PCL ')))
   sink.emit(row,combined_line=combined,semantic=semantic)
 status=sink.close();elapsed=time.perf_counter()-wall;used=time.process_time()-cpu
 return {'generation':module.__name__,'wall_seconds':elapsed,'CPU_seconds':used,'records':sink.record_count,'records_per_second':sink.record_count/elapsed,'bytes':(output/'trace.jsonl').stat().st_size+(output/'combined.log').stat().st_size,'status':status}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();raw=a.sample.read_bytes()
 if len(raw)>8<<20 or not 1<=a.repeats<=10:raise ValueError('Bounded CPU sample/replay required')
 rows=json.loads(raw);a.output.mkdir();results=[]
 for index,module in enumerate([old,new,new,old]):results.append(replay(module,rows,a.output/('run'+str(index)),a.repeats))
 profiler=cProfile.Profile();profiler.enable();profile_result=replay(old,rows,a.output/'profile-old',1);profiler.disable();profiler.dump_stats(str(a.output/'profile-old.pstats'))
 report={'schema':1,'CONFIG':'CPU-only V5/V6 ABBA replay of bounded copied trace records; no live producer/GPU/model payload changed','COMMAND':'benchmark_buffered_native_trace_epoch_v6_cpu.py --sample '+str(a.sample)+' --output '+str(a.output)+' --repeats '+str(a.repeats),'sample_sha256':hashlib.sha256(raw).hexdigest(),'sample_rows':len(rows),'source_sha256':{str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in [old,new]},'runtime':{'Python':platform.python_version(),'kernel':platform.release(),'CPU_affinity':sorted(os.sched_getaffinity(0))},'results':results,'profiled_run':profile_result,'VERDICT':'Component CPU replay only; no GPU producer equivalence, complete epoch, request latency or inference speed claim'}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(12)
 print(json.dumps({'report':str(a.output/'report.json'),'runs':[{k:r[k]for k in ['generation','wall_seconds','CPU_seconds','records_per_second']}for r in results]}))
if __name__=='__main__':main()
