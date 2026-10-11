"""Explicit hardware/CPU boundary; numerical and historical gates remain pending."""
import os
import time
import stat
from pathlib import Path
import native_qsa40_runtime_v9 as d
from serial37_canonical_json_v3 import canonical

HERE = Path(__file__).resolve().parent
PLAN = HERE/'native-qsa40-split-source-plan-v1.json'


def no_leases():
    locks=[os.stat('/mnt/vm_8tb/b70/gpu.lock.'+str(card))for card in(0,1)]
    for name in os.listdir('/proc/self/fd'):
        try: s=os.fstat(int(name))
        except OSError: continue  # A descriptor may retire during the census.
        d.require(not any((s.st_dev,s.st_ino)==(lock.st_dev,lock.st_ino)for lock in locks),
                  'CPU reader/orchestrator must not inherit any duplicated GPU lease descriptor')


def closure():
    value = d.read(PLAN)
    for path,digest in value['files'].items():
        d.require(d.sha(d.ROOT/path) == digest, 'Split source changed '+path)
    d.closure()  # Complete unchanged original runtime9 source dependency gate.
    return d.sha(PLAN)


def terminals(out,plan,report):
    from native_qsa40_reader_v9 import terminal
    d.require(set(report['phases']) == {'off','on'}, 'Complete actual two-arm hardware roster required')
    for label,on in (('off',False),('on',True)):
        ph = report['phases'][label];directory=out/label;result=d.read(directory/'result.json')
        d.require(type(ph['return_code']) is int and ph['return_code']==0 and ph['forced_cleanup'] is False and
                  ph['error'] is None and ph['stdout_capture']['eof'] is True and ph['stdout_capture']['reader_retired'] is True,
                  'Actual model child/EOF normal retirement required before release')
        argv,_,name=d.command(plan,directory,ph['child_pid'],on,result['render_gid'])
        d.require(d.read(directory/'command.json')==argv and d.c.absent(name), 'Actual original owner recipe/absence differs')
        terminal(result,directory,plan,argv,ph['child_pid'])
    return True


def hardware_source(plan):
    """Fresh source/SDK/pack bytes, not cached historical numerical admission."""
    start=time.time();m=plan['candidate_binding']['prepared']
    generation=d.c.combined_generation_gate(Path(m['engine_receipt']).parent)
    d.require(canonical(generation)==canonical(m['combined_generation']), 'Release SDK source/all8 ABI association differs')
    rows={}
    for path,expected in ((Path(plan['prepared'])/'prepared.json',plan['prepared_sha256']),
                          (Path(plan['prepared'])/'server-config.json',plan['server_config_sha256']),
                          (Path(plan['prepared'])/'artifact-identity.json',m['artifact_manifest_sha256']),
                          (d.ROOT/'evals/configs/models.yaml',plan['registry_sha256']),
                          (Path(m['runtime_receipt']),m['runtime_receipt_sha256']),
                          (Path(m['pack_receipt']),m['pack_receipt_sha256'])):
        actual=d.sha(path);d.require(actual==expected,'Current release metadata bytes differ '+str(path));rows[str(path)]=actual
    runtime=d.read(m['runtime_receipt']);packages=Path(runtime['packages_path'])
    actual=d.sha(packages);d.require(actual==runtime['packages_sha256'],'Current runtime package/library census bytes differ');rows[str(packages)]=actual
    for rel,record in d.read(m['pack_receipt'])['RESULT']['files'].items():
        path=Path(m['pack'])/rel;actual=d.sha(path);d.require(actual==record['sha256'],'Current release pack bytes differ '+rel);rows[str(path)]=actual
    d.require(canonical(d.kernel_binding(plan['engine_root']))==canonical(plan['kernel_source_flags']), 'Release kernel/object/link source changed')
    return {'schema':1,'started_epoch':start,'finished_epoch':time.time(),'original_runtime_source_sha256':d.closure(),
            'split_source_sha256':closure(),'generation':generation,'current_bytes':rows,
            'historical_reference_semantic_admission_pending':True,'source_or_model_stat_only_reuse':False}


def hardware_binding(root):
    root=Path(root);h=d.read(root/'hardware-report.json');plan=d.read(root/'plan.snapshot.json')
    d.require(type(h['schema']) is int and h['schema']==2 and h['parent_generation']=='qsa-split-v1' and
              h['hardware_passed'] is True and h['numerical_admission_pending'] is True and h['passed'] is False and
              h['errors']==[], 'Exact hardware-only pending receipt required; not a model qualification')
    d.require(h['split_source_plan_sha256']==closure() and h['plan_sha256']==d.sha(root/'plan.snapshot.json'), 'Actual hardware plan/source receipt changed')
    d.require(h['source_plan_sha256']==d.closure() and d.read(root/'source-plan.snapshot.json')==d.read(d.PLAN) and
              (root/'parent.snapshot.py').read_bytes()==(HERE/'qualify_native_qsa40_split_v1.py').read_bytes(), 'Exact actual child9/new-parent source snapshots differ')
    d.require(type(h['producer_pid'])is int and h['producer_pid']==h['producer_identity']['pid'] and d.sha(h['producer_interpreter'])==h['producer_interpreter_sha256'], 'Actual hardware producer/interpreter differs')
    from batch54_health_handshake_v1 import pid_identity
    try: live=pid_identity(h['producer_pid'])
    except (FileNotFoundError,ProcessLookupError):live=None
    d.require(live!=h['producer_identity'],'Actual leased hardware worker must exit before CPU admission')
    supervisor=Path(str(root)+'.hardware-supervisor.json');row=d.read(supervisor)
    d.require(row['passed']is True and type(row['return_code'])is int and row['return_code']==0 and row['eof']is True and
              row['reader_retired']is True and row['error']is None and row['command_error']is None and row['phase_cleanup_error']is None and
              row['sha256']==d.sha(row['path'])and h['finished_epoch']<=row['completed_epoch']<=row['finished_epoch'], 'Actual leased shell/worker EOF and normal terminal required before CPU phase')
    argv=row['command'];d.require(argv[:4]==[str(d.ROOT/'bin/gpu-run'),h['producer_interpreter'],str(HERE/'qualify_native_qsa40_split_v1.py'),'hardware']and
                argv[argv.index('--output')+1]==str(root)and '--leased'in argv and
                canonical(d.read(argv[argv.index('--plan')+1]))==canonical(plan),'Actual hardware supervisor command/plan differs')
    from native_qsa40_evidence_v9 import parent_identity
    for phase in h['phases'].values():parent_identity(h,phase['ready'])
    terminals(root,plan,h)
    keys=set(h['artifact_sha256']);current=set()
    for path in root.rglob('*'):
        d.require(not path.is_symlink()and path.resolve().is_relative_to(root.resolve()), 'Confined regular hardware/CPU evidence tree required')
        d.require(stat.S_ISREG(path.stat().st_mode)or path.is_dir(),'No special hardware/CPU evidence files')
        if path.is_file():current.add(str(path.relative_to(root)))
    extra=current-keys-{'hardware-report.json'}
    allowed={'numerical-comparison.json','localization.json','original-input33.json','cpu-candidate-report.json','report.json'}
    d.require(extra<=allowed|{r for r in extra if r.startswith('cpu-closing-source/')}, 'Unexpected hardware tree additions')
    for rel,expected in h['artifact_sha256'].items():
        path=root/rel;d.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) and d.sha(path)==expected,
                              'Immutable hardware artifact changed '+rel)
    from native_qsa40_evidence_v9 import health,journal
    health(root,'post',h['post_health'],h['GPU_terminal_epoch']);journal(root,'post',h['post_journal'],h['started_epoch'],h['post_health']['finished_epoch'])
    from native_rms_publisher_binding_v2 import publisher_binding
    publisher_binding(root,h,d.read(d.HERE/'model-lock.json'),d.sha(d.HERE/'model-lock.json'),d.ROOT)
    d.require(h['post_full4']['finished']<=h['hardware_source']['started_epoch']<=h['hardware_source']['finished_epoch']<=h['finished_epoch'],
              'Hardware source must close after actual hardware publisher scan')
    fresh=hardware_source(plan)
    d.require(canonical({k:v for k,v in fresh.items()if k not in('started_epoch','finished_epoch')})==
              canonical({k:v for k,v in h['hardware_source'].items()if k not in('started_epoch','finished_epoch')}), 'Saved hardware source receipt differs from actual current release-byte gates')
    return h,plan


def cpu_closing_binding(root,report,hardware):
    """Rejoin the producer's actual closing scan; fresh public scan is extra."""
    import math
    from source_page_watchdog_v3 import KNOWN_PAGES
    root=Path(root).resolve();closing=report['CPU_closing_source'];directory=root/'cpu-closing-source'
    epochs=[hardware['finished_epoch'],report['cpu_started_epoch'],report['numerical_admission_finished_epoch'],
            report['CPU_read_phase_finished_epoch'],report['cpu_finished_epoch'],report['finished_epoch']]
    d.require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in epochs)and epochs==sorted(epochs),'Exact hardware/CPU numerical/closing chronology required')
    identity=report['cpu_identity'];d.require(set(identity)=={'pid','start_ticks'}and type(identity['pid'])is int and identity['pid']>0 and
                type(identity['start_ticks'])is int and identity['start_ticks']>=0 and identity!=hardware['producer_identity'],'Exact separate CPU producer identity required')
    d.require(closing['output']==str(directory)and closing['GPU_touched']is False and closing['source_stat_only_reused']is False and
              closing['boundary_epoch']==report['CPU_read_phase_finished_epoch']and closing['full4_sha256']==d.sha(directory/'full4.json')and
              canonical(closing['full4'])==canonical(d.read(directory/'full4.json')),'Original CPU closing path/hash/body differs')
    proof=closing['full4'];lock=d.read(d.HERE/'model-lock.json');wanted=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    d.require(type(proof['schema'])is int and proof['schema']==1 and proof['passed']is True and proof['lock_sha256']==d.sha(d.HERE/'model-lock.json')and proof['model_revision']==lock['revision']and
              len(proof['rows'])==4 and proof['after_terminal_and_post_health_epoch']==closing['boundary_epoch']and
              closing['boundary_epoch']<=proof['started']<=proof['finished']<=report['cpu_finished_epoch'],'Actual producer closing publisher/epoch differs')
    for row,want in zip(proof['rows'],wanted):
        path=d.ROOT/lock['destination']/want['path']
        d.require(row['path']==str(path)and row['passed']is True and type(row['bytes'])is int and row['bytes']==want['size']and row['sha256']==row['expected_sha256']==want['sha256']and
                  row['stat_before']==row['stat_after']==d.c.stat_signature(path),'Exact CPU closing ordered4 byte/hash/stat differs')
    names={'full4.json'}
    for label,key in (('before','before_pages'),('after','after_pages')):
        guard=closing[key];name=label+'-source-pages.json';names.add(name)
        d.require(canonical(guard)==canonical(d.read(directory/name))and guard['passed']is True and guard['path']==proof['rows'][2]['path']and
                  guard['stat_before']==guard['stat_after']==proof['rows'][2]['stat_after']and
                  [(r['offset'],r['expected_sha256'])for r in guard['rows']]==list(KNOWN_PAGES),'Exact producer closing knownpage body/stat differs')
        for row in guard['rows']:
            name=label+'-buffered-page-'+str(row['offset'])+'.raw';names.add(name);path=directory/name
            d.require(row['preserved_path']==str(path)and row['passed']is True and row['bytes']==4096 and path.stat().st_size==4096 and
                      d.sha(path)==row['sha256']==row['expected_sha256'],'Original producer closing preserved bytes differ')
    d.require(closing['before_pages']['epoch']<=proof['started']<=proof['finished']<=closing['after_pages']['epoch']<=report['cpu_finished_epoch']and
              {p.name for p in directory.iterdir()}==names,'Exact CPU closing page brackets/file roster required')
    return True
