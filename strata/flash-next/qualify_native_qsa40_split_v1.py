#!/usr/bin/env python3
"""New owned hardware worker, followed by original numerical gates without lease."""
import argparse
import ast
import copy
import os
import sys
import time
import traceback
from pathlib import Path
import qualify_native_qsa40_v9 as old
import native_qsa40_runtime_v9 as d
import native_qsa40_split_contract_v1 as split


def finish_hardware(out,plan,report):
    report['schema']=2;report['parent_generation']='qsa-split-v1'
    report['split_source_plan_sha256']=split.closure()
    report['numerical_admission_pending']=True;report['passed']=False
    report['model_computation_terminal_kind']='owned OFF/ON model children retired host upper bound; not device clock'
    report['hardware_passed']=not report['errors'] and report.get('hardware_terminals_closed') is True and 'hardware_source' in report
    report['finished_epoch']=time.time()
    report['artifact_sha256']={str(p.relative_to(out)):d.sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='hardware-report.json'}
    d.write(out/'hardware-report.json',report)
    return int(not report['hardware_passed'])


def hardware_node():
    tree=ast.parse(Path(old.__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    node.args.args.append(ast.arg(arg='argv'));node.args.defaults.append(ast.Constant(value=None))
    count=0
    for n in ast.walk(node):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and ast.unparse(n.func)=='ap.parse_args':n.args=[ast.Name(id='argv',ctx=ast.Load())];count+=1
    d.require(count==1,'Exactly one explicit argument parser route required')
    body=next(n for n in node.body if isinstance(n,ast.Try))
    d.require(len(body.body)==18 and isinstance(body.body[5],ast.ImportFrom)and body.body[5].module=='native_qsa40_reader_v9','Exact original numerical block before hardware close required')
    body.body=body.body[:5]+ast.parse("report['hardware_terminals_closed'] = split.terminals(out,plan,report)\nreport['model_computation_terminal_epoch'] = time.time()\n").body
    # Original owner-fence/cleanup, real health/journal and fresh full4 stay.
    post=body.finalbody[2]
    d.require(isinstance(post,ast.Try)and ast.unparse(post.body[-1])=='d.manifest(plan)','Exact original final manifest location required')
    post.body[-1]=ast.parse("report['hardware_source'] = split.hardware_source(plan)").body[0]
    # Keep the original full4 file as the CPU-entry publisher byte proof.
    body.finalbody=body.finalbody[:3]+ast.parse('finish_hardware(out,plan,report)').body
    node.body=node.body[:-2]+ast.parse("return int(not report['hardware_passed'])").body
    return node


def hardware_main(argv):
    split.closure();node=hardware_node()
    namespace=dict(vars(old));namespace.update(__file__=__file__,__name__=__name__,split=split,finish_hardware=finish_hardware)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    return namespace['main'](argv)


def cpu_phase(root):
    split.no_leases();root=Path(root).resolve();hardware,plan=split.hardware_binding(root)
    d.require(not(root/'report.json').exists()and not(root/'cpu-candidate-report.json').exists(),'Never overwrite executed CPU finalization')
    report=copy.deepcopy(hardware);report['hardware_report_sha256']=d.sha(root/'hardware-report.json')
    report['cpu_identity']=old.pid_identity(os.getpid());report['cpu_started_epoch']=time.time()
    report['hardware_health_is_current_CPU_authority']=False
    try:
        # All original post-operation semantic gates remain, now explicitly CPU.
        d.manifest(plan)
        from native_qsa40_reader_v9 import arm,comparisons
        off=arm(root/'off',plan,report['phases']['off'],False,report['plan_sha256'])
        on=arm(root/'on',plan,report['phases']['on'],True,report['plan_sha256'])
        report['comparison']=comparisons(off,on)
        from native_qsa40_input33_v9 import verify
        from native_qsa40_localization_v9 import join
        report['original_input33']={'off':verify(off['source33'],plan['current_identity']),'on':verify(on['source33'],plan['current_identity'])}
        report['localization']=join(plan['own_producer_root'],on['targets'],plan['own_producer_binding']['model_identity_association'])
        for key,name in (('comparison','numerical-comparison.json'),('original_input33','original-input33.json'),('localization','localization.json')):d.write(root/name,report[key])
        report['passed']=True;report['finished_epoch']=time.time()
        report['artifact_sha256']={str(p.relative_to(root)):d.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='cpu-candidate-report.json'}
        d.write(root/'cpu-candidate-report.json',report)
        from validate_native_qsa40_split_v1 import legacy_binding
        original_result=legacy_binding(root,'cpu-candidate-report.json')
        report['numerical_admission_finished_epoch']=time.time()
        report['original_numerical_binding']=original_result
    except BaseException as exc:
        report['errors'].append('CPU: '+type(exc).__name__+': '+str(exc));report['failure_traceback']=traceback.format_exc();report['passed']=False
    finally:
        report['CPU_read_phase_finished_epoch']=time.time()
        try:
            from validate_native_qsa40_split_v1 import closing_hash
            report['CPU_closing_source']=closing_hash(root,report['CPU_read_phase_finished_epoch'],root/'cpu-closing-source')
            split.closure();split.no_leases()
            if report['passed']and not report['errors']and 'original_numerical_binding'in report:report['numerical_admission_pending']=False
        except BaseException as exc:
            report['errors'].append('CPU closing source: '+type(exc).__name__+': '+str(exc));report['passed']=False
        report['cpu_finished_epoch']=time.time();report['finished_epoch']=time.time()
        report['artifact_sha256']={str(p.relative_to(root)):d.sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='report.json'}
        d.write(root/'report.json',report)
    return int(not report['passed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','hardware','cpu'));p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--one-card-receipt',type=Path);p.add_argument('--max-runtime',type=int,default=10800);p.add_argument('--leased',action='store_true');a=p.parse_args()
    d.require(sys.flags.optimize==0,'Ordinary Python required')
    if a.mode=='cpu':return cpu_phase(a.output)
    d.require(a.plan is not None,'Actual runtime9 prepared plan required')
    argv=['--plan',str(a.plan.resolve()),'--output',str(a.output.resolve()),'--max-runtime',str(a.max_runtime),'--leased']
    if a.one_card_receipt:argv+=['--one-card-receipt',str(a.one_card_receipt.resolve())]
    if a.mode=='hardware':
        if not a.leased:os.execv(str(d.ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,'hardware',*argv])
        return hardware_main(argv)
    split.no_leases();d.require(not a.leased,'Run orchestrator must remain outside lease')
    for sig in(old.signal.SIGTERM,old.signal.SIGINT,old.signal.SIGHUP):
        old.signal.signal(sig,lambda number,frame:(_ for _ in()).throw(InterruptedError('Controlled unleased orchestrator stop '+str(number))))
    from native_rms_phase_supervisor_v3 import supervise
    log=Path(str(a.output)+'.hardware-supervisor.log')
    command=[str(d.ROOT/'bin/gpu-run'),sys.executable,__file__,'hardware',*argv]
    row=supervise(command,log,2*a.max_runtime+7200)
    d.write(Path(str(a.output)+'.hardware-supervisor.json'),row)
    d.require(row['passed']is True and row['eof']is True and row['reader_retired']is True,'Actual leased worker/process streams must retire before CPU phase')
    split.no_leases()
    return cpu_phase(a.output)


if __name__=='__main__':raise SystemExit(main())
