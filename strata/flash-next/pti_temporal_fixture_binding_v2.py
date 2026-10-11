"""Mandatory genuine CPU-only actual CLI proof; no model authority."""
import hashlib,json
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
import pti_owned_temporal_control_v2 as helper
ROOT=Path('/mnt/vm_8tb/github/b70_ai_things')
FIXTURE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/pti-temporal-nodri-fixture-v2')
RECEIPT_SHA='2bfaa4ac0ece1cd981a3a6965e49b2d413dbe41ff006d1b1b89024db8a2e964e'
INSIDE_SHA='ac2fc4f35ed9d6cfa6cb119a447d17838cb793e402463680b39201d42a729cde'
require=helper.require

def check_inside(body):
    require(body['passed']is True and body['errors']==[]and body['GPU_devices_exposed']is False and body['model_payload_read']is False and body['model_runtime_qualification_transferred']is False and body['synthetic_IPC_removed']is True,'Actual CPU fixture false scopes/cleanup required')
    require(type(body['before']['state'])is int and body['before']['state']==0 and type(body['paused']['state'])is int and body['paused']['state']==1,'Actual CLI pause transition missing');helper.check_transition(body['paused'],body['resumed'],'resume')
    session=body['session'];rows=body['commands'];require(len(rows)==3 and canonical(rows[0]['command'])==canonical(['/pti/bin/unitrace','--pause',session])and canonical(rows[1]['command'])==canonical(['/pti/bin/unitrace','--resume',session])and all(type(r['return_code'])is int and r['return_code']==0 for r in rows[:2]),'Actual installed CLI/argv/return values differ')
    require(body['missing_session_helper_refused']is True and body['missing_session_object_created']is False,'Actual absent-session helper refusal required')

def binding():
    def read(name,want):
        p=FIXTURE/name;require(p.resolve()==p and not p.is_symlink()and p.stat().st_size<=1<<20,'Bounded original fixture file required');before=p.stat();raw=p.read_bytes();after=p.stat();sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);require(sig(before)==sig(after)and hashlib.sha256(raw).hexdigest()==want,'Fixture original bytes changed');return read_unique(p)
    receipt=read('receipt.json',RECEIPT_SHA);inside=read('inside.json',INSIDE_SHA);require(receipt['passed']is True and receipt['errors']==[]and receipt['GPU_devices_exposed']is False and receipt['model_payload_read']is False and receipt['normal_owned_removed']is True and receipt['inside_sha256']==INSIDE_SHA,'Actual original NoDri receipt required');check_inside(inside)
    require(inside['helper_source_sha256']==hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),'Actual tested helper bytes changed')
    command=read('command.json',receipt['command_sha256']);terminal=read_unique(FIXTURE/'terminal-inspection.json');absence=read_unique(FIXTURE/'absence.json');require(terminal['Id']==receipt['original_container_id']and terminal['Config']['Image']=='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'and not terminal['HostConfig']['Devices']and not terminal['HostConfig']['DeviceRequests']and terminal['HostConfig']['Privileged']is False,'Actual no-device typed original terminal required')
    require(type(terminal['State']['ExitCode'])is int and terminal['State']['ExitCode']==0 and terminal['State']['Running']is False and terminal['State']['OOMKilled']is False and terminal['State']['Error']=='' and type(absence['return_code'])is int and absence['return_code']!=0 and 'no such object'in absence['stderr'].lower(),'Actual normal terminal/removal required')
    return dict(receipt_sha256=RECEIPT_SHA,inside_sha256=INSIDE_SHA,actual_CLI_resume_accepted=True,synthetic_IPC_only=True,model_runtime_qualification_transferred=False)
