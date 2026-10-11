"""Original attach/pipe/container epoch joins; failure-only setup is never normal."""
from datetime import datetime
from source41_model_cli_contract_v1 import epoch,require
from serial37_canonical_json_v3 import canonical

def positive_integer(value):
    require(type(value)is int and value>0,'Typed original positive PID/start/session required');return value

def binding(launch,retired,capture,result,phase,parent):
    require(launch['actual_client_started']is True and launch['client_identity_observed']is True,'Normal original launch identity required')
    owner=launch['owner'];client=launch['client'];positive_integer(owner['pid']);positive_integer(owner['start_ticks']);positive_integer(client['pid']);positive_integer(client['start_ticks']);positive_integer(client['session'])
    require(owner['pid']==parent['producer_pid']and owner['start_ticks']==parent['producer_start_ticks']and client['session']==client['pid'],'Actual original parent/client ownership differs')
    require(canonical(launch['command'])==canonical(parent['actual_command'])==canonical(retired['command'])and canonical(client)==canonical(retired['client'])and retired['session_members']==[],'Exact reconstructed command/client retirement differs')
    require(type(retired['return_code'])is int and retired['return_code']==0 and type(capture['return_code'])is int and capture['return_code']==0 and type(result['return_code'])is int and result['return_code']==0,'Actual original attach/capture/result exit differs')
    require(phase['started_epoch']==result['created_epoch']and phase['finished_epoch']==result['finished_epoch'],'Actual phase/result epochs differ')
    ordered=[result['created_epoch'],capture['process_started_epoch'],launch['started_epoch'],capture['started_epoch'],capture['completed_epoch'],retired['retired_epoch'],capture['finished_epoch'],result['finished_epoch'],parent['actual_model_terminal_epoch']]
    for value in ordered:epoch(value)
    require(ordered==sorted(ordered),'Actual launch/pipe EOF/retirement/result/model chronology differs')
    epoch(phase['started_epoch']);epoch(phase['finished_epoch'])
    state=result['terminal']['State'];text=state['FinishedAt'];require(type(text)is str and text.endswith('Z'),'Actual UTC original container FinishedAt required')
    finished=datetime.fromisoformat(text[:-1]+'+00:00').timestamp();epoch(finished);require(finished<=capture['completed_epoch']and finished<=result['finished_epoch'],'Actual engine container terminal after captured EOF/result')
    return {'original_launch_pipe_retirement_crossjoined':True,'actual_command':launch['command'],'owner':owner,'client':client,'original_container_finished_epoch':finished}
