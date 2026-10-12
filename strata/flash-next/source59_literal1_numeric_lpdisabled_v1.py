"""Exact existing literal1/PCL numeric grammar with preregistered LP omission."""
from types import FunctionType,SimpleNamespace
import source44_num38_numeric_v2 as original
from source41_lpdisabled_contract_v1 import _REQUEST

def port():
 old=original.port()[0];namespace=dict(old.__globals__)
 # qsa.IDS remains the corrected literal[248045]. Only the command formatter
 # was stale: its original LP20 is replaced with the already executed omission.
 namespace['r']=SimpleNamespace(**{**vars(namespace['r']),'request_command':_REQUEST['request_command']})
 return FunctionType(old.__code__,namespace,old.__name__,old.__defaults__,old.__closure__)
def numeric(raw,directory,stages):return port()(raw,directory,stages)
