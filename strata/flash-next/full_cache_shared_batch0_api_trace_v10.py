"""NEW serial persisted observer namespace; no strict-batch capability rewrite."""
from pathlib import Path
import full_cache_shared_api_trace_v10 as base

def require(ok,message):
 if not ok:raise ValueError(message)

def cfg_gate(cfg):
 args=cfg['args'];env=cfg['env'];require(args.count('--batch')==1 and args[args.index('--batch')+1]=='0' and cfg['parallel']==1 and cfg['slot_save_path']=='/results/sessions','Actual purpose serial batch0/FIFO/session profile required')
 require(args.count('--prompt-cache')==1 and args[args.index('--prompt-cache')+1]=='3','Purpose persisted original serial cache3 required')
 expected={'STRATA_BATCH_FIDELITY_DIAG':'0','STRATA_BATCH_FULL_STATE_CHAIN':'0','STRATA_BATCH_PUBLIC_PREFIX':'0','STRATA_FULL_CACHE_OBSERVER38':'0','STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1','STRATA_PREFIX30':'0','STRATA_PLE_INPUT33':'0','STRATA_LAYER0_Q8_DIAG':'0'}
 require(all(env.get(k,'0')==v for k,v in expected.items()) and 'STRATA_VERIFY_EAGER' not in env and 'STRATA_CKPT_REREAD' not in env,'Declared existing normal serial PCL/full49 route required')
 return True

def adapted_source():
 source=Path(base.__file__).read_text();load="cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());eos=";require(source.count(load)==1,'Exact original config loading boundary changed');source=source.replace(load,"cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());cfg_gate(cfg);eos=")
 boundary=' generate_original=server.StrataEngine.generate\n';require(source.count(boundary)==1,'Exact original producer session hook boundary changed');source=source.replace(boundary," from full_cache_shared_session_observer_v10 import install as session_observer\n session_observer(server,emit,local,counter,lock,active_calls)\n"+boundary)
 guard="if __name__=='__main__':";require(source.count(guard)==1,'Exact original observer entry guard changed');return source[:source.index(guard)]

def main():
 namespace=dict(vars(base));namespace.update(__name__='owned_batch0_PCL_persisted_observer',__file__=str(Path(__file__).resolve()),cfg_gate=cfg_gate);exec(compile(adapted_source(),str(Path(base.__file__))+'[NEW-batch0-PCL-session-only]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
