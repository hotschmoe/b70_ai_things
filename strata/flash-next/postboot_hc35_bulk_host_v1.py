"""Historical full bulk math probes; no old runtime/current-health promotion."""
from pathlib import Path
import qualify_hc35_host_bulk_runtime_v1 as original
import qualify_hc35_host_runtime_v1 as scalar
from postboot_hc35_host_identity_v1 import Association,derived,model_port
import postboot_hc35_scalar_host_v1 as scalar_history

def finalized_binding(build_root,model_association):
 model_port.recheck(model_association);root=Path(build_root).resolve();report=original.read(root/'qualification/report.json');ctx=Association(report,model_association);scalar_evidence=[]
 def scalar_binding(run,helper,build):
  binding=scalar_history.finalized_binding(run,helper,build,model_association);scalar_evidence.append(binding);return binding['original_binding']
 scalar_ns=derived(scalar,('validate_fixture',),ctx)
 ns=derived(original,('finalized_binding',),ctx,{'_historical_runtime':report['runtime_post'],'_scalar_history':scalar_binding,'_validate_fixture':scalar_ns['validate_fixture']});before=original.sha(root/'qualification/report.json');result=ns['finalized_binding'](root);evidence=ctx.evidence();original.require(before==original.sha(root/'qualification/report.json'),'Historical bulk report changed');model_port.recheck(model_association)
 return {'original_binding':result,'host_identity_association':evidence,'scalar_host_association':scalar_evidence,'old_current_runtime_gate_passed':False,'historical_evidence_only':True,'helper_numerical_fixture_execution_repeated':False,'current_GPU_or_model_runtime_qualified':False,'original_reports_changed':False}
