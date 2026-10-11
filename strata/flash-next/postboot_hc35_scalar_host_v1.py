"""Historical scalar HC known-answer predicates with current file/PATH association."""
from pathlib import Path
import qualify_hc35_host_runtime_v1 as original
from postboot_hc35_host_identity_v1 import Association,derived,current_runtime,model_port

def finalized_binding(run_root,helper,build_root,model_association):
 model_port.recheck(model_association);root=Path(run_root).resolve();report=original.read(root/'report.json');ctx=Association(report,model_association);ns=derived(original,('validate_fixture','finalized_binding'),ctx,{'_historical_runtime':report['runtime_post']});before=original.sha(root/'report.json');result=ns['finalized_binding'](run_root,helper,build_root);evidence=ctx.evidence();original.require(before==original.sha(root/'report.json'),'Historical scalar report changed');model_port.recheck(model_association)
 return {'original_binding':result,'host_identity_association':evidence,'old_current_runtime_gate_passed':False,'historical_evidence_only':True,'helper_numerical_fixture_execution_repeated':False,'current_GPU_or_model_runtime_qualified':False,'original_reports_changed':False}
