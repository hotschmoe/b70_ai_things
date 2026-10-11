"""Explicit recursive historical RMS/HC dependencies, output-target scope only."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from postboot_original_model_association_v1 import require
from postboot_original_saved_input_v1 import saved_original_input

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return read_unique(p)
def historical_fixture(original_module,root,model_association):
 p,*_=original_module.modules();root=Path(root).resolve();r=read(root/'fixture.json');require(r['source_binding']==p.source_binding() and r['builder_flags']==p.builder_flags() and r['compile_argv']==p.compile_argv(),'Actual preregistration source/flags changed');residual,b=saved_original_input(model_association);require(r['original_saved_input_binding']==b and (root/'residual.f32').read_bytes()==residual.tobytes(),'Original saved residual fixture changed')
 expected=p.mathref.variants(residual);require(set(r['hypotheses'])==set(expected),'Complete preregistered families required')
 for family,fields in expected.items():
  require(set(r['hypotheses'][family])==set(fields),'Complete preregistered variant roster differs')
  for name,raw in fields.items():
   path=root/(family+'-'+name+'.f32');item=r['hypotheses'][family][name];require(item['path']==str(path) and item['sha256']==sha(path) and item['bytes']==len(raw) and path.read_bytes()==raw,'Preregistered hypothesis bytes changed')
 return {'fixture_sha256':sha(root/'fixture.json'),'root':str(root),'input_sha256':sha(root/'residual.f32'),'record':r}

def prior_v7(root,model_association):
 import native_rms_owned_device_ops_binding_v8 as old
 old.header(root)
 from postboot_rms37_v7_historical_reader_v3 import finalized_binding
 binding=finalized_binding(root,model_association)['original_binding']
 require(binding['report_sha256']==old.REPORT_SHA and canonical(binding)==canonical(read_unique(old.EXTERNAL)),'Historical V7 contradicts original external admission')
 report=read_unique(Path(root)/'report.json')
 return {'root':str(Path(root).resolve()),'finalized_binding':binding,'captured_fields_used_as_inputs':False,'prior_fields_output_targets_only':True,'fixture_binding':report['fixture_binding'],'external_binding_sha256':old.EXTERNAL_SHA}

def prior_v8(root,model_association):
 import owned_hc_device_rs_experiment_v3 as old
 old.prior_header(root)
 from postboot_rms37_v8_historical_reader_v3 import finalized_binding
 binding=finalized_binding(root,model_association)['original_binding']
 require(binding['report_sha256']==old.REPORT_SHA and canonical(binding)==canonical(read_unique(old.EXTERNAL)),'Historical V8 contradicts original external admission')
 return {'root':str(old.PRIOR_ROOT),'binding':binding,'external_sha256':old.EXTERNAL_SHA,'captured_values_output_targets_only':True}

def closed_HC(root,model_association):
 import owned_layer3_qsa_runtime_contract_v4 as old
 from postboot_owned_hc_v3_historical_reader_v3 import finalized_binding
 root=Path(root).resolve();require(sha(root/'report.json')==old.HC_REPORT_SHA,'Exact admitted HC V3 report required')
 binding=finalized_binding(root,model_association)['original_binding']
 require(binding['report_sha256']==old.HC_REPORT_SHA and binding['reference_first_gate']['passed'] is True and binding['prefix4_attempted'] is True and binding['full_model_math_qualified'] is False,'Historical HC component scope differs')
 return {'root':str(root),'report_sha256':old.HC_REPORT_SHA,'binding':binding,'current_binding_canonical':canonical(binding),'native_values_used_as_inputs':False,'QSA_math_authority_transferred':False}
