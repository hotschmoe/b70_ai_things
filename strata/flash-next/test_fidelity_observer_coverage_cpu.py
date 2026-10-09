#!/usr/bin/env python3
"""Negative controls for required head/stage/layer coverage, no GPU or inference."""
import json
from pathlib import Path
from audit_fidelity_observer_coverage import coverage,source_audit,DEFAULT_SOURCE


def main():
    stages={0:(0,32),1:(32,48)}
    req={'pid':'123','request':'1','tokens':'2','ids':'11,12'}
    def vector(phase,layer,stage):
        lb,le=stages[stage]
        return dict(pid='123',request='1',phase=phase,layer=str(layer),pos='1',token='12',stage=str(stage),lb=str(lb),le=str(le))
    vectors=[vector('first_logits_before_sampler',-1,1)]+[vector('first_window_residual',l,0 if l<32 else 1) for l in range(48)]
    good={'requests':[req],'vectors':vectors,'skips':[]}
    results={}
    results['complete_first_window']=coverage(good,1,True,stages)['passed'];assert results['complete_first_window']
    for name,trace,expected,act,owners in [
        ('empty_log',dict(requests=[],vectors=[],skips=[]),1,True,stages),
        ('request_no_vectors',dict(requests=[req],vectors=[],skips=[]),1,True,stages),
        ('missing_layer',dict(requests=[req],vectors=vectors[:-1],skips=[]),1,True,stages),
        ('missing_head',dict(requests=[req],vectors=vectors[1:],skips=[]),1,True,stages),
        ('duplicate_layer',dict(requests=[req],vectors=vectors+[vectors[-1]],skips=[]),1,True,stages),
        ('missing_expected_request',good,2,True,stages),
        ('no_stage_contract',good,1,True,{}),
        ('wrong_stage_ranges',good,1,True,{0:(0,24),1:(24,48)}),
        ('zero_required_requests',dict(requests=[],vectors=[],skips=[]),0,True,stages),
    ]:
        rejected=not coverage(trace,expected,act,owners)['passed'];assert rejected,name;results[name+'_rejected']=rejected
    logits_only={'requests':[req],'vectors':vectors[:1],'skips':[]}
    assert coverage(logits_only,1,False,stages)['passed'];results['logits_only_explicit_scope']=True
    assert coverage(dict(requests=[req],vectors=[],skips=[]),1,True,stages,[1])['passed'];results['cancelled_request_explicit_unobserved']=True
    source=source_audit(DEFAULT_SOURCE);assert source['passed']
    receipt={'CONFIG':'actual built0013..17 source; synthetic metadata negative controls only','COMMAND':'python3 strata/flash-next/test_fidelity_observer_coverage_cpu.py','RESULT':results,'VERDICT':'PASS strict observed coverage negative controls; not GPU/copy execution/state proof','source_audit':source}
    Path(__file__).with_name('fidelity-observer-coverage-cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS strict head/stage/all48 layer coverage and cancellation negative controls; no GPU')

if __name__=='__main__':main()
