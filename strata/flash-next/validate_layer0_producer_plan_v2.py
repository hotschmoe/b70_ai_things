"""CPU manifest consistency gate; no build, SDK, model or GPU operations."""
import hashlib
from pathlib import Path


def validate(plan):
    expected=plan['expected_patched_source_sha256'];seen=set()
    for row in plan['added_header_payloads']:
        path=row['path']
        if path in seen:raise ValueError('Duplicate consumed header '+path)
        seen.add(path)
        if row['sha256']!=expected.get(path):raise ValueError('Consumed header/final source contradiction: '+path)
    c=plan['layer0_capture_contract']
    if c.get('observed_fields')!=33 or c.get('source_value_fields')!=31 or c.get('derived_value_fields')!=2 or c.get('source_value_fields')+c.get('derived_value_fields')!=c.get('observed_fields'):raise ValueError('Raw/source versus DERIVED counts incomplete')
    if c.get('raw_fused_hidden_observed') is not False or c.get('full_model_math_qualified') is not False:raise ValueError('Unqualified raw hidden/math claim')
    return {'manifest_consistent':True,'source_value_fields':31,'derived_value_fields':2,'observed_fields':33,'raw_fused_hidden_observed':False,'full_model_math_qualified':False}
