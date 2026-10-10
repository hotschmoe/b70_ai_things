"""Read-only actual HC prerequisite; no native activation input authority."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from owned_layer3_qsa_contract_v1 import require

HC_REPORT_SHA = '502566b296bb5809f01c5c5cd3b3938f460a7c862d61aae8573bdd7028983d2a'


def qsa_builder_flags(sdk, hc_fields):
    ninja = (Path(sdk)/'build/build.ninja').read_text()
    result = {}
    for name in ('native_qsa.dp.cpp','native_rope.dp.cpp','native_qsa_indexer.dp.cpp','qsa_decode_attn.dp.cpp','qsa.dp.cpp'):
        needle = 'build CMakeFiles/strata_kernels.dir/src/kernels/cuda/'+name+'.o:'
        require(ninja.count(needle) == 1, 'Exact actual QSA object compile target required '+name)
        block = ninja.split(needle,1)[1].split('\n\n',1)[0]
        fields = {line.strip().split(' = ',1)[0]:line.strip().split(' = ',1)[1]
                  for line in block.splitlines() if ' = ' in line}
        require(fields['FLAGS'] == hc_fields['FLAGS'] and fields['DEFINES'] == hc_fields['DEFINES'],
                'Actual linked QSA object flags/defines differ from admitted precise source '+name)
        result[name] = fields
    return {'objects':result, 'production_device_LINK_FLAGS':hc_fields['production_device_LINK_FLAGS'],
            'object_and_device_link_authority_distinct':True, 'compiler_lowering_observed':False}


def closed_HC(root):
    import qualify_owned_hc_device_rs_v3 as original
    root = Path(root).resolve()
    require(original.sha(root/'report.json') == HC_REPORT_SHA, 'Exact independently admitted HC V3 prerequisite required')
    binding = original.finalized_binding(root)
    require(binding['report_sha256'] == HC_REPORT_SHA and binding['reference_first_gate']['passed'] is True
            and binding['prefix4_attempted'] is True and binding['full_model_math_qualified'] is False,
            'Original HC V3 component scope differs')
    return {'root': str(root), 'report_sha256': HC_REPORT_SHA, 'binding': binding,
            'current_binding_canonical': canonical(binding),
            'native_values_used_as_inputs': False, 'QSA_math_authority_transferred': False}
