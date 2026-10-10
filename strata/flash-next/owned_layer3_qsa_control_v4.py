"""Own producer input preparation and strict full-vector device recollection."""
import hashlib
import re
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique, canonical
from owned_layer3_qsa_contract_v1 import require
from owned_layer3_qsa_helper_contract_v3 import recollect, load_own
from owned_layer3_qsa_contract_v1 import norm256_argument

CONFIG = ('QSA37_OWN_INPUTS_V4\nids 248045 8678 198 15666\n'
          'geometry 24 2 256 4 128 64 4 2048\nrope none 10000000 1 0 262144 0 1 32 1\n'
          'epsilon source_qsa_rms_eps\nwindows 0:2 2:1 3:1\norigin independent_original_projection_only\n')
INPUT_EXTENTS = {'config.txt':210, 'qfull.f32':196608, 'k.f32':8192, 'v.f32':8192,
                 'raw.f32':2048, 'iq.f32':8192, 'qgamma.f32':1024, 'kgamma.f32':1024,
                 'iqgamma.f32':512, 'ikgamma.f32':512}
FIELDS = {'q_normalized.f32': ('<f4', (4,24,256)), 'q_RoPE.f32': ('<f4', (4,24,256)),
          'k_normalized.f32': ('<f4', (4,2,256)), 'k_RoPE.f32': ('<f4', (4,2,256)),
          'iq_normalized.f32': ('<f4', (4,4,128)), 'iq_RoPE.f32': ('<f4', (4,4,128)),
          'attention.f32': ('<f4', (4,24,256)), 'gated.f32': ('<f4', (4,24,256)),
          'indexer_tail_windows.f32': ('<f4', (3,3,128)),
          'indexer_dead_windows.f32': ('<f4', (3,128)),
          'indexer_pooled_windows.f32': ('<f4', (3,2,128)),
          'indexer_block_pos_windows.i32': ('<i4', (3,)),
          'K_pool_windows.f16': ('<f2', (3,1,2,4,256)),
          'V_pool_windows.f16': ('<f2', (3,1,2,4,256))}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def candidate_arrays(own, identity):
    require(len(own['own_candidate_outputs']) == 4, 'Exact four own candidate intermediate outputs required')
    out = {'q_RoPE.f32': [], 'k_RoPE.f32': [], 'iq_RoPE.f32': [], 'gated.f32': []}
    for position, record in enumerate(own['own_candidate_outputs']):
        require(type(record['position']) is int and record['position'] == position,
                'Exact own candidate position roster required')
        for field, shape in [('q_RoPE',(24,256)),('k_RoPE',(2,256)),('iq_RoPE',(4,128))]:
            value = load_own(record['arrays'][field], own['root'], shape,
                             'blk.3.own_candidate.'+field, position, identity, True)
            out[field+'.f32'].append(value)
        value = load_own(record['arrays']['gated'], own['root'], (6144,),
                         'blk.3.attn_output.weight.own_input', position, identity, True)
        out['gated.f32'].append(value.reshape(24,256))
    out = {name:np.stack(values).astype('<f4') for name,values in out.items()}
    final = own['own_candidate_final_state']
    for field, shape in [('tail',(3,128)),('dead',(128,)),('pooled',(2,128))]:
        out['final_'+field] = load_own(final[field], own['root'], shape,
                                       'blk.3.own_candidate.final_'+field, 3, identity, True)
    require(type(final['block_pos']) is int and final['block_pos'] == 0,
            'Exact bounded own indexer final position required')
    return out


def producer_binding(root):
    from qualify_owned_layer3_qsa_producer_v4 import finalized_binding
    root = Path(root).resolve(); binding = finalized_binding(root)
    require(binding['own_projection_producer_generation'] == 4 and binding['reference_first_gate']['passed'] is True
            and binding['prefix4_attempted'] is True, 'Genuine own original four-row producer required')
    return binding


def arrays(root):
    root = Path(root).resolve(); parent = read_unique(root/'report.json')
    work = read_unique(root/'reference-work/work-report.json'); own = work['own_layer3_projection']
    identity_path = Path(parent['original_work_config']['model_identity'])
    identity = digest(identity_path.read_bytes())
    require(canonical(parent['reference_phase']['work']) == canonical(work)
            and Path(own['root']).resolve() == root/'reference-work/own-qsa-phase1',
            'Original own producer/work/snapshot association changed')
    rows = recollect(own['records'], own['root'], identity, True)
    meta = own['metadata_binding']
    from owned_layer3_qsa_work_v4 import META_KEYS
    require(meta['original_metadata'][META_KEYS[0]]['value'] == float(np.float32(1e-6))
            and meta['captured_metadata_substitution_used'] is False
            and meta['effective_RopeScaling'] == {'type':'none','freq_base':1e7,'factor':1.,'freq_scale_in':0.,
                'orig_ctx':262144.,'ext_factor':0.,'attn_factor':1.,'beta_fast':32.,'beta_slow':1.},
            'Admitted original unscaled QSA metadata required')
    out = {'config.txt': CONFIG.encode('ascii')}
    for name, role in [('qfull','attn_q.weight'),('k','attn_k.weight'),('v','attn_v.weight'),
                       ('raw','indexer.k_proj.weight'),('iq','indexer.q_proj.weight')]:
        value = np.stack([r['projections'][role] for r in rows]).astype('<f4')
        if name in ('qfull','k','iq'):
            normalized_inputs = (value.reshape(4,24,2,256)[:,:,0,:].reshape(-1,256)
                                 if name == 'qfull' else value.reshape(-1,256 if name == 'k' else 128))
            for x in normalized_inputs:
                argument = norm256_argument(x, np.float32(1e-6))
                require(np.isfinite(argument['argument_F32']) and argument['argument_F32'] > 0,
                        'Own finite source norm reduction precondition failed')
        if name in ('v','raw'):
            require(np.isfinite(value.astype('<f2')).all(), 'Own source finite FP16 storage precondition failed')
        out[name+'.f32'] = value.tobytes()
    for name, role in [('qgamma','attn_q_norm.weight'),('kgamma','attn_k_norm.weight'),
                       ('iqgamma','indexer.q_norm.weight'),('ikgamma','indexer.k_norm.weight')]:
        values = [r['vectors'][role].tobytes() for r in rows]
        require(all(v == values[0] for v in values), 'Original gamma differs across own tokens')
        out[name+'.f32'] = values[0]
    return out


def prepare(producer_root, output):
    producer_root = Path(producer_root).resolve(); before = producer_binding(producer_root)
    raw = arrays(producer_root); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for name, value in raw.items():
        (output/name).write_bytes(value)
    record = {'schema':4, 'own_producer_root':str(producer_root), 'own_producer_binding':before,
              'files':{name:{'sha256':digest(value),'bytes':len(value)} for name,value in raw.items()},
              'captured_operands_or_values_used':False, 'actual_device_operation_observed':False,
              'full_model_math_qualified':False}
    require(producer_binding(producer_root) == before and arrays(producer_root) == raw,
            'Own producer changed during input preparation')
    import json
    (output/'input-binding.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def fixture_binding(root):
    root = Path(root).resolve(); saved = read_unique(root/'input-binding.json')
    require(saved['schema'] == 4 and saved['captured_operands_or_values_used'] is False,
            'Exact own input provenance required')
    producer = producer_binding(saved['own_producer_root']); raw = arrays(saved['own_producer_root'])
    require(producer == saved['own_producer_binding'] and saved['files'] == {
        name:{'sha256':digest(value),'bytes':len(value)} for name,value in raw.items()},
        'Current genuine producer/input binding changed')
    require({p.name for p in root.iterdir()} == set(raw)|{'input-binding.json'}, 'Exact helper own input roster required')
    for name,value in raw.items():
        path = root/name
        require(not path.is_symlink() and path.read_bytes() == value, 'Own helper input bytes differ '+name)
    return {'root':str(root),'fixture_sha256':digest((root/'input-binding.json').read_bytes()),'record':saved}


def native_trace(path):
    raw_lines = Path(path).read_text().splitlines(keepends=True)
    require(all(line.endswith('\n') for line in raw_lines if line.startswith('OWNQSA37_')),
            'Final actual native semantic rows require complete newline')
    lines = [line.rstrip('\r\n') for line in raw_lines]; markers = [x for x in lines if x.startswith('OWNQSA37_')]
    require(len(markers) == 8 and markers[0].startswith('OWNQSA37_DEVICE backend=level_zero affinity=0 vendor=')
            and ' driver=' in markers[0] and ' name=' in markers[0],
            'Exact owned QSA device/marker roster required')
    require(re.fullmatch(r'OWNQSA37_FP_CONFIG flags=(?:[0-9]+,)+ observed_device_flags_only=1 compiler_lowering_unobserved=1',markers[1]),
            'Actual FP capability roster with no lowering claim required')
    require(markers[2] == 'OWNQSA37_CONFIG rope=none freq_base=10000000 factor=1 freq_scale_in=0 '+
            'orig_ctx=262144 ext_factor=0 attn_factor=1 beta_fast=32 beta_slow=1 '+
            'epsilon=source_qsa_rms_eps native_flags=1,1,1 actual_resolved_checked=1',
            'Actual resolved source RopeScaling/native flags required')
    require(markers[3] == 'OWNQSA37_INPUTS files=10 own_projection_only=1 input_echo_bound=1',
            'Actual consumed own input echo marker required')
    for route in range(3):
        require(markers[route+4] == 'OWNQSA37_FRAME route='+str(route)+' graph_replay='+str(route)+
                ' fields=14 own_zero_restored=1 windows=2,1,1', 'Exact ordered actual QSA frame required')
    require(markers[7] == 'OWNQSA37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 '+
            'internal_norm_argument_observed=0 internal_scores_softmax_observed=0 whole_model_math_qualified=0',
            'Actual QSA graph/free terminal required')
    require(not any('OWNED_ERROR' in x or 'RMS37_ERROR' in x or 'OWNQSA37_ERROR' in x for x in lines),
            'Actual native helper error cannot qualify')
    return {'actual_marker_roster':markers,'actual_graph_retired_and_owned_free_observed':True,
            'device_PCI_identity_observed':False}


def compare_routes(output, fixture, producer):
    output = Path(output).resolve()
    require(fixture['record']['own_producer_binding'] == producer, 'Exact own producer/fixture join required')
    require({p.name for p in output.iterdir()} == {'route0','route1','route2','input_echo'}, 'Exact three-route/consumed-input output roster required')
    echo = output/'input_echo'; source_files = fixture['record']['files']
    require(not echo.is_symlink() and set(source_files) == set(INPUT_EXTENTS)
            and {p.name for p in echo.iterdir()} == set(INPUT_EXTENTS), 'Complete actual consumed-input echo required')
    for name,expected in source_files.items():
        path=echo/name;require(path.is_file() and not path.is_symlink(), 'Regular actual input echo required')
        raw=path.read_bytes()
        require(type(expected['bytes']) is int and len(raw)==expected['bytes']==INPUT_EXTENTS[name]
                and digest(raw)==expected['sha256'] and path.read_bytes()==raw,
                'Actual loaded own operand bytes differ '+name)
    require((echo/'config.txt').read_bytes()==CONFIG.encode('ascii'), 'Actual loaded config differs')
    all_fields = {}; direct_values = {}
    for field,(dtype,shape) in FIELDS.items():
        raw = []
        for route in range(3):
            directory = output/('route'+str(route))
            require(not directory.is_symlink() and {p.name for p in directory.iterdir()} == set(FIELDS),
                    'Exact native full-field output roster required')
            path = directory/field; require(path.is_file() and not path.is_symlink(), 'Regular actual output required')
            before = path.stat(); b = path.read_bytes(); value = np.frombuffer(b,dtype=dtype)
            require(len(b) == int(np.prod(shape))*np.dtype(dtype).itemsize and np.isfinite(value).all(),
                    'Exact finite actual QSA output extent required '+field)
            require(path.read_bytes() == b and path.stat() == before, 'Actual consumed output changed '+field)
            raw.append(b)
        require(raw[0] == raw[1] == raw[2], 'Full own QSA direct/replay output differs '+field)
        all_fields[field] = {'bytes':len(raw[0]),'shape':list(shape),'dtype':dtype,
                             'sha256':digest(raw[0]),'direct_two_graph_replays_bitwise':True}
        direct_values[field] = np.frombuffer(raw[0],dtype=dtype).reshape(shape).copy()
    producer_root = Path(fixture['record']['own_producer_root'])
    parent = read_unique(producer_root/'report.json')
    work = read_unique(producer_root/'reference-work/work-report.json')
    candidates = candidate_arrays(work['own_layer3_projection'],
                                  digest(Path(parent['original_work_config']['model_identity']).read_bytes()))
    comparisons = {}
    from explore_full48_original_prefix1_v4 import compare_observation
    for field in ('q_RoPE.f32','k_RoPE.f32','iq_RoPE.f32','gated.f32'):
        value = direct_values[field]
        comparisons[field] = compare_observation(candidates[field],value)
    for name,field in [('tail','indexer_tail_windows.f32'),('dead','indexer_dead_windows.f32'),
                       ('pooled','indexer_pooled_windows.f32')]:
        value = direct_values[field][-1]
        comparisons['final_'+name] = compare_observation(candidates['final_'+name],value)
    return {'fields':all_fields,'field_count':14,'actual_consumed_own_inputs_SHA_verified':True,
            'own_inherited_candidate_comparisons':comparisons,
            'candidate_internal_device_intrinsics_qualified':False,'tolerance_gate':None,
            'full_model_math_qualified':False,
            'native_internal_norm_argument_observed':False,'internal_score_softmax_observed':False,
            'captured_operand_substitution_used':False,'normal_model_graph_qualified':False}
