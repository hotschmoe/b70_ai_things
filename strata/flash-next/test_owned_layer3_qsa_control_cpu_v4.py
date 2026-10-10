"""Synthetic CPU producer/leaf evidence controls; no runtime/model actions."""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import owned_layer3_qsa_control_v4 as control
import owned_layer3_qsa_runtime_contract_v4 as runtime
import qualify_owned_layer3_qsa_control_v4 as qualifier
import qualify_owned_layer3_qsa_producer_v4 as producer
from owned_layer3_qsa_work_v4 import metadata_binding, META_KEYS
import owned_layer3_qsa_work_v4 as work_module
import full48_owned_layer3_qsa_projection_v4 as composition
from qsa_owned_state_storage_v1 import QsaOwnedState
from owned_layer3_qsa_evidence_v4 import artifacts


class Controls(unittest.TestCase):
    def markers(self):
        return ('OWNQSA37_DEVICE backend=level_zero affinity=0 vendor=Intel driver=fixture name=B70\n'
                'OWNQSA37_FP_CONFIG flags=1,2, observed_device_flags_only=1 compiler_lowering_unobserved=1\n'
                'OWNQSA37_CONFIG rope=none freq_base=10000000 factor=1 freq_scale_in=0 orig_ctx=262144 '+
                'ext_factor=0 attn_factor=1 beta_fast=32 beta_slow=1 epsilon=source_qsa_rms_eps '+
                'native_flags=1,1,1 actual_resolved_checked=1\n'+
                'OWNQSA37_INPUTS files=10 own_projection_only=1 input_echo_bound=1\n'+
                ''.join('OWNQSA37_FRAME route='+str(r)+' graph_replay='+str(r)+
                        ' fields=14 own_zero_restored=1 windows=2,1,1\n' for r in range(3))+
                'OWNQSA37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 '+
                'internal_norm_argument_observed=0 internal_scores_softmax_observed=0 whole_model_math_qualified=0\n')

    def test_actual_shaped_marker_roster_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'runtime.log';p.write_text(self.markers())
            self.assertTrue(control.native_trace(p)['actual_graph_retired_and_owned_free_observed'])
            for bad in (self.markers()+self.markers(),self.markers().replace('graph_replay=2','graph_replay=1'),
                        self.markers().replace('actual_resolved_checked=1','actual_resolved_checked=0'),
                        self.markers()+'OWNQSA37_ERROR fixture\n',self.markers().replace('graph_retired=1','graph_retired=0'),
                        self.markers().rstrip('\n')):
                p.write_text(bad)
                with self.assertRaises(ValueError):control.native_trace(p)

    def outputs(self, root):
        echo=root/'input_echo';echo.mkdir()
        for field,size in control.INPUT_EXTENTS.items():
            (echo/field).write_bytes(control.CONFIG.encode('ascii') if field=='config.txt' else bytes(size))
        for route in range(3):
            p=root/('route'+str(route));p.mkdir()
            for field,(dtype,shape) in control.FIELDS.items():
                (p/field).write_bytes(np.zeros(shape,dtype=dtype).tobytes())

    def test_full_raw_three_route_and_candidate_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'output';out.mkdir();self.outputs(out)
            identity=root/'identity.json';identity.write_text('{}')
            (root/'reference-work').mkdir();(root/'report.json').write_text(json.dumps({'original_work_config':{'model_identity':str(identity)}}))
            (root/'reference-work/work-report.json').write_text(json.dumps({'own_layer3_projection':{}}))
            proof={'root':str(root)};fixture={'record':{'own_producer_binding':proof,'own_producer_root':str(root),
                'files':{name:{'bytes':size,'sha256':control.digest((out/'input_echo'/name).read_bytes())} for name,size in control.INPUT_EXTENTS.items()}}}
            own={k:np.zeros(control.FIELDS[k][1],dtype='<f4') for k in ('q_RoPE.f32','k_RoPE.f32','iq_RoPE.f32','gated.f32')}
            own.update(final_tail=np.zeros((3,128)),final_dead=np.zeros(128),final_pooled=np.zeros((2,128)))
            with patch.object(control,'candidate_arrays',return_value=own):
                result=control.compare_routes(out,fixture,proof)
                self.assertEqual(result['field_count'],14);self.assertIsNone(result['tolerance_gate'])
                self.assertTrue(all(r['bitwise_equal'] for r in result['own_inherited_candidate_comparisons'].values()))
                echo=out/'input_echo/qfull.f32';original=echo.read_bytes();bad=bytearray(original);bad[0]=1;echo.write_bytes(bad)
                with self.assertRaises(ValueError):control.compare_routes(out,fixture,proof)
                echo.write_bytes(original)
                field=out/'route1/q_RoPE.f32';raw=bytearray(field.read_bytes());raw[0]=1;field.write_bytes(raw)
                with self.assertRaises(ValueError):control.compare_routes(out,fixture,proof)

    def test_wrong_output_roster_finite_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.outputs(root)
            fixture={'record':{'own_producer_binding':{},'own_producer_root':str(root),
                'files':{name:{'bytes':size,'sha256':control.digest((root/'input_echo'/name).read_bytes())} for name,size in control.INPUT_EXTENTS.items()}}}
            (root/'route0/extra').write_text('fixture')
            with self.assertRaises(ValueError):control.compare_routes(root,fixture,{})
            (root/'route0/extra').unlink();(root/'route0/q_normalized.f32').write_bytes(np.full((4,24,256),np.nan,dtype='<f4').tobytes())
            with self.assertRaises(ValueError):control.compare_routes(root,fixture,{})

    def test_exact_artifact_tree_mutation_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'report.json').write_text('{}');(root/'field').write_bytes(b'own')
            report={'artifact_sha256':{'field':control.digest(b'own')}}
            self.assertEqual(artifacts(root,report),report['artifact_sha256'])
            (root/'extra').write_bytes(b'new')
            with self.assertRaises(ValueError):artifacts(root,report)
            (root/'extra').unlink();(root/'link').symlink_to(root/'field')
            with self.assertRaises(ValueError):artifacts(root,report)

    def test_original_metadata_defaults_and_foreign_scaling_refusal(self):
        meta={META_KEYS[0]:{'value':float(np.float32(1e-6))}}
        p=SimpleNamespace(reader=SimpleNamespace(files=[{'metadata':meta}]))
        self.assertEqual(metadata_binding(p)['effective_RopeScaling']['type'],'none')
        meta[META_KEYS[2]]={'value':'linear'}
        with self.assertRaises(ValueError):metadata_binding(p)

    def test_actual_compile_blocks_and_foreign_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'build').mkdir()
            names=('native_qsa.dp.cpp','native_rope.dp.cpp','native_qsa_indexer.dp.cpp','qsa_decode_attn.dp.cpp','qsa.dp.cpp')
            text='\n\n'.join('build CMakeFiles/strata_kernels.dir/src/kernels/cuda/'+name+'.o: fixture\n  FLAGS = precise\n  DEFINES = fixture' for name in names)
            p=root/'build/build.ninja';p.write_text(text)
            fields={'FLAGS':'precise','DEFINES':'fixture','production_device_LINK_FLAGS':'linked-precise'}
            self.assertEqual(len(runtime.qsa_builder_flags(root,fields)['objects']),5)
            p.write_text(text.replace('FLAGS = precise','FLAGS = fast',1))
            with self.assertRaises(ValueError):runtime.qsa_builder_flags(root,fields)

    def test_full_owned_recipe_image_resources_flags_and_mounts(self):
        with patch.object(qualifier,'sha',return_value='a'*64),patch.object(qualifier.os,'stat',return_value=SimpleNamespace(st_gid=123)):
            recipe=qualifier.docker_recipe('owned','image',[('/tmp/input','/inputs','ro')],
                {'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'},'fixture')
            obj={'Name':'/owned','Image':'image','Config':{'Image':'image','Labels':{'b70.qsa37.plan':'a'*64},
                'User':'1000:1000','Entrypoint':['/bin/bash'],'Cmd':['-c','fixture'],'OpenStdin':False,'Tty':False,
                'Env':['ZE_AFFINITY_MASK=0','ONEAPI_DEVICE_SELECTOR=level_zero:gpu']},
                'HostConfig':{'GroupAdd':['123'],'NetworkMode':'none','Privileged':False,'Memory':2<<30,'MemorySwap':2<<30,
                'NanoCpus':2000000000,'PidsLimit':256,'DeviceRequests':[],
                'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}]},
                'Mounts':[{'Source':'/tmp/input','Destination':'/inputs','RW':False,'Type':'bind'}]}
            qualifier.observed_container(obj,'owned','image',recipe)
            bad=copy.deepcopy(obj);bad['Image']='foreign'
            with self.assertRaises(ValueError):qualifier.observed_container(bad,'owned','image',recipe)
            bad=copy.deepcopy(obj);bad['HostConfig']['NanoCpus']=0
            with self.assertRaises(ValueError):qualifier.observed_container(bad,'owned','image',recipe)
            bad=copy.deepcopy(obj);bad['Config']['Env'].append('STRATA_NO_NORM_ROPE=0')
            with self.assertRaises(ValueError):qualifier.observed_container(bad,'owned','image',recipe)
            bad=copy.deepcopy(obj);bad['Mounts'][0]['Destination']='/model'
            with self.assertRaises(ValueError):qualifier.observed_container(bad,'owned','image',recipe)

    def test_actual_recipe_derivation_and_both_snapshot_sources(self):
        module=SimpleNamespace(SDK=Path('/sdk'),IMAGE='compiler',
            compile_argv=lambda:['icpx','/leaf/native_rms_rsqrt37_gpu_v1.cpp','-o','/out/native-rms-rsqrt37'])
        with patch.object(qualifier,'modules',return_value=(module,)),patch.object(qualifier,'sha',return_value='a'*64),patch.object(qualifier.os,'stat',return_value=SimpleNamespace(st_gid=123)):
            argv,build,run=qualifier.expected_recipes(Path('/new'),' /unused'.strip(),12)
            self.assertIn('/leaf/owned_layer3_qsa_gpu_v4.cpp',argv)
            self.assertIn('/out/owned-layer3-qsa37',argv)
            self.assertNotIn('-i',run)
            self.assertIn('--maps /out/qsa37.after',run[-1])
            self.assertIn('--inputs /inputs --output /out/native-output',run[-1])
            self.assertIn('--cpus',build)
        self.assertNotIn('args.',inspect.getsource(qualifier.finalized_binding))
        self.assertIn('owned_layer3_qsa_work_v4',inspect.getsource(producer.finalized_binding))

    def test_transplanted_strict_receipt_and_publisher_guard_functions(self):
        import qualify_native_rms_rsqrt37_v8 as original
        for name in ('command_binding','terminal_receipt_binding','runtime_receipt_binding','chronology'):
            self.assertEqual(inspect.getsource(getattr(qualifier,name)),inspect.getsource(getattr(original,name)))
        self.assertIn('cleanup_exact_phase',inspect.getsource(qualifier.main))
        self.assertIn('active_commands',inspect.getsource(qualifier.main))

    def test_failed_original_producer_refused_before_source_or_GPU(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'report.json').write_text('{"passed":false}')
            argv=['qualifier','--fixture',str(root),'--prepared',str(root),'--output',str(root/'out'),'--own-producer-root',str(root)]
            with patch('sys.argv',argv),patch.object(qualifier,'source_binding',side_effect=AssertionError('boundary')):
                with self.assertRaises(ValueError):qualifier.main()

    def test_producer_requires_prefix_before_any_dependency_or_GPU(self):
        argv=['producer','--fixture','/fixture','--prepared','/prepared','--output','/output',
              '--prior-rms-root','/prior','--first-hc-native-root','/first','--model-identity','/identity',
              '--bulk-build-root','/bulk','--hc-prerequisite-root','/hc']
        with patch('sys.argv',argv),patch.object(producer,'source_binding',side_effect=AssertionError('boundary')):
            with self.assertRaises(ValueError):producer.main()

    def test_real_shaped_phase0_first_gate_then_phase1_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=work_module.OriginalWork.__new__(work_module.OriginalWork)
            obj.output=Path(directory);obj.identity=Path('/synthetic-identity');obj.bulk_root=Path('/synthetic-bulk')
            obj.prefix_root=Path('/synthetic-target');obj.prefix_plan={'prefixes':{'4':[248045,8678,198,15666]}}
            provider=SimpleNamespace(reader=SimpleNamespace(files=[{'metadata':{META_KEYS[0]:{'value':float(np.float32(1e-6))}}}]))
            scope={};created=[]
            def model(*args):
                root=args[8];row=SimpleNamespace(qsa={3:SimpleNamespace(output=root,own_records=[],scope=lambda:scope)},
                    own_candidate_outputs=[],own_candidate_final_state={})
                created.append(row);return row
            def base_call(this,device):
                # The actual source computes one first HC only before prefix4.
                this.make_model({'args':[]},{},Path('/source'),device)
                self.assertEqual(this._models[0][0].qsa[3].own_records,[])
                this.make_model({'args':[]},{},Path('/source'),device)
                created[1].qsa[3].own_records=[{'synthetic_position':p} for p in range(4)]
                return {'first_gate':{'passed':True},'prefix4_attempted':True}
            with patch.object(work_module,'BoundOriginalFull48Rows',return_value=provider),patch.object(work_module,'sha',return_value='a'*64),patch.object(work_module,'OwnedLayer3ProjectionProducer',side_effect=model),patch.object(work_module.BaseWork,'__call__',base_call),patch.object(work_module,'recollect',return_value=[]):
                result=obj(None)
            self.assertEqual(str(created[0].qsa[3].output),str(Path(directory)/'own-qsa-phase0'))
            self.assertEqual(str(created[1].qsa[3].output),str(Path(directory)/'own-qsa-phase1'))
            self.assertEqual(len(result['own_layer3_projection']['records']),4)
            obj.prefix_plan['prefixes']['4']=[248045.,8678,198,15666]
            with patch.object(work_module.BaseWork,'__call__',side_effect=AssertionError('expensive')):
                with self.assertRaises(ValueError):obj(None)

    def test_first_gate_failure_never_grants_projection_record(self):
        obj=work_module.OriginalWork.__new__(work_module.OriginalWork);obj.prefix_root=Path('/target')
        obj.prefix_plan={'prefixes':{'4':[248045,8678,198,15666]}}
        with patch.object(work_module.BaseWork,'__call__',return_value={'first_gate':{'passed':False},'prefix4_attempted':False}):
            with self.assertRaises(ValueError):obj(None)

    def test_real_owned_four_cell_state_candidate_snapshot_shape(self):
        state=QsaOwnedState(max_cells=8)
        for pos in range(4):
            state.advance(pos,np.zeros((24,256)),np.zeros((2,256)),np.zeros((2,256)),
                          np.zeros((24,256)),np.zeros(128),np.zeros((4,128)))
        with tempfile.TemporaryDirectory() as directory:
            obj=composition.OwnedLayer3ProjectionProducer.__new__(composition.OwnedLayer3ProjectionProducer)
            obj.identity='a'*64;obj.p=SimpleNamespace(actual_source=False)
            fake=Path(directory)/'gated.f32';fake.write_bytes(bytes(6144*4))
            obj.qsa={3:SimpleNamespace(output=Path(directory),own_records=[
                {'projections':[{'own_input_snapshot':{'path':str(fake)}}]} for _ in range(4)])}
            with patch.object(composition.BaseProducer,'tokens',return_value={'qsa_states_owned':{3:state}}):
                obj.tokens([248045,8678,198,15666])
            self.assertEqual(obj.own_candidate_final_state['pooled']['shape'],[2,128])
            self.assertEqual(obj.own_candidate_final_state['block_pos'],0)
            self.assertEqual(len(obj.own_candidate_outputs),4)


if __name__=='__main__':unittest.main()
