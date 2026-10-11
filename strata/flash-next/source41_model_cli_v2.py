#!/usr/bin/env python3
"""Fresh source41 native model intake/recipe; never a source40 runtime view."""
import argparse,hashlib,json,os,time
from pathlib import Path
from source41_sdk_binding_v1 import SDK,binding as sdk_binding,require,sha
from admit_source_upload41_v1 import finalized_binding as upload_binding
from serial37_canonical_json_v3 import canonical,read_unique
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXTRA_PAGE=(33530449920,'d6143e2592b92f8780244fa33efe380b0df686cdeb8e445f2db9ac149b6db20b')


def recipes(metadata,config,inputs,sdk):
    from prepare_decode40_host_profile_v1 import build
    # This proves only the original argument/math/native serial contract.
    shape=build(metadata,config,inputs,'0'*64,'OFF')
    require(metadata['cards']==[0]and type(metadata['cards'][0])is int,'Initial constructor/model control is native0 only')
    output={}
    for mode in(0,1):
        env=dict(shape['native_env']);env['STRATA_QSA_INDEXER_MATERIALIZED_HALF']=str(mode)
        alias='qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-hc-ple-mtp0-source41-materializedhalf'+str(mode)+'-CLI-diagnostic-gpu0-ctx2048-prefill64-ple65536'
        argv=[str(SDK/'build/strata'),'--serve',*config['args']]
        output[str(mode)]={'mode':mode,'engine_root':str(SDK),'engine_receipt_sha256':sdk['engine_receipt_sha256'],'argv_host':argv,'argv_container':['/build/strata','--serve',*config['args']],
            'env':env,'input_requests':shape['requests'],'primary_future_API_alias':'hotschmoe-dd','research_record_id':alias,
            'API_server_started':False,'eval_registry_alias_claimed':False,'source40_runtime_proof_transferred':False,
            'expected_gate':'source41OFF vs separately actual source40 same-config IDs/finish; source41ON vs independent original fixed-input QSA fidelity',
            'full_model_math_qualified':False,'normal_model_graph_qualified':False,'shelf_qualified':False}
    require(output['0']['argv_container']==output['1']['argv_container'],'Only declared mode env may differ')
    a=dict(output['0']['env']);b=dict(output['1']['env']);a.pop('STRATA_QSA_INDEXER_MATERIALIZED_HALF');b.pop('STRATA_QSA_INDEXER_MATERIALIZED_HALF');require(a==b,'Model/math/observer environment drift')
    return output


def guard(shards,output,label):
    """Additional active buffered guardian; not a repair or complete identity."""
    import source_page_watchdog_v3 as old
    require(len(shards)==4 and Path(shards[2]).name.endswith('00003-of-00004.gguf'),'Exact original third shard guardian required')
    pages=(*old.KNOWN_PAGES,EXTRA_PAGE);value=old.inspect_pages(shards[2],pages);directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    # Preserve every actual observed view, including failure, before refusal.
    for row in value['rows']:
        path=directory/(label+'-'+str(row['offset'])+'.raw');require(not path.exists(),'Guardian evidence must be new');path.write_bytes(row.pop('data'));row['preserved_path']=str(path)
    value.update(schema=1,scope='three buffered source-page views only; additional guardian may populate cache/residency',full_model_identity_qualified=False,causal_attribution_qualified=False,no_DONTNEED_or_invalidation_or_source_write=True)
    path=directory/(label+'-pages.json');require(not path.exists(),'Guardian receipt must be new');path.write_text(json.dumps(value,indent=2)+'\n');require(value['passed'],'Original model page integrity failed; actual views preserved')
    return value


def prepare(template,inputs_path,upload,oracle,pack,masked,current_identity,out):
    from run_source41_model_cli_v1 import closure
    closure()
    require(not Path(out).exists(),'New source41 model plan output required')
    import prefill_chunk64_512_control_v2 as current40
    from qualify_indexer_materialized41_masked_v2 import finalized_binding as masked_binding
    metadata=current40.validate_prepared(Path(template).resolve());config=read_unique(Path(template)/'server-config.json')
    require(metadata['profile']=='chunk64','Declared2048/64/65536 onecard template required')
    sdk=sdk_binding();fresh_upload=upload_binding(upload,oracle,pack);component=masked_binding(masked)
    identity=read_unique(current_identity);lock=read_unique(HERE/'model-lock.json');expected=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')]
    require(identity['passed']is True and identity['lock_sha256']==sha(HERE/'model-lock.json')and identity['model_revision']==lock['revision']and len(identity['rows'])==len(expected)==4,'Fresh complete publisher identity required')
    for row,want in zip(identity['rows'],expected):
        path=ROOT/lock['destination']/want['path'];st=path.stat();sig=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        require(row['path']==str(path)and row['passed']is True and row['sha256']==row['expected_sha256']==want['sha256']and row['bytes']==want['size']and row['stat_before']==row['stat_after']==sig,'Current publisher byte/stat identity differs')
    inputs=read_unique(inputs_path);modes=recipes(metadata,config,inputs,sdk)
    result={'schema':2,'scope':'source41 native CLI constructor+model gates; no API/eval/registry serving authority','source41':sdk,'fresh41_upload':fresh_upload,'fresh41_masked_component':component,
        'upload_receipt':str(Path(upload).resolve()),'oracle_receipt':str(Path(oracle).resolve()),'pack_receipt':str(Path(pack).resolve()),'masked_root':str(Path(masked).resolve()),
        'intake_producer_sha256':sha(__file__),
        'template_path':str(Path(template).resolve()),'template_prepared_sha256':sha(Path(template)/'prepared.json'),'template_config_sha256':sha(Path(template)/'server-config.json'),
        'template_source40_software_only':True,'actual_model_identity_path':str(Path(current_identity).resolve()),'actual_model_identity_sha256':sha(current_identity),'inputs_path':str(Path(inputs_path).resolve()),'inputs_sha256':sha(inputs_path),
        'source41_argv_env':modes,'source41_buffer_layout_must_be_fresh':True,'required_followup':'new owned native processes + pre/post health/journal/full4/three-page guardian; OFF actual source40 comparison; ON original fidelity reader',
        'model_or_constructor_executed_by_preparer':False,'source40_alias_claimed_for_source41':False,'canonical_registry_changed':False,'full_model_math_qualified':False}
    Path(out).mkdir();(Path(out)/'plan.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source40-template',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    for key in('upload-receipt','oracle-receipt','pack-receipt','masked-root','current-identity','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.source40_template,a.inputs,a.upload_receipt,a.oracle_receipt,a.pack_receipt,a.masked_root,a.current_identity,a.output)

if __name__=='__main__':main()
