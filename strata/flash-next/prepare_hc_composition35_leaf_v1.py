#!/usr/bin/env python3
"""Fresh source35 SDK composition leaf recipe; CPU metadata only."""
import argparse,json
from pathlib import Path
import prepare_hc_projection35_leaf_v2 as frozen
import hc_composition_arithmetic35_fixture_v1 as fixture
ROOT=frozen.ROOT;sha=frozen.sha;read=frozen.read;require=frozen.require
def corpus_contract(plan=None,manifest=None):
 c=fixture.corpus_counts();counts={'cases':c['cases'],'words':c['output_floats'],'negative_controls':c['negative_cases']}
 require(c=={'cases':4,'frames':36,'output_floats':3802788,'fields_per_frame':19,'negative_cases':2},'Canonical composition shape/count differs')
 if plan is not None:require(plan['computed_corpus_counts']==c and plan['expected_raw_cases']==counts['cases'] and plan['expected_raw_words']==counts['words'] and plan['negative_controls']==2 and plan['cases']==[x['id'] for x in fixture.fixtures()],'Actual plan composition count differs before GPU')
 if manifest is not None:fixture.count_binding(manifest)
 return counts
def prepare(engine):
 old=frozen.prepare(engine);fixture.source_binding();c=fixture.corpus_counts();plan=dict(old);source=ROOT/'strata/flash-next/hc_composition_arithmetic35_gpu_v1.cpp';argv=[str(x).replace('/leaf/hc_projection_arithmetic35_gpu_v1.cpp','/leaf/hc_composition_arithmetic35_gpu_v1.cpp').replace('/out/hc_projection_arithmetic35_gpu_v2','/out/hc_composition_arithmetic35_gpu_v1') for x in old['compile_argv_inside_pinned_image']]
 extra=Path(engine).resolve()/'source/sycl/include/strata/kernels/hc_native_composition.hpp';dependencies=list(old['current_SDK_dependencies'])+[{'path':str(extra),'sha256':sha(extra)}]
 plan.update(producer_sha256=sha(__file__),leaf_source=str(source),leaf_source_sha256=sha(source),collector_source_sha256=sha(Path(fixture.__file__)),fixture_source_plan_sha256=sha(fixture.SOURCE_PLAN),current_SDK_dependencies=dependencies,compile_argv_inside_pinned_image=argv,run_argv_inside_pinned_image=['/out/hc_composition_arithmetic35_gpu_v1','--inputs','/inputs','--output','/results/raw-new'],computed_corpus_counts=c,synthetic_geometry={'N':2560,'H':4,'L':320,'T':[1,2],'cases':4,'frames':36,'fields_per_frame':19,'output_words':c['output_floats'],'routes':['direct_queue','graph_replay1','same_graph_replay2'],'apply':['none','nonalias','alias']},cases=[x['id'] for x in fixture.fixtures()],expected_raw_cases=4,expected_raw_words=c['output_floats'],negative_controls=2,scope='Actual compiled synthetic HC composition + separate shadow/source intrinsic observations; no best-candidate policy, universal intrinsic or model qualification',normal_model_graph_qualified=False,full_HC_qualified=False)
 corpus_contract(plan);return plan
def main():
 p=argparse.ArgumentParser();p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New compile plan required');a.output.write_text(json.dumps(prepare(a.engine_root),indent=2)+'\n',encoding='ascii');print('CPU prepared; compile/GPU not executed')
if __name__=='__main__':main()
