"""Original saved independent residuals; model identity explicitly associated."""
from native_rms_rsqrt37_proposal_v1 import *
from postboot_original_model_association_v1 import historical_stat
def saved_original_input(model_association):
 require(sha(OWN/'report.json')==OWN_REPORT_SHA,'Exact successful independent firstHC report required');r=read(OWN/'report.json');require(r['errors']==[] and r['provenance']['ids']==[248045] and r['provenance']['captured_inputs_used'] is False and r['provenance']['captured_state_or_routes_used'] is False and r['provenance']['full_model_math_qualified'] is False,'Independent firstHC embedding provenance differs')
 identity=r['post_original_identity'];require(identity['complete_four_publisher_hashes_verified'] is True and identity['current_stat_verified'] is True and sha(identity['path'])==identity['sha256'],'Saved original postCPU full4 association differs');proof=read(identity['path']);lock=read(HERE/'model-lock.json');require(proof['lock_sha256']==sha(HERE/'model-lock.json') and proof['model_revision']==lock['revision'],'Saved original publisher lock/revision differs')
 for row in proof['rows']:
  require(row['passed'] is True and row['stat_before']==row['stat_after'],'Original saved shard receipt changed');historical_stat(model_association,Path(row['path']),row['stat_after'])
 require(proof['passed'] is True and len(proof['rows'])==4 and proof['started']>=r['computation_terminal_epoch'] and r['initial_pages']['passed'] is True and r['final_pages']['passed'] is True,'Saved independent-input source/full4/pages chronology failed')
 arrays={}
 for name,shape in [('embedding_owned',[2560]),('residual_owned',[4,2560])]:
  b=r['owned_arrays'][name];path=OWN/('own-'+name+'.f32');require(b['path']==str(path) and b['shape']==shape and b['encoding']=='LE_F32' and not path.is_symlink() and sha(path)==b['sha256'] and path.stat().st_size==b['bytes']==4*np.prod(shape),'Exact own embedding/residual field association differs');arrays[name]=np.frombuffer(path.read_bytes(),dtype='<f4').reshape(shape).copy();require(np.isfinite(arrays[name]).all(),'Finite original own input required')
 require(np.array_equal(arrays['residual_owned'],np.broadcast_to(arrays['embedding_owned'],(4,2560))),'Own residual must broadcast own original embedding');return arrays['residual_owned'],{'original_report_sha256':OWN_REPORT_SHA,'original_post_full4':identity,'input_sha256':sha(OWN/'own-residual_owned.f32'),'captured_native_operand_used':False,'fresh_model_payload_read':False,'synthetic_norm':True,'synthetic_down_up':True,'full_model_math_qualified':False}
