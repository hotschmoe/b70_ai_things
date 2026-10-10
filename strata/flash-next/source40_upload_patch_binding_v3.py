"""Bind required upload patch to actual corrected SDK before oracle admission."""
ENGINE_SHA='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
PATCH='strata/flash-next/patches/0040-default-off-native-layer3-qsa-target-observer-v2.patch'
PATCH_SHA='3a4e239c81635fc33651944648a4a621e36428789a409fbe29a39e0f8a3df0b8'
def require(ok,message):
 if not ok:raise ValueError(message)
def binding(upload,engine):
 require(type(upload)is dict and type(engine)is dict and {'source_generation','engine_source_plan','required_new_patch','required_new_patch_sha256'}<=set(upload) and {'patches','expected_patched_source_sha256','added_header_payloads','build_targets','runtime_python_sources'}<=set(engine),'Complete actual upload/SDK required-patch metadata required')
 require(type(upload['source_generation'])is int and upload['source_generation']==40 and upload['engine_source_plan']=={'path':'strata/flash-next/native-layer3-qsa-observer-engine-build-plan-v2.json','sha256':ENGINE_SHA},'Actual corrected source40 engine association required')
 require(upload['required_new_patch']==PATCH and upload['required_new_patch_sha256']==PATCH_SHA,'Upload must require corrected0040V2; old patch cannot transfer')
 require(len(engine['patches'])==40 and len(engine['expected_patched_source_sha256'])==67 and len(engine['added_header_payloads'])==31 and len(engine['build_targets'])==8 and len(engine['runtime_python_sources'])==6 and engine['patches'][-1]=={'path':PATCH,'sha256':PATCH_SHA},'Actual complete corrected SDK patch/ABI/header/source roster differs')
 return {'actual_required_patch_matches_corrected_SDK':True,'required_patch':PATCH,'required_patch_sha256':PATCH_SHA,'old_upload_plan_or_oracle_transferred':False}
