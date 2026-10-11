"""Default-off retention of paused regular-list descriptors; fresh tool only."""
import difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-sampled-query-source-v1')
PARENT=HERE/'pti-sampled-query-tool-build-proposal-v1.json'
HEADER='tools/unitrace/src/levelzero/ze_collector.h'
INSERT=r'''
  // Opt-in descriptor/event retention for SYCL graphs implemented as ordinary
  // non-immediate closed command lists. Collection/normalization submission
  // remains controlled by the original named pause/resume state.
  bool RetainPausedRegularCommandList(ze_command_list_handle_t list) {
    static const bool enabled=[] {
      const char* value=std::getenv("UNITRACE_RETAIN_PAUSED_REGULAR_LISTS");
      if(value==nullptr||value[0]=='\0'||std::strcmp(value,"0")==0)return false;
      if(std::strcmp(value,"1")!=0||!b70_pti_clock_v1::Enabled())
        b70_pti_clock_v1::Fail("paused regular-list retention requires exact1 and metadata");
      return true;
    }();
    if(!enabled)return false;
    if(options_.metric_query||options_.metric_stream)
      b70_pti_clock_v1::Fail("paused regular-list retention excludes metric modes");
    std::shared_lock<std::shared_mutex> held(command_lists_mutex_);
    const auto found=command_lists_.find(list);
    return found!=command_lists_.end()&&!found->second->immediate_;
  }
'''
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def original():
    value=json.loads(PARENT.read_bytes());raw=(BASE/HEADER).read_bytes();require(sha(raw)==value['expected_patched_source_sha256'][HEADER],'Exact sampled source parent required');return raw

def changed():
    raw=original();text=raw.decode('utf-8');marker='  bool ShouldCaptureAppend(ze_command_list_handle_t command_list,';require(text.count(marker)==1,'Exact append predicate required');text=text.replace(marker,INSERT+'\n'+marker,1)
    old='           WouldTriggerGraphCapture(num_wait_events, wait_events);';new='           WouldTriggerGraphCapture(num_wait_events, wait_events) ||\n           RetainPausedRegularCommandList(command_list);';require(text.count(old)==1,'Exact default append gates required');text=text.replace(old,new,1);return text.encode('utf-8')
def changes():return {HEADER:changed()}
def patch_text():return ''.join(difflib.unified_diff(original().decode().splitlines(True),changed().decode().splitlines(True),fromfile='a/'+HEADER,tofile='b/'+HEADER))
def proposal():
    value=json.loads(PARENT.read_bytes());value.update(status='SOURCE ONLY opt-in paused regular command-list descriptor retention; fresh tool/loaded fixture required',source_parent=str(BASE),source_parent_proposal_sha256=sha(PARENT.read_bytes()),patch={'path':'strata/flash-next/pti-paused-regular-lists-v1.patch','sha256':sha((HERE/'pti-paused-regular-lists-v1.patch').read_bytes())},expected_patched_source_sha256={HEADER:sha(changed())})
    value['expected_full_source_sha256'][HEADER]=sha(changed());value.update(paused_regular_list_retention_defaultOFF=True,additional_executed_paused_instrumentation_queries_unmeasured=True,no_extra_GPU_query_claim=False,existing_active_submission_and_normalization_code_unchanged=True,model_math_changed=False,loaded_paused_fixture_qualified=False);return value
if __name__=='__main__':print(patch_text(),end='')
