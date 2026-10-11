"""Exact output-only host bridge over frozen actual clock source V2; no build."""
from pathlib import Path
import difflib,hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-clock-source-v2')
PLAN=ROOT/'strata/flash-next/pti-unitrace-clock-export-source-plan-v2.json'
EXPECTED=json.loads(PLAN.read_bytes())['expected_reconstructed_source_sha256']
HEADER='tools/unitrace/src/pti_unitrace_clock_export_v1.h'
ZE='tools/unitrace/src/levelzero/ze_collector.h'
TIMER='tools/unitrace/src/unitimer.h'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
HELPERS='''
inline bool BridgeEnabled() {
  const char* flag = std::getenv("UNITRACE_HOST_CLOCK_BRIDGE");
  if (flag == nullptr || flag[0] == '\\0' || std::strcmp(flag, "0") == 0) return false;
  if (std::strcmp(flag, "1") != 0 || !Enabled()) Fail("host bridge requires exact1 and metadata output");
  return true;
}
inline uint64_t ReadHostClock(clockid_t clock) {
  timespec value{};
  if (::clock_gettime(clock, &value) != 0 || value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L)
    Fail("host bridge CPU clock read failed");
  const uint64_t seconds = static_cast<uint64_t>(value.tv_sec);
  const uint64_t nanoseconds = static_cast<uint64_t>(value.tv_nsec);
  if (seconds > (UINT64_MAX - nanoseconds) / 1000000000ULL) Fail("host bridge CPU clock overflow");
  return seconds * 1000000000ULL + nanoseconds;
}
struct HostBridgeSample { uint64_t raw_before, monotonic, raw_after; };
inline HostBridgeSample ReadHostBridge() {
  HostBridgeSample sample{ReadHostClock(CLOCK_MONOTONIC_RAW), 0, 0};
  sample.monotonic = ReadHostClock(CLOCK_MONOTONIC);
  sample.raw_after = ReadHostClock(CLOCK_MONOTONIC_RAW);
  if (sample.raw_after < sample.raw_before) Fail("host bridge RAW clock reordered");
  return sample;
}
'''
AFTER_WRITER='''
inline void HostBridge(uint64_t site) {
  if (!BridgeEnabled()) return;
  const auto sample = ReadHostBridge();
  Emit("host_clock_bridge", {{"site", site}, {"raw_before_ns", sample.raw_before},
       {"monotonic_ns", sample.monotonic}, {"raw_after_ns", sample.raw_after}});
}
struct QueryBracket { bool enabled; uint64_t raw_before; };
inline QueryBracket BeginQueryBracket() {
  if (!BridgeEnabled()) return {false, 0};
  return {true, ReadHostClock(CLOCK_MONOTONIC_RAW)};
}
inline void EndQueryBracket(const QueryBracket& bracket, uint64_t site, uint64_t device,
                            uint64_t status, uint64_t host_ns, uint64_t device_ticks) {
  if (!bracket.enabled) return;
  // First operation after returning from the original GPU query is this CPU read.
  const uint64_t raw_after = ReadHostClock(CLOCK_MONOTONIC_RAW);
  if (!BridgeEnabled()) Fail("host bridge flag changed during original query");
  if (raw_after < bracket.raw_before) Fail("original timestamp query RAW bracket reordered");
  Emit("global_timestamp_query_bracket", {{"query_site", site}, {"device_handle", device},
       {"return_code", status}, {"returned_pair_present", uint64_t(status == 0)},
       {"host_ns", host_ns}, {"device_ticks", device_ticks},
       {"raw_before_ns", bracket.raw_before}, {"raw_after_ns", raw_after}});
  HostBridge(100 + site);
}
'''
def changes():
 for name,digest in EXPECTED.items():require(sha(BASE/name)==digest,'Frozen clock source changed '+name)
 h=(BASE/HEADER).read_text();z=(BASE/ZE).read_text();t=(BASE/TIMER).read_text()
 require(h.count('#include <unistd.h>')==1,'header include anchor');h=h.replace('#include <unistd.h>','#include <unistd.h>\n#include <time.h>')
 marker='inline uint64_t DoubleBits(double value) {';require(h.count(marker)==1,'helper anchor');h=h.replace(marker,HELPERS+marker)
 old='  ~Writer() { if (fd >= 0 && ::close(fd) != 0) Fail("final close failed"); }'
 close='''  ~Writer() {
    if (fd >= 0) {
      if (BridgeEnabled()) {
        const auto sample = ReadHostBridge();
        Emit("host_clock_bridge", {{"site", 2}, {"raw_before_ns", sample.raw_before},
             {"monotonic_ns", sample.monotonic}, {"raw_after_ns", sample.raw_after}});
      }
      if (::close(fd) != 0) Fail("final close failed");
    }
  }'''
 require(h.count(old)==1,'close anchor');h=h.replace(old,close)
 marker='} // namespace b70_pti_clock_v1';require(h.count(marker)==1,'namespace anchor');h=h.replace(marker,AFTER_WRITER+marker)
 token=re.compile(r'(?m)^([ \t]*)(?:ze_result_t )?status = ZE_FUNC\(zeDeviceGetGlobalTimestamps\)\(([^,]+), &(\w+), &(\w+)\);$')
 matches=list(token.finditer(z));require(len(matches)==6,'Exact six original query sites required')
 for site,match in reversed(list(enumerate(matches,1))):
  indent,device,host,ticks=match.groups();require('\n'not in indent,'single source line indent')
  original=match.group(0);before=indent+'const auto b70_bridge_query_'+str(site)+' = b70_pti_clock_v1::BeginQueryBracket();\n'
  after='\n'+indent+'b70_pti_clock_v1::EndQueryBracket(b70_bridge_query_'+str(site)+', '+str(site)+', reinterpret_cast<uintptr_t>('+device+'), static_cast<uint64_t>(status), status == ZE_RESULT_SUCCESS ? '+host+' : 0, status == ZE_RESULT_SUCCESS ? '+ticks+' : 0);'
  z=z[:match.start()]+before+original+after+z[match.end():]
 marker='''          {"monotonic_raw_minus_tai_i64_bits", static_cast<uint64_t>(tai_monotonic_raw_time_diff_)}});'''
 require(t.count(marker)==1,'host origin exact anchor');t=t.replace(marker,marker+'\n        b70_pti_clock_v1::HostBridge(1);')
 return {HEADER:h,ZE:z,TIMER:t}
def patch_text():
 result=[]
 for name,body in changes().items():result.extend(difflib.unified_diff((BASE/name).read_text().splitlines(True),body.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 return ''.join(result)
def interval(sample):
 require(set(sample)=={'raw_before_ns','monotonic_ns','raw_after_ns'},'Exact CPU bridge sample fields')
 for value in sample.values():require(type(value)is int and 0<=value<(1<<64),'Typed original CPU clock values')
 before=sample['raw_before_ns'];after=sample['raw_after_ns'];mono=sample['monotonic_ns'];require(before<=after,'CPU RAW clock reordered')
 return {'monotonic_minus_raw_offset_interval_ns':[mono-after,mono-before],'bracket_width_ns':after-before,'interpolation_or_drift_bound_assumed':False}
if __name__=='__main__':print(patch_text(),end='')
