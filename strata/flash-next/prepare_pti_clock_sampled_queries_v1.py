"""New default-off output sampling; original GPU queries are byte-preserved."""
from pathlib import Path
import difflib,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-model-finalizer-source-v1')
HEADER='tools/unitrace/src/pti_unitrace_clock_export_v1.h'
COLLECTOR='tools/unitrace/src/levelzero/ze_collector.h'
EXPECTED_HEADER='8a3284a2ce792f61a3e40f35ec7cb783b09669a5d9d33c3f60e7b4513569087e'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
INSERT=r'''
inline bool SampledQueriesEnabled() {
  const char* flag=std::getenv("UNITRACE_CLOCK_QUERY_SAMPLE_512");
  if (flag==nullptr || flag[0]=='\0' || std::strcmp(flag,"0")==0) return false;
  if (std::strcmp(flag,"1")!=0 || !BridgeEnabled()) Fail("query sampling requires exact1 and bridge");
  return true;
}
struct QueryCounts {
  std::atomic<uint64_t> seen[7]{}, selected[7]{}, completed[7]{};
};
inline QueryCounts& Counts() { static QueryCounts counts; return counts; }
'''
NEW_BEGIN=r'''inline QueryBracket BeginQueryBracket(uint64_t site) {
  if (!BridgeEnabled()) return {false, 0};
  if (SampledQueriesEnabled()) {
    if (site<1 || site>6) Fail("query sampling site outside declared roster");
    auto& count=Counts();
    const uint64_t ordinal=count.seen[site].fetch_add(1)+1;
    // Initial root/subdevice origins are never sampled. Sites3..6 retain
    // first and every512th bracket; ALL original query calls still execute.
    if (site>2 && (ordinal-1)%512!=0) return {false,0};
    count.selected[site].fetch_add(1);
  }
  return {true, ReadHostClock(CLOCK_MONOTONIC_RAW)};
}'''
def original():
    values={n:(BASE/n).read_bytes() for n in(HEADER,COLLECTOR)}
    require(sha(values[HEADER])==EXPECTED_HEADER,'Exact accepted finalizer header required')
    return values

def changes():
    old=original();h=old[HEADER].decode('ascii');c=old[COLLECTOR].decode('utf-8')
    h=h.replace('#include <cstdint>','#include <cstdint>\n#include <atomic>',1)
    marker='inline uint64_t ReadHostClock(clockid_t clock) {'
    require(h.count(marker)==1,'Exact bridge insertion point required');h=h.replace(marker,INSERT+'\n'+marker,1)
    begin='inline QueryBracket BeginQueryBracket() {\n  if (!BridgeEnabled()) return {false, 0};\n  return {true, ReadHostClock(CLOCK_MONOTONIC_RAW)};\n}'
    require(h.count(begin)==1,'Exact old begin required');h=h.replace(begin,NEW_BEGIN,1)
    point='  HostBridge(100 + site);\n}'
    require(h.count(point)==1,'Exact end bracket required');h=h.replace(point,'  HostBridge(100 + site);\n  if (SampledQueriesEnabled()) Counts().completed[site].fetch_add(1);\n}',1)
    close='    if (BridgeEnabled()) {\n      const auto sample=ReadHostBridge();'
    summary='''    if (SampledQueriesEnabled()) {
      auto& count=Counts();
      for (uint64_t site=1;site<=6;++site) {
        const uint64_t seen=count.seen[site].load(), selected=count.selected[site].load();
        const uint64_t completed=count.completed[site].load();
        const uint64_t expected=site<=2?seen:(seen?1+(seen-1)/512:0);
        if (completed!=selected || selected!=expected || selected>seen) Fail("query sampling incomplete at close");
        EmitLocked("query_sampling_summary", {{"query_site",site},{"sample_every",site<=2?1:512},
            {"original_query_calls",seen},{"selected_brackets",selected},
            {"completed_brackets",completed},{"suppressed_brackets",seen-selected}});
      }
    }
'''
    require(h.count(close)==1,'Exact finalizer insertion point required');h=h.replace(close,summary+close,1)
    for site in range(1,7):
        line=f'const auto b70_bridge_query_{site} = b70_pti_clock_v1::BeginQueryBracket();'
        require(c.count(line)==1,'Exact query site required '+str(site));c=c.replace(line,line.replace('BeginQueryBracket()','BeginQueryBracket('+str(site)+')'),1)
    return {HEADER:h.encode('ascii'),COLLECTOR:c.encode('utf-8')}
def patch_text():
    before=original();after=changes()
    return ''.join(''.join(difflib.unified_diff(before[n].decode().splitlines(True),after[n].decode().splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in before)
def selected(site,ordinal):
    require(type(site)is int and 1<=site<=6 and type(ordinal)is int and ordinal>0,'Typed site/ordinal required')
    return site<=2 or (ordinal-1)%512==0
if __name__=='__main__':print(patch_text(),end='')
