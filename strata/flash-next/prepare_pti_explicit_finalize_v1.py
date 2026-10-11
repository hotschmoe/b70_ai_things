"""Source-only fresh tool finalizer/clock-close overlay; never mutates base."""
import difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-clock-budget-source-v1');H='tools/unitrace/src/pti_unitrace_clock_export_v1.h';T='tools/unitrace/src/tracer.cc';GUARD='tools/unitrace/src/pti_model_finalize_guard_v1.hpp'
def require(ok,message):
    if not ok:raise ValueError(message)
def once(text,old,new):require(text.count(old)==1,'Exact source finalization point required '+old[:80]);return text.replace(old,new)
def changes():
    old={name:(BASE/name).read_text()for name in(H,T)};header=old[H]
    begin=header.index('  ~Writer() {');end=header.index('  void Emit(',begin)
    destructor=header[begin:end]
    replacement='''  bool closed = false;
  bool close_error = false;
  ~Writer() { if (!CloseOnce(false)) Fail("final close failed"); }
  bool CloseOnce(bool require_opened) {
    std::lock_guard<std::mutex> held(lock);
    if (closed) return !close_error;
    if (fd < 0) { closed=true;close_error=require_opened;return !close_error; }
    if (owner != ::getpid()) { close_error=true;return false; }
    if (BridgeEnabled()) {
      const auto sample=ReadHostBridge();
      EmitLocked("host_clock_bridge", {{"site",2},{"raw_before_ns",sample.raw_before},
          {"monotonic_ns",sample.monotonic},{"raw_after_ns",sample.raw_after}});
    }
    const int result=::close(fd);fd=-1;closed=true;close_error=result!=0;
    return !close_error;
  }
'''
    header=header[:begin]+replacement+header[end:]
    header=once(header,'    std::lock_guard<std::mutex> held(lock);\n    const pid_t pid', '''    std::lock_guard<std::mutex> held(lock);
    if (closed) Fail("metadata emission after explicit close");
    EmitLocked(kind,fields);
  }
  void EmitLocked(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
    const pid_t pid''')
    header=once(header,'inline void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {\n  if (!Enabled()) return;\n  static Writer writer;\n  writer.Emit(kind, fields);\n}', '''inline Writer& OwnedWriter() { static Writer writer; return writer; }
inline void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
  if (!Enabled()) return;
  OwnedWriter().Emit(kind,fields);
}
inline bool CloseOnce() {
  if (!Enabled() || !BridgeEnabled()) return false;
  return OwnedWriter().CloseOnce(true);
}''')
    tracer=once(old[T],'void CONSTRUCTOR Init(void) {','''void CONSTRUCTOR Init(void) {
  const char* model_role=std::getenv("UNITRACE_MODEL_FINALIZE_V1");
  if (model_role && std::strcmp(model_role,"0")!=0) {
    if (std::strcmp(model_role,"1")!=0 || !b70_pti_clock_v1::Enabled() || !b70_pti_clock_v1::BridgeEnabled())
      b70_pti_clock_v1::Fail("invalid explicit model finalizer role");
    b70_pti_finalize_v1::state().Initialize();
  }''')
    tracer='#include "pti_model_finalize_guard_v1.hpp"\n#include "pti_unitrace_clock_export_v1.h"\n'+tracer
    tracer=once(tracer,'void DESTRUCTOR Fini(void) {', '''extern "C" int b70_pti_model_finalize_v1(uint32_t version,uint64_t pid,uint64_t tid) noexcept {
  const char* role=std::getenv("UNITRACE_MODEL_FINALIZE_V1");
  if (!role || std::strcmp(role,"1")!=0) return 1;
  const int result=b70_pti_finalize_v1::state().Finalize(version,pid,tid,
      [] { Teardown(); }, [] { return b70_pti_clock_v1::CloseOnce(); });
  std::fprintf(stderr,"B70_PTI_MODEL_FINALIZE_V1 pid=%ld result=%d\\n",long(::getpid()),result);
  std::fflush(stderr);
  return result;
}

void DESTRUCTOR Fini(void) {''')
    return old,{H:header,T:tracer,GUARD:(HERE/'pti_model_finalize_guard_v1.hpp').read_text()}
def patch_text():
    old,new=changes();return ''.join(''.join(difflib.unified_diff(old.get(name,'').splitlines(True),text.splitlines(True),fromfile='a/'+name if name in old else '/dev/null',tofile='b/'+name))for name,text in new.items())
