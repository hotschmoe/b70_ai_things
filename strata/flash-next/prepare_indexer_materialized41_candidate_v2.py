"""SOURCE ONLY exact source40 -> default-off source41 semantic candidate."""
import difflib,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo')
BASE_PLAN=HERE/'native-layer3-qsa-observer-engine-build-plan-v2.json'
BASE_SHA='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
PATCH=HERE/'patches/0041-qsa-indexer-materialized-half-candidate-v2.patch'
PLAN=HERE/'indexer-materialized41-engine-build-proposal-v2.json'
PATHS=['include/strata/kernels/native_qsa_indexer.hpp','include/strata/core/layer.hpp','sycl/include/strata/core/verify.hpp','sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp','sycl/src/core/layer.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp','sycl/src/program/generate.cpp','include/strata/core/verify.hpp']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def once(s,a,b):require(s.count(a)==1,'Exact source point differs '+a[:80]);return s.replace(a,b)
DECL='''// Source41 candidate. OFF returns raw without launches or scratch access.
// Owned scratch must survive graph replay; ignored device positions are zeroed
// without reading raw. This is a SYCL-only model integration, not a cache proof.
bool native_qsa_indexer_materialized_half_enabled();
const float* native_qsa_indexer_materialize_input(const float* raw,
    const int32_t* positions, int stride, int rows, int64_t p0, int64_t max_cells, int idx_dim,
    uint16_t* half_base, float* expanded_base, std::size_t scratch_offset,
    std::size_t scratch_capacity, void* stream);

'''
KERNEL='''bool native_qsa_indexer_materialized_half_enabled() {
    static const bool requested = [] {
        const char* value = std::getenv("STRATA_QSA_INDEXER_MATERIALIZED_HALF");
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("materialized indexer mode requires absent/0/1");
    }();
    if (requested && !native_qsa_indexer_enabled())
        throw std::invalid_argument("materialized indexer requires native indexer route");
    return requested;
}
const float* native_qsa_indexer_materialize_input(const float* raw,
    const int32_t* positions, int stride, int rows, int64_t p0, int64_t max_cells, int idx_dim,
    uint16_t* half_base, float* expanded_base, std::size_t scratch_offset,
    std::size_t scratch_capacity, void* stream) {
    if (!native_qsa_indexer_materialized_half_enabled()) return raw;
    if (idx_dim != D || rows < 1 || rows > INT32_MAX / D || stride < 0 || max_cells < 1 || max_cells > INT32_MAX ||
        (!positions && (p0 < 0 || p0 + rows > max_cells)) || !stream)
        throw std::invalid_argument("materialized indexer bounded input/position/stream required");
    const std::size_t count = std::size_t(rows) * D;
    if (scratch_capacity > SIZE_MAX / 4 || scratch_offset > scratch_capacity ||
        count > scratch_capacity - scratch_offset)
        throw std::invalid_argument("materialized indexer scratch capacity/offset exceeded");
    const Span owners[] = {{half_base,scratch_capacity*2},{expanded_base,scratch_capacity*4}};
    for (const auto& span : owners) validate(span);
    if (overlaps(owners[0],owners[1])) throw std::invalid_argument("materialized indexer owners overlap");
    // Null OFF arguments are never advanced. ON proves bounded owner extents
    // before performing typed pointer arithmetic within the actual arena.
    uint16_t* half_store = half_base + scratch_offset;
    float* expanded = expanded_base + scratch_offset;
    const Span spans[] = {{raw,count*4},{half_store,count*2},{expanded,count*4}};
    for (const auto& span : spans) validate(span);
    for (int a=0;a<3;++a) for (int b=a+1;b<3;++b)
        if (overlaps(spans[a],spans[b])) throw std::invalid_argument("materialized indexer scratch overlaps");
    if (positions) {
        const Span pos{positions,(std::size_t(rows-1)*std::size_t(stride)+1)*4};validate(pos);
        for (const auto& span : spans) if (overlaps(pos,span)) throw std::invalid_argument("materialized indexer positions overlap");
    }
    auto* queue = strata::q_of(stream);
    if (!queue->has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("materialized indexer requires ordered owned queue");
    static_assert(sizeof(sycl::half)==2 && sizeof(float)==4);
    auto* half = reinterpret_cast<sycl::half*>(half_store);
    queue->parallel_for<class Indexer41MaterializedStore>(sycl::range<1>(count),[=](sycl::id<1> item) {
        const std::size_t i=item[0],row=i/D;
        const int64_t pos=positions ? positions[row*std::size_t(stride)] : p0+int64_t(row);
        // Match the production append mask before touching rejected raw rows.
        if (pos>=0 && pos<max_cells)
            half[i]=sycl::vec<float,1>(raw[i]).convert<sycl::half,sycl::rounding_mode::rte>()[0];
        else half[i]=sycl::half(0.0f);
    });
    queue->parallel_for<class Indexer41MaterializedLoad>(sycl::range<1>(count),[=](sycl::id<1> item) {
        const std::size_t i=item[0];
        expanded[i]=sycl::vec<sycl::half,1>(half[i]).convert<float,sycl::rounding_mode::automatic>()[0];
    });
    return expanded;
}

'''
def split_args(body):
 args=[];depth=0;start=0
 for i,ch in enumerate(body):
  if ch=='(':depth+=1
  elif ch==')':depth-=1
  elif ch==','and depth==0:args.append(body[start:i].strip());start=i+1
 args.append(body[start:].strip());return args

def calls(source):
 found=[]
 for m in re.finditer(r'(?<!\w)(?:strata::kernels::)?native_qsa_indexer_append(?:_steps|_batch)?\(',source):
  depth=1;i=m.end()
  while depth:
   if source[i]=='(':depth+=1
   elif source[i]==')':depth-=1
   i+=1
  found.append((m.start(),i,m.group(0)[:-1],split_args(source[m.end():i-1])))
 return found

def replace_calls(source,kind):
 found=calls(source);require(len(found)=={'layer':1,'verify':5,'prefill':2}[kind],'Complete actual caller roster differs')
 for begin,end,name,args in reversed(found):
  raw=args[0]
  if name.endswith('_steps'):pos,stride,rows,p0,maxcells,stream=args[1],args[2],args[3],'0',args[9],args[11]
  elif name.endswith('_batch'):pos,stride,rows,p0,maxcells,stream='nullptr','0',args[1],args[2],args[8],args[10]
  else:pos,stride,rows,p0,maxcells,stream=args[1],'0','1','0',args[7],args[9]
  if kind=='layer':half,expanded,offset,capacity='b.idx_half_store','b.idx_expanded','0','std::size_t(g.idx_key_dim)'
  elif kind=='prefill':
   offset='std::size_t(t) * 128'if 't * 128'in raw else '0';half,expanded,capacity='m.idx_half_store','m.idx_expanded','std::size_t(T) * 128'
  else:
   if raw.startswith('idx_raw +'):offset='qi * MT * ID + '+raw.split('+',1)[1].strip()
   else:require(raw.startswith('idx_raw_L_ +'),'Exact retained commit raw source differs');offset=raw.split('+',1)[1].strip()
   half,expanded,capacity='idx_half_store_L_','idx_expanded_L_','std::size_t(g.n_qsa_layers()) * MT * ID'
  geom=args[8]if name.endswith('_steps')else args[7]if name.endswith('_batch')else args[6]
  helper='strata::kernels::native_qsa_indexer_materialize_input('+', '.join([raw,pos,stride,rows,p0,maxcells,geom+'.idx_dim',half,expanded,offset,capacity,stream])+')'
  args[0]=helper;source=source[:begin]+name+'('+', '.join(args)+')'+source[end:]
 return source

def reconstruct():
 require(sha(BASE_PLAN)==BASE_SHA,'Source40 engine prerequisite changed');before={n:(SDK/'source'/n).read_text()for n in PATHS};after=dict(before)
 n=PATHS[0];after[n]=once(after[n],'#pragma once','#pragma once\n#include <cstddef>');after[n]=once(after[n],'// Configure before session construction/capture.',DECL+'// Configure before session construction/capture.')
 n=PATHS[1];after[n]=once(after[n],'    float* idx_raw = nullptr;', '    uint16_t* idx_half_store = nullptr; // conditional arena scratch, never cache state\n    float* idx_expanded = nullptr;\n    float* idx_raw = nullptr;')
 n=PATHS[2];after[n]=once(after[n],'    float *qfull_ = nullptr,', '    uint16_t* idx_half_store_L_ = nullptr; // nQ * MT * ID, arena lifetime\n    float* idx_expanded_L_ = nullptr;\n    float *qfull_ = nullptr,')
 n=PATHS[3];after[n]=once(after[n],'#include <cmath>','#include <cmath>\n#include <cstdlib>\n#include <cstring>');after[n]=once(after[n],'void native_qsa_indexer_append_steps(',KERNEL+'void native_qsa_indexer_append_steps(')
 n=PATHS[4];s=after[n];s=once(s,'// attn_scratch\nreturn align_up16(n) + 256;}','// attn_scratch\nif (strata::kernels::native_qsa_indexer_materialized_half_enabled()) n += (uint64_t) g.idx_key_dim * 6 + 32;\nreturn align_up16(n) + 256;}');s=once(s,'b.idx_raw = c.take<float>((uint64_t) g.idx_key_dim);','b.idx_raw = c.take<float>((uint64_t) g.idx_key_dim); if (strata::kernels::native_qsa_indexer_materialized_half_enabled()) { b.idx_half_store = c.take<uint16_t>((uint64_t) g.idx_key_dim); b.idx_expanded = c.take<float>((uint64_t) g.idx_key_dim); }');after[n]=replace_calls(s,'layer')
 n=PATHS[5];s=once(after[n],'idx_raw_L_ = b.take<float>(nQ * T * ID);','idx_raw_L_ = b.take<float>(nQ * T * ID);\n        if (native_qsa_indexer_materialized_half_enabled()) { idx_half_store_L_ = b.take<uint16_t>(nQ * T * ID); idx_expanded_L_ = b.take<float>(nQ * T * ID); }');after[n]=replace_calls(s,'verify')
 n=PATHS[6];s=once(after[n],'    uint16_t* attn_h = nullptr;','    uint16_t* idx_half_store = nullptr;\n    float* idx_expanded = nullptr;\n    uint16_t* attn_h = nullptr;');s=once(s,'a.take<float>(T * 128, ok);','a.take<float>(T * 128, ok);\n    if (strata::kernels::native_qsa_indexer_materialized_half_enabled()) { a.take<uint16_t>(T * 128, ok); a.take<float>(T * 128, ok); }');s=once(s,'m.idx_raw = b.take<float>(T * 128, ok);','m.idx_raw = b.take<float>(T * 128, ok);\n        if (strata::kernels::native_qsa_indexer_materialized_half_enabled()) { m.idx_half_store = b.take<uint16_t>(T * 128, ok); m.idx_expanded = b.take<float>(T * 128, ok); }');after[n]=replace_calls(s,'prefill')
 n=PATHS[7];s=after[n];needle='                config_fp = strata::core::session_config_fingerprint(c);';require(s.count(needle)==1,'Exact persisted config fingerprint point differs');after[n]=s.replace(needle,'                if (strata::kernels::native_qsa_indexer_materialized_half_enabled())\n                    c.switches.emplace_back("native_qsa_indexer_materialized_half", int64_t(1));\n'+needle)
 n=PATHS[8];after[n]=once(after[n],'    float *qfull_ = nullptr,','    uint16_t* idx_half_store_L_ = nullptr; // nQ * MT * ID, matches SYCL public layout\n    float* idx_expanded_L_ = nullptr;\n    float *qfull_ = nullptr,')
 return before,after

def patch_bytes():
 before,after=reconstruct();return ''.join(''.join(difflib.unified_diff(before[n].splitlines(True),after[n].splitlines(True),fromfile='a/'+n,tofile='b/'+n))for n in PATHS).encode('ascii')

def build_proposal():
 before,after=reconstruct();base=json.loads(BASE_PLAN.read_text());require(PATCH.read_bytes()==patch_bytes(),'Exact candidate patch differs');row=dict(base);row['patches']=base['patches']+[{'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)}];expected=dict(base['expected_patched_source_sha256']);expected.update({n:hashlib.sha256(v.encode()).hexdigest()for n,v in after.items()});row.update(expected_patched_source_sha256=expected,status='SOURCE ONLY source41 semantic candidate; fresh build/model/cache proof absent',derived_from_plan=str(BASE_PLAN.relative_to(ROOT)),derived_from_plan_sha256=BASE_SHA,source41_materialized_half={'default_off':True,'all_eight_native_model_callers_wrapped':True,'persisted_ON_mode_key_added':True,'nonnative_indexer_ON_refused':True,'old_source40_proofs_transferred':False,'isolated_helper_public_binding_sha256':'953b5fa16e183f6173bac7801d1553bb0d70e82278dd3314e35c50245c25b686','source40_input_bindings':{n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()},'actual_compilation_qualified':False,'actual_model_qualified':False,'full_model_math_qualified':False,'full_cache_qualified':False,'latency_qualified':False,'fresh_all8_ABI_and_new_upload_oracle_required':True});return row
if __name__=='__main__':
 require(not PATCH.exists()and not PLAN.exists(),'Existing source proposal must be immutable');PATCH.write_bytes(patch_bytes());PLAN.write_text(json.dumps(build_proposal(),indent=2,ensure_ascii=True)+'\n')
