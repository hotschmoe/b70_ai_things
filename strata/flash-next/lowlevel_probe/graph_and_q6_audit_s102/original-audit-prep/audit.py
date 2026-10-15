from pathlib import Path
import ast, csv, hashlib, json, mmap, re, struct, subprocess

P = Path(__file__).resolve().parent
R = P.parent
O = R / 'q6-a4-s102-audit'
SRC = Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
EXE = R / 'routed-iq2-multi-s100-build/build/strata-routed-iq2-multi-s100'
EXPECTED = '7ee675fff8c342772e4f6eae402d40c655a29cf7dcc119d504cdd8d1386b7bbc'
OBJ = SRC / 'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o'
OBJECT_SHA = 'b9f931cb65927a2869a1b26bce8cfa834b112bccdfe08d26e9364634473294dd'
SOURCE = SRC / 'sycl/src/kernels/cuda/native_mmvq.dp.cpp'
SOURCE_SHA = 'c1cdeabe747754e5f0b3d2f4710c06d4dc50911fc692100be5191559a9ac8584'
OLD = R / 'q6-center-s85-isa'
OLD_NAME = '_ZTSN3s859PrivateC6ILb0EEE'
sha = lambda b: hashlib.sha256(b).hexdigest()
assert not O.exists(), 'Existing output: inspect prior handle/result; never blindly rerun'
O.mkdir()
assert sha(SOURCE.read_bytes()) == SOURCE_SHA
proof = json.loads((R / 'routed-iq2-multi-s100-build/build/root-build-audit.json').read_text())
assert proof['hashes']['/out/strata-routed-iq2-multi-s100'] == EXPECTED
assert proof['unchanged_inputs_after_checked'] is True
assert (OBJECT_SHA + '  CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o') in (R / 'routed-iq2-multi-s100-build/expected-unchanged-build-inputs.sha256').read_text()
stats = {str(f): [f.stat().st_ino, f.stat().st_size, f.stat().st_mtime_ns] for f in (EXE, OBJ, SOURCE)}
with EXE.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED

# Reuse ONLY the audited ELF class, bounded as in S97; no helper top-level execution.
helper = R / 'attn-value-binding-s92-build-audit/host_binding.py'
assert sha(helper.read_bytes()) == '37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
node = next(n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ELF')
def cstr(buf, start):
    end = buf.find(b'\0', start)
    assert end >= start
    return buf[start:end]
class Bound(ast.NodeTransformer):
    replacements = 0
    def visit_Subscript(self, n):
        n = self.generic_visit(n)
        c = n.value
        if isinstance(n.slice, ast.Constant) and n.slice.value == 0 and isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'split' and isinstance(c.func.value, ast.Subscript):
            s = c.func.value
            if ((isinstance(s.value, ast.Name) and s.value.id == 'strings') or (isinstance(s.value, ast.Attribute) and s.value.attr == 'data' and isinstance(s.value.value, ast.Name) and s.value.value.id == 'self')) and isinstance(s.slice, ast.Slice) and s.slice.upper is None:
                self.replacements += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='cstr', ctx=ast.Load()), args=[s.value, s.slice.lower], keywords=[]), n)
        return n
transform = Bound()
node = transform.visit(node)
assert transform.replacements == 4
ast.fix_missing_locations(node)
code = ast.unparse(node).replace('self.data = path.read_bytes()', "self._file=path.open('rb'); self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)").replace('self.sha = hashlib.sha256(self.data).hexdigest()', 'self.sha=EXPECTED if path==EXE else hashlib.sha256(self.data).hexdigest()')
exec(compile(code, str(helper) + ':class-only-bounded-mmap', 'exec'))
obj, final = ELF(OBJ), ELF(EXE)
assert obj.sha == OBJECT_SHA
def wanted(n):
    return 'native_mmvq_q6k_wide_a2' in n and 'launch_q6k_wideILi6EE' in n and 'dpct_kernel_scalarILi6EE' in n
infos = [s for s in obj.symbols if s['section'] and 'CompileTimeKernelInfoI16dpct_kernel_name' in s['name'] and wanted(s['name'])]
assert len(infos) == 1, len(infos)
info = infos[0]
related = [s for s in obj.symbols if s['section'] and 'getDeviceKernelInfo' in s['name'] and wanted(s['name'])]
assert len(related) == 2, related
names, refs = [], []
for sec in obj.ss:
    if sec[1] != 4:
        continue
    for at in range(sec[4], sec[4] + sec[5], sec[9]):
        pos, ri, add = struct.unpack_from('<QQq', obj.data, at)
        z, rt = obj.symbols[ri >> 32], ri & 0xffffffff
        if sec[7] == info['section'] and pos == info['offset']:
            dest = obj.ss[z['section']][4] + z['offset'] + add
            names.append(cstr(obj.data, dest).decode())
        if rt not in (1, 2, 10):
            continue
        destination = z['offset'] + add + (4 if rt == 2 else 0)
        for target in [info] + related:
            if z['section'] == target['section'] and destination == target['offset']:
                fns = [s['name'] for s in obj.symbols if s['type'] == 2 and s['section'] == sec[7] and s['offset'] <= pos < s['offset'] + s['size']]
                refs.append({'target': target['name'], 'symbol': z['name'], 'section_symbol': z['type'] == 3, 'relocation_type': rt, 'addend': add, 'destination': destination, 'source_section': obj.names[sec[7]], 'functions': fns})
assert len(names) == 1 and wanted(names[0])
name = names[0]
callers = sorted({fn for row in refs if row['target'] == info['name'] for fn in row['functions'] if 'launch_q6k_wideILi6EE' in fn})
assert len(callers) == 1, callers
linked_info = final.symbol(info['name'])
pointer = final.nameptr(linked_info)
assert pointer['name'] == name
linked_related = [final.symbol(s['name']) for s in related]
assert len({s['offset'] for s in linked_related}) == 2
hostasm = subprocess.check_output(['objdump', '-dr', '--disassemble=' + callers[0], str(OBJ)], text=True)
assert 'getDeviceKernelInfo' in hostasm and 'CompileTimeKernelInfoTyE' in hostasm
(O / 'host-launch-q6-c6.asm').write_text(hostasm, encoding='ascii')
observed = {r['operation'].split('[SIMD32')[0] for r in csv.DictReader((Path('/mnt/vm_8tb/github/b70_ai_things') / 'strata/flash-next/lowlevel_probe/current_c2_profile_s95/runtime/analysis/gpu/operations.csv').open()) if r['kind'] == 'gpu' and 'native_mmvq_q6k_wide_a2' in r['operation'] and 'scalar<6>' in r['operation']}
demangled = subprocess.check_output(['c++filt', name], text=True).strip().removeprefix('typeinfo name for ')
assert observed == {demangled}, (observed, demangled)

def sections(buf, base=0):
    assert buf[base:base + 6] == b'\x7fELF\x02\x01'
    so = struct.unpack_from('<Q', buf, base + 40)[0]
    se, sn, si = struct.unpack_from('<HHH', buf, base + 58)
    assert se == 64 and 0 < sn < 1000 and si < sn
    ss = [struct.unpack_from('<IIQQQQIIQQ', buf, base + so + k * se) for k in range(sn)]
    st = ss[si]
    strings = buf[base + st[4]:base + st[4] + st[5]]
    named = {cstr(strings, s[0]).decode(): s for s in ss}
    size = max(so + se * sn, max(s[4] + s[5] for s in ss if s[1] != 8))
    assert base + size <= len(buf)
    return named, size
unique = {}
for match in re.compile(rb'(?m)^  - name:[ \t]+' + re.escape(name.encode()) + rb'[ \t]*\r?$').finditer(final.data):
    at = match.start()
    base = final.data.rfind(b'\x7fELF\x02\x01', max(0, at - 4 * 1024 * 1024), at)
    ss, size = sections(final.data, base)
    z = ss['.ze_info']
    assert base + z[4] <= at < base + z[4] + z[5]
    unique[base] = (ss, size)
assert len(unique) == 1, len(unique)
base, (ss, size) = next(iter(unique.items()))
image = bytes(final.data[base:base + size])
(O / 'q6-c6.zebin').write_bytes(image)
# A copy changes ONLY the section-name string for optional later selected decoding.
alias = bytearray(image)
so = struct.unpack_from('<Q', image, 40)[0]
se, sn, si = struct.unpack_from('<HHH', image, 58)
st = struct.unpack_from('<IIQQQQIIQQ', image, so + si * se)
name_at = st[4] + ss['.text.' + name][0]
old_section_name = cstr(alias, name_at)
short = b'.text.s102_q6c6'
assert old_section_name == ('.text.' + name).encode() and len(short) < len(old_section_name)
alias[name_at:name_at + len(old_section_name)] = short + b'\0' * (len(old_section_name) - len(short))
changed = [i for i, (a, b) in enumerate(zip(image, alias)) if a != b]
assert all(st[4] <= i < st[4] + st[5] for i in changed)
(O / 'q6-c6-shortname.zebin').write_bytes(alias)
old_image = (OLD / 'original.zebin').read_bytes()
assert sha(old_image) == '6bc2acb286c7c00eea4678f404681b329d42e6dac05d268bd995b5e6feb2d6b8'
old_ss, _ = sections(old_image)
get = lambda buf, tab, k: buf[tab[k][4]:tab[k][4] + tab[k][5]] if tab[k][1] != 8 else b''
ze = get(image, ss, '.ze_info').decode()
old_ze = get(old_image, old_ss, '.ze_info').decode()
assert re.findall(r'^  - name:\s+(\S+)', ze.split('kernels_misc_info:')[0], re.M) == [name, 'Intel_Symbol_Table_Void_Program']
(O / 'q6-c6.ze_info').write_text(ze, encoding='ascii')
text = get(image, ss, '.text.' + name)
old_text = get(old_image, old_ss, '.text.' + OLD_NAME)
asm_path = OLD / ('original-disasm/.text.' + OLD_NAME + '.asm')
asm = asm_path.read_text()
assert sha(asm.encode()) == '668ee30e1417c8c70c2aedd99660f8a94bb215653531281e4c5a492d02711a09'
assert sha(old_text) == '4a3dd695ef0dee53fed7ebaf9c9491d399564ae5ae3d2693c81f9f09e62d808d'
assert asm.count('{EOT,') == 1
eot_end = int(re.search(r'\{EOT,[^\n]*\nL(\d+):', asm)[1])
assert eot_end == 16384
assert all(int(x) < eot_end for line in asm.splitlines() if re.search(r'\b(goto|jmpi|join)\b', line) for x in re.findall(r'\bL(\d+)\b', line))
prefix_equal = text[:eot_end] == old_text[:eot_end] and len(text) >= eot_end
env_equal = ze.split('kernels_misc_info:')[0].replace(name, '<TARGET>') == old_ze.split('kernels_misc_info:')[0].replace(OLD_NAME, '<TARGET>')
options = get(image, ss, '.misc.buildOptions')
options_equal = options == get(old_image, old_ss, '.misc.buildOptions')
assert options.rstrip(b'\0') == b'-cl-fp32-correctly-rounded-divide-sqrt'
def inventory(buf, tab):
    return {k: {'type': s[1], 'flags': s[2], 'bytes': s[5], 'sha256': sha(get(buf, tab, k)) if s[1] != 8 else None} for k, s in tab.items()}
new_inv, old_inv = inventory(image, ss), inventory(old_image, old_ss)
data_keys = {k for tab in (ss, old_ss) for k in tab if k.startswith(('.data', '.bss', '.rodata'))}
data_equal = all(k in ss and k in old_ss and new_inv[k] == old_inv[k] for k in data_keys)
reloc_keys = {k.replace(name, '<TARGET>').replace(OLD_NAME, '<TARGET>') for tab in (ss, old_ss) for k in tab if k.startswith(('.rel.', '.rela.'))}
relocs_equal = all(get(image, ss, k.replace('<TARGET>', name)) == get(old_image, old_ss, k.replace('<TARGET>', OLD_NAME)) for k in reloc_keys if k.replace('<TARGET>', name) in ss and k.replace('<TARGET>', OLD_NAME) in old_ss)
relocs_equal &= {k.replace(name, '<TARGET>') for k in ss if k.startswith(('.rel.', '.rela.'))} == {k.replace(OLD_NAME, '<TARGET>') for k in old_ss if k.startswith(('.rel.', '.rela.'))}
reuse = prefix_equal and env_equal and options_equal and data_equal and relocs_equal
differences = [i for i, (a, b) in enumerate(zip(text, old_text)) if a != b]
v = {'status': 'PASS_EXISTING_S85_COMPUTATIONAL_ASM_REUSE_PENDING_ROOT_INTERPRETATION' if reuse else 'HOLD_MAIN_VS_PRIVATE_DIFFERENCE_ROOT_DECISION_BEFORE_DECODER', 'binary_sha256': EXPECTED, 'source_sha256': SOURCE_SHA, 'object_sha256': OBJECT_SHA, 'host': {'object_Info': info, 'object_cache_guard': related, 'object_relocations': refs, 'caller': callers[0], 'final_Info': linked_info, 'final_cache_guard': linked_related, 'name_pointer': pointer, 'observed_S95_demangled': demangled}, 'selected': {'kernel': name, 'executable_offset': base, 'image_bytes': size, 'image_sha256': sha(image), 'text_bytes': len(text), 'text_sha256': sha(text), 'ze_info_sha256': sha(ze.encode()), 'inventory': new_inv}, 'alias_copy': {'sha256': sha(alias), 'only_section_name_string_changed': True, 'changed_byte_count': len(changed), 'text_unchanged': True}, 'S85_original': {'kernel': OLD_NAME, 'image_sha256': sha(old_image), 'text_bytes': len(old_text), 'text_sha256': sha(old_text), 'ASM': str(asm_path), 'ASM_sha256': sha(asm.encode()), 'inventory': old_inv}, 'comparison': {'whole_image_equal': image == old_image, 'whole_text_equal': text == old_text, 'computational_prefix_bytes': eot_end, 'computational_prefix_equal': prefix_equal, 'text_differing_offsets': differences, 'text_length_equal': len(text) == len(old_text), 'normalized_execution_payload_metadata_equal': env_equal, 'options_equal': options_equal, 'constant_data_and_bss_equal': data_equal, 'raw_relocation_sections_equal': relocs_equal}, 'limitations': ['No new decoder, build, GPU, timing or numerical claim.', 'Private binding never substitutes for main: actual main host/name/image resolved above.', 'Metadata misc argument type names and embedded SPV remain inventoried raw differences.', 'S85 code identity, if accepted by root, permits reusing its existing target instructions; extra helper differences remain raw inventory and do not imply helper equivalence.', 'Dynamic costs and A4 candidate lowering remain unknown.'], 'GPU': 0, 'build': 0, 'decoder': 0, 'all_image_walk': False}
for f in (EXE, OBJ, SOURCE):
    assert stats[str(f)] == [f.stat().st_ino, f.stat().st_size, f.stat().st_mtime_ns]
(O / 'facts.json').write_text(json.dumps(v, indent=2, ensure_ascii=True) + '\n', encoding='ascii')
print(json.dumps({'status': v['status'], 'selected_image_bytes': size, 'comparison': v['comparison']}, indent=2))
