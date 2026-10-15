"""ROOT execution only, after build.exit=0; static CPU audit, no GPU/build."""
from pathlib import Path
import ast, hashlib, json, mmap, re, shlex, struct, sys

P = Path(__file__).resolve().parent
R = P.parent
S = R / 'attn-value-main-s93-prep'
B = R / 'attn-value-main-s93-build/build'
SRC = Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
EXE = B / 'strata-attn-value-s93'
EXPECTED = sys.argv[1]
assert re.fullmatch('[0-9a-f]{64}', EXPECTED)
assert (B / 'build.exit').read_text().strip() == '0'
assert (B / 'tee.exit').read_text().strip() == '0'
initial_stat = EXE.stat()
sha = lambda b: hashlib.sha256(b).hexdigest()
digest = lambda p: sha(p.read_bytes())
records = dict(line.split(maxsplit=1)[::-1] for line in (B / 'build-sha256.txt').read_text().splitlines())
assert records['/out/strata-attn-value-s93'] == EXPECTED
plan = (B / 'ninja-plan.txt').read_text()
commands = [line for line in plan.splitlines() if re.match(r'^\[\d+/\d+\] ', line)]
assert len(commands) == 5
compiles = [line for line in commands if ' -c ' in line]
compile_args = [shlex.split(line.split('] ', 1)[1]) for line in compiles]
assert len(compile_args) == 2 and all('-fp-model=precise' in args for args in compile_args)
compiled_paths = [args[args.index('-c') + 1] for args in compile_args]
assert set(compiled_paths) == {
    '/work/src-ep/sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp',
    '/work/src-ep/sycl/src/prefill/prefill.cpp'}, compiled_paths

# Reuse ONLY the parser class definition, with mmap/bounded-string reads.
# No import/module execution can write to frozen S92 audit roots.
helper = R / 'attn-value-binding-s92-build-audit/host_binding.py'
assert digest(helper) == '37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
tree = ast.parse(helper.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ELF')
def bounded_cstring(buf, start):
    end = buf.find(b'\0', start)
    assert end >= start, 'ELF string table entry lacks NUL terminator'
    return buf[start:end]

class BoundTableStrings(ast.NodeTransformer):
    replacements = 0
    def visit_Subscript(self, item):
        item = self.generic_visit(item)
        call = item.value
        if (isinstance(item.slice, ast.Constant) and item.slice.value == 0
                and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'split' and isinstance(call.func.value, ast.Subscript)):
            suffix = call.func.value
            if (((isinstance(suffix.value, ast.Name) and suffix.value.id == 'strings')
                    or (isinstance(suffix.value, ast.Attribute) and suffix.value.attr == 'data'
                        and isinstance(suffix.value.value, ast.Name) and suffix.value.value.id == 'self'))
                    and isinstance(suffix.slice, ast.Slice) and suffix.slice.upper is None):
                self.replacements += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='bounded_cstring', ctx=ast.Load()),
                    args=[suffix.value, suffix.slice.lower], keywords=[]), item)
        return item

bounded = BoundTableStrings()
node = bounded.visit(node)
assert bounded.replacements == 4, bounded.replacements
ast.fix_missing_locations(node)
code = ast.unparse(node)
code = code.replace('self.data = path.read_bytes()', "self._file = path.open('rb'); self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)")
code = code.replace('self.sha = hashlib.sha256(self.data).hexdigest()', "self.sha = EXPECTED if path == EXE else hashlib.sha256(self.data).hexdigest()")
exec(compile(code, str(helper) + ':class-only-mmap', 'exec'))

pins = json.loads((S / 'source-pins.json').read_text())
for row in pins['before']:
    if 'candidate' in row:
        assert digest(S / row['candidate']) == row['candidate_sha256']
        assert digest(Path(row['path'])) == row['candidate_sha256']
    else:
        assert digest(Path(row['path'])) == row['sha256']
header = SRC / 'include/strata/kernels/qsa_prefill_attn.hpp'
assert digest(header) == '87af3d42d17fe6f74343e90bbdea4b3aee8c7c5fb7095a19b22dc64d69d642c5'
assert header.read_bytes() == (S / 'candidate/qsa_prefill_attn.hpp').read_bytes()
source = (S / 'candidate/qsa_decode_attn.dp.cpp').read_text()
assert 'attn_chunk_kernel<1, false, true>' in source
objpath = SRC / 'build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_decode_attn.dp.cpp.o'
assert digest(objpath) == records['CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_decode_attn.dp.cpp.o']
obj, final = ELF(objpath), ELF(EXE)
oldtags = set(re.findall(r'class (attn_[A-Za-z0-9_]+)', (S / 'original/qsa_decode_attn.dp.cpp').read_text()))
newtags = {'attn_chunk_value_prefill_s93', 'attn_merge_value_prefill_s93'}
assert len(oldtags) == 11 and not oldtags & newtags
assert set(re.findall(r'class (attn_[A-Za-z0-9_]+)', source)) == oldtags | newtags

def target(tag, owner, scalar=None):
    def match(x):
        return tag in x['name'] and x['section'] != 0 and (scalar is None or scalar in x['name'])
    infos = [x for x in obj.symbols if match(x) and 'CompileTimeKernelInfoI16dpct_kernel_name' in x['name']]
    assert len(infos) == 1, (tag, len(infos))
    info = infos[0]
    related = [x for x in obj.symbols if match(x) and 'getDeviceKernelInfo' in x['name']]
    assert len(related) == 2, (tag, related)
    # CompileTimeKernelInfo first field is a relocatable registered-name pointer.
    names = []
    for sec in obj.ss:
        if sec[1] != 4 or sec[7] != info['section']:
            continue
        for off in range(sec[4], sec[4] + sec[5], sec[9]):
            pos, ri, add = struct.unpack_from('<QQq', obj.data, off)
            if pos != info['offset']:
                continue
            symbol = obj.symbols[ri >> 32]
            dst = obj.ss[symbol['section']]
            at = dst[4] + symbol['offset'] + add
            names.append(bounded_cstring(obj.data, at).decode())
    assert len(names) == 1 and tag in names[0] and owner in names[0]
    refs = obj.relocs({info['name']} | {x['name'] for x in related})
    owning_functions = sorted({fn for x in refs for fn in x['containing_functions'] if owner in fn})
    assert owning_functions, (tag, 'No object host callsite references its metadata')
    linked_info = final.symbol(info['name'])
    linked_related = [final.symbol(x['name']) for x in related]
    pointer = final.nameptr(linked_info)
    assert pointer['name'] == names[0]
    return {'tag': tag, 'owner': owner, 'object_host_relocation_count': len(refs),
            'object_owning_functions': owning_functions,
            'compiletime_info_symbol': info['name'], 'info_address': linked_info['offset'],
            'cache_guard_addresses': [x['offset'] for x in linked_related],
            'registered_name_pointer': pointer['pointer'], 'registered_kernel': names[0]}

hosts = {
    'original': target('attn_chunk_kernel_2ad0a4', 'qsa_decode_attn_batch', 'dpct_kernel_scalarILi1EE'),
    'candidate': target('attn_chunk_value_prefill_s93', 'qsa_prefill_attn_value_batch'),
    'oldmerge': target('attn_merge_kernel_338175', 'qsa_decode_attn_batch'),
    'newmerge': target('attn_merge_value_prefill_s93', 'qsa_prefill_attn_value_batch')}
for field in ['info_address', 'registered_name_pointer', 'registered_kernel']:
    assert len({x[field] for x in hosts.values()}) == 4, field
addresses = [a for x in hosts.values() for a in x['cache_guard_addresses']]
assert len(addresses) == len(set(addresses)) == 8
# Every materialized attention tag's metadata is unique in final ELF; retain
# compact names/counts rather than another large repeated metadata dump.
tag_counts = {}
for tag in sorted(oldtags | newtags):
    metadata = [x for x in obj.symbols if x['section'] != 0 and tag in x['name'] and
                any(y in x['name'] for y in ['CompileTimeKernelInfo', 'getDeviceKernelInfo', 'getKernelParamDesc'])]
    assert metadata, tag
    for symbol in metadata:
        final.symbol(symbol['name'])
    tag_counts[tag] = len(metadata)
host_facts = {'status': 'PASS_ACTUAL_MAIN_HOST_BINDINGS', 'binary_sha256_from_build': EXPECTED,
              'attention_object_sha256': obj.sha, 'source_candidate_sha256': digest(S / 'candidate/qsa_decode_attn.dp.cpp'),
              'selected': hosts, 'materialized_tag_metadata_counts': tag_counts,
              'old_tags': sorted(oldtags), 'new_tags': sorted(newtags),
              'all_selected_info_cache_guard_name_pointers_distinct': True,
              'parser_source_sha256': digest(helper), 'scope': 'Object host relocations plus actual final metadata/name pointers; not runtime dispatch.'}
assert digest(P / 'host-binding-v2.json') == 'd2743c058e9ea99bbaa8367185a511eb439311c526a26be5ce1068eb24c287e2'
assert json.loads((P / 'host-binding-v2.json').read_text()) == host_facts
(P / 'host-binding.json').write_text(json.dumps(host_facts, indent=2) + '\n', encoding='ascii')

# Locate four names directly; do NOT iterate/index every offload image.
def elf_at(base):
    data = final.data
    assert data[base:base + 6] == b'\x7fELF\x02\x01'
    so = struct.unpack_from('<Q', data, base + 40)[0]
    se, sn, si = struct.unpack_from('<HHH', data, base + 58)
    assert se == 64 and 0 < sn < 1000 and si < sn
    assert base + so + se * sn <= len(data)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, base + so + k * se) for k in range(sn)]
    z = sections[si]
    strings = data[base + z[4]:base + z[4] + z[5]]
    named = {bounded_cstring(strings, z[0]).decode(): z for z in sections}
    size = max(so + se * sn, max(z[4] + z[5] for z in sections if z[1] != 8))
    assert base + size <= len(data)
    return named, size

prior = json.loads((R / 'attn-value-binding-s92-build-audit/facts.json').read_text())
known = {row['image_sha256']: row for row in prior['resources'].values()}
selection = {}
todo = []
for arm, host in hosts.items():
    # ze_info aligns name values with multiple spaces and repeats names in
    # kernels_misc_info. Deduplicate by validated enclosing full-image ELF.
    pattern = re.compile(rb'(?m)^  - name:[ \t]+' + re.escape(host['registered_kernel'].encode()) + rb'[ \t]*\r?$')
    unique_images = {}
    hits = [match.start() for match in pattern.finditer(final.data)]
    assert hits, (arm, 'No selected ze_info name')
    for at in hits:
        base = final.data.rfind(b'\x7fELF\x02\x01', max(0, at - 4 * 1024 * 1024), at)
        assert base >= 0
        sections, size = elf_at(base)
        z = sections['.ze_info']
        assert base + z[4] <= at < base + z[4] + z[5]
        unique_images[base] = (sections, size)
    assert len(unique_images) == 1, (arm, 'selected distinct enclosing images', list(unique_images))
    base, (sections, size) = next(iter(unique_images.items()))
    z = sections['.ze_info']
    image = final.data[base:base + size]
    info = image[z[4]:z[4] + z[5]].decode()
    names = re.findall(r'^  - name:\s+(\S+)', info.split('kernels_misc_info:')[0], re.M)
    assert names == [host['registered_kernel']], (arm, names)
    textsec = sections['.text.' + host['registered_kernel']]
    text = image[textsec[4]:textsec[4] + textsec[5]]
    options_sec = sections['.misc.buildOptions']
    options = image[options_sec[4]:options_sec[4] + options_sec[5]].decode().strip('\0')
    assert options == '-cl-fp32-correctly-rounded-divide-sqrt'
    assert len(('.text.' + host['registered_kernel'] + '.asm').encode()) <= 255
    image_sha = sha(image)
    reuse = known.get(image_sha)
    if reuse:
        assert digest(Path(reuse['asm_path'])) == reuse['asm_sha256']
    else:
        todo.append(arm)
    (P / (arm + '.zebin')).write_bytes(image)
    (P / (arm + '.ze_info')).write_text(info, encoding='ascii')
    selection[arm] = {'kernel': host['registered_kernel'], 'executable_offset': base,
                      'image_bytes': size, 'image_sha256': image_sha,
                      'text_bytes': len(text), 'text_sha256': sha(text),
                      'ze_info_sha256': sha(info.encode()), 'build_options': options,
                      'unique_host_name_and_selected_image': True,
                      'reuse_exact_full_image_ISA': reuse}
(P / 'selection.json').write_text(json.dumps({'status': 'SELECTED_MAIN_IMAGES_MATCH_HOST_NAMES',
    'binary_sha256_from_build': EXPECTED, 'selected': selection,
    'all_image_metadata_walk': False, 'selected_images': 4,
    'binary_identity_scope': 'Expected root SHA matches build record; mmap stat tracked, no redundant whole-main hash.'}, indent=2) + '\n', encoding='ascii')
(P / 'decode-todo.txt').write_text(''.join(x + '\n' for x in todo), encoding='ascii')
after_stat = EXE.stat()
assert (initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns) == (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns)
(P / 'binary-stat.json').write_text(json.dumps({'inode': initial_stat.st_ino, 'bytes': initial_stat.st_size,
    'mtime_ns': initial_stat.st_mtime_ns, 'unchanged_during_audit': True,
    'root_build_sha256': EXPECTED, 'no_redundant_full_binary_hash': True}, indent=2) + '\n', encoding='ascii')
print(json.dumps({'host': 'PASS', 'selected': {k: {'image_sha256': v['image_sha256'], 'reuse': bool(v['reuse_exact_full_image_ISA'])} for k, v in selection.items()}, 'decode_todo': todo}, indent=2))
