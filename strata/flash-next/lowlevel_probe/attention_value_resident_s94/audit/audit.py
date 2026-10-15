"""S94 root-only CPU audit after build exit0; no GPU/build/ISA decoder."""
from pathlib import Path
import ast, hashlib, json, mmap, re, shlex, struct, sys

P = Path(__file__).resolve().parent
R = P.parent
S = R / 'attn-value-resident-s94-prep'
B = R / 'attn-value-resident-s94-build/build'
SRC = Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
OLD = R / 'attn-value-main-s93-audit-prep'
EXE = B / 'strata-attn-value-resident-s94'
EXPECTED = sys.argv[1]
assert re.fullmatch('[0-9a-f]{64}', EXPECTED)
assert (B / 'build.exit').read_text().strip() == (B / 'tee.exit').read_text().strip() == '0'
initial_stat = EXE.stat()
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
sha = lambda data: hashlib.sha256(data).hexdigest()
records = dict(line.split(maxsplit=1)[::-1] for line in (B / 'build-sha256.txt').read_text().splitlines())
assert records['/out/strata-attn-value-resident-s94'] == EXPECTED

# Existing actual S93 root/object/callsite proof must match before inheritance.
proof_hashes = {
    R / 'attn-value-main-s93-build/build/root-build-closure.json': '20533d56d6e522fa29dff01af080769256ab13cee8b3f4af99f497e436420f4b',
    OLD / 'host-binding.json': 'd2743c058e9ea99bbaa8367185a511eb439311c526a26be5ce1068eb24c287e2',
    OLD / 'selection.json': '334f00fa8dbf17c561d88a47dfb064d3e3770d507ec9f04262b6765de64de144',
    OLD / 'facts.json': '43b123177e346c31fef19f0e287c9e24a93c18113958f75794f9e7b8d1d14bbc',
    OLD / 'audit-v3.py': 'b258e2326cc1a0b6a93e376ee750278014e442106048a62e6ef8d3f4ebc575bc'}
for path, expected in proof_hashes.items():
    assert digest(path) == expected, path
old_host = json.loads((OLD / 'host-binding.json').read_text())
old_selection = json.loads((OLD / 'selection.json').read_text())
old_facts = json.loads((OLD / 'facts.json').read_text())
old_closure = json.loads((R / 'attn-value-main-s93-build/build/root-build-closure.json').read_text())
assert old_closure['status'] == 'PASS_ACTUAL_MAIN_SOURCE_BUILD_BINDING_SELECTED_RESOURCES'
object_key = 'CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_decode_attn.dp.cpp.o'
object_sha = '06eb664c9a991790c01b413117f78475d9ff2836cde77534d69fef2b2c85014d'
archive_sha = '5d465c2d1b23b880e100121b6439721b93fe540a9b3adba8b2ffe9e54a482cbf'
assert records[object_key] == digest(SRC / 'build-sycl-ep' / object_key) == old_host['attention_object_sha256'] == object_sha
assert records['libstrata_kernels.a'] == digest(SRC / 'build-sycl-ep/libstrata_kernels.a') == archive_sha

# S94 has a NEW prefill pin; never check the superseded S93 prefill hash.
pins = json.loads((S / 'source-pins.json').read_text())
assert pins['candidate_prefill_sha256'] == '2a1d8faa8bd58f112f4555a5ba7ab988b143fdb773106c8907903f0ecfafe16e'
assert digest(S / 'prefill.candidate.cpp') == digest(SRC / 'sycl/src/prefill/prefill.cpp') == pins['candidate_prefill_sha256']
for name, row in pins['before'].items():
    if not row['modified_by_candidate']:
        assert digest(SRC / name) == row['sha256'], name
commands = [line for line in (B / 'ninja-plan.txt').read_text().splitlines() if re.match(r'^\[\d+/\d+\] ', line)]
assert len(commands) == 3
compiles = [shlex.split(line.split('] ', 1)[1]) for line in commands if ' -c ' in line]
assert len(compiles) == 1 and '-fp-model=precise' in compiles[0]
assert compiles[0][compiles[0].index('-c') + 1] == '/work/src-ep/sycl/src/prefill/prefill.cpp'

# Reuse definitions only, no imports/top-level execution of old audit modules.
linear_tree = ast.parse((OLD / 'audit-v3.py').read_text())
definitions = [n for n in linear_tree.body if getattr(n, 'name', None) in ['bounded_cstring', 'BoundTableStrings', 'elf_at']]
assert len(definitions) == 3
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(OLD / 'audit-v3.py') + ':definitions-only', 'exec'))
helper = R / 'attn-value-binding-s92-build-audit/host_binding.py'
assert digest(helper) == '37ecf55c43300319f0a3baaf15bede814934bd3ee1b0a11b32a38dc6066d4aa6'
tree = ast.parse(helper.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ELF')
bounded = BoundTableStrings()
node = bounded.visit(node)
assert bounded.replacements == 4
ast.fix_missing_locations(node)
code = ast.unparse(node)
code = code.replace('self.data = path.read_bytes()', "self._file = path.open('rb'); self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)")
code = code.replace('self.sha = hashlib.sha256(self.data).hexdigest()', 'self.sha = EXPECTED')
exec(compile(code, str(helper) + ':linear-class-only', 'exec'))
final = ELF(EXE)
hosts = {}
for arm, old in old_host['selected'].items():
    info = final.symbol(old['compiletime_info_symbol'])
    related = [x for x in final.symbols if x['section'] != 0 and 'getDeviceKernelInfo' in x['name'] and old['tag'] in x['name']
               and (arm != 'original' or 'dpct_kernel_scalarILi1EE' in x['name'])]
    assert len(related) == 2, (arm, related)
    pointer = final.nameptr(info)
    assert pointer['name'] == old['registered_kernel']
    hosts[arm] = {'tag': old['tag'], 'owner': old['owner'], 'info_address': info['offset'],
                  'cache_guard_addresses': [x['offset'] for x in related],
                  'registered_name_pointer': pointer['pointer'], 'registered_kernel': pointer['name'],
                  'inherited_S93_object_host_relocations': old['object_host_relocation_count']}
for field in ['info_address', 'registered_name_pointer', 'registered_kernel']:
    assert len({x[field] for x in hosts.values()}) == 4
addresses = [a for x in hosts.values() for a in x['cache_guard_addresses']]
assert len(addresses) == len(set(addresses)) == 8
assert len(old_host['old_tags']) == 11 and len(old_host['new_tags']) == 2
final_tag_counts = {}
for tag, count in old_host['materialized_tag_metadata_counts'].items():
    matches = [x for x in final.symbols if x['section'] != 0 and tag in x['name'] and
               any(term in x['name'] for term in ['CompileTimeKernelInfo', 'getDeviceKernelInfo', 'getKernelParamDesc'])]
    assert len(matches) == count, (tag, count, len(matches))
    assert len({x['name'] for x in matches}) == count
    final_tag_counts[tag] = len(matches)

selected = {}
mismatches = []
for arm, host in hosts.items():
    pattern = re.compile(rb'(?m)^  - name:[ \t]+' + re.escape(host['registered_kernel'].encode()) + rb'[ \t]*\r?$')
    hits = [match.start() for match in pattern.finditer(final.data)]
    assert hits, arm
    unique = {}
    for at in hits:
        base = final.data.rfind(b'\x7fELF\x02\x01', max(0, at - 4 * 1024 * 1024), at)
        assert base >= 0
        sections, size = elf_at(base)
        z = sections['.ze_info']
        assert base + z[4] <= at < base + z[4] + z[5]
        unique[base] = (sections, size)
    assert len(unique) == 1, (arm, list(unique))
    base, (sections, size) = next(iter(unique.items()))
    image = final.data[base:base + size]
    z = sections['.ze_info']
    info = image[z[4]:z[4] + z[5]].decode()
    names = re.findall(r'^  - name:\s+(\S+)', info.split('kernels_misc_info:')[0], re.M)
    assert names == [host['registered_kernel']]
    t = sections['.text.' + host['registered_kernel']]
    text = image[t[4]:t[4] + t[5]]
    original = old_selection['selected'][arm]
    equal = sha(image) == original['image_sha256']
    selected[arm] = {'kernel': names[0], 'executable_offset': base, 'image_bytes': size,
                     'image_sha256': sha(image), 'text_bytes': len(text), 'text_sha256': sha(text),
                     'ze_info_sha256': sha(info.encode()), 'unique_validated_image': True,
                     'exact_S93_full_image': equal}
    (P / (arm + '.zebin')).write_bytes(image)
    (P / (arm + '.ze_info')).write_text(info, encoding='ascii')
    if not equal:
        mismatches.append(arm)
host_output = {'selected': hosts, 'S93_proof_hashes': {str(p): h for p, h in proof_hashes.items()},
               'attention_object_sha256': object_sha, 'kernel_archive_sha256': archive_sha,
               'old11_new2_object_metadata_and_relocation_proof_inherited': True,
               'all13_final_materialized_tag_metadata_counts': final_tag_counts,
               'final_four_Info_cache_guard_name_pointers_distinct': True}
(P / 'host-binding.json').write_text(json.dumps(host_output, indent=2) + '\n', encoding='ascii')
(P / 'selection.json').write_text(json.dumps({'selected': selected, 'image_mismatches': mismatches,
    'all_image_walk': False, 'new_ISA_decodes': 0}, indent=2) + '\n', encoding='ascii')
if mismatches:
    (P / 'HOLD.json').write_text(json.dumps({'status': 'HOLD_SELECTED_FULL_IMAGE_CHANGED_ROOT_INSPECTION_REQUIRED',
        'image_mismatches': mismatches, 'no_automatic_ISA_decode_build_or_GPU': True}, indent=2) + '\n', encoding='ascii')
    raise SystemExit(2)
resources = old_facts['resources']
for arm, resource in resources.items():
    assert resource['image_sha256'] == selected[arm]['image_sha256']
    assert digest(Path(resource['asm_path'])) == resource['asm_sha256']
image_id = json.loads((B / 'image.json').read_text())[0]['Id']
assert image_id == old_facts['image']
assert (B / 'compiler-version.txt').read_text().strip() == old_facts['compiler']
end_stat = EXE.stat()
assert (initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns) == (end_stat.st_ino, end_stat.st_size, end_stat.st_mtime_ns)
facts = {'status': 'STATIC_S94_MAIN_BINDING_EXACT_ISA_REUSE_PASS_ROOT_CLOSURE_PENDING',
    'binary_sha256_from_root_build_record': EXPECTED,
    'prefill_source_sha256': pins['candidate_prefill_sha256'], 'attention_object_sha256': object_sha,
    'kernel_archive_sha256': archive_sha, 'host': host_output, 'selected': selected,
    'resources': resources, 'ISA_reuse': 'All four full images exact S93 and prior ASM byte-hashes rechecked; no ocloc.',
    'image': image_id, 'compiler': old_facts['compiler'],
    'main_identity': {'inode': initial_stat.st_ino, 'bytes': initial_stat.st_size, 'mtime_ns': initial_stat.st_mtime_ns,
                      'stat_unchanged': True, 'no_redundant_full_main_hash': True},
    'limits': ['Static selected host/image ownership, not caller activation or runtime dispatch/count/speed proof.',
               'Old11+new2 object metadata/callsites inherited ONLY after exact whole object/archive and S93 proof hashes.',
               'Resource field absence is not zero. No occupancy/traffic/cause claim; no fullimage walk.',
               'Real main -fsycl build, not thin-host-link zero-AOT claim. Root owns final runtime closure.'],
    'root_closure_written_by_this_audit': False, 'new_builds': 0, 'GPU': 0, 'new_ISA_decodes': 0}
(P / 'facts.json').write_text(json.dumps(facts, indent=2) + '\n', encoding='ascii')
files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(P.iterdir())
         if p.is_file() and p.name not in ['final-facts-manifest.json', 'audit.stdout', 'audit.stderr']}
(P / 'final-facts-manifest.json').write_text(json.dumps({'status': 'S94_STATIC_AUDIT_FINISHED_ROOT_READ_PENDING', 'files': files}, indent=2) + '\n', encoding='ascii')
print(json.dumps({'status': facts['status'], 'binary_sha256': EXPECTED, 'all_four_full_images_exact_S93': True,
    'manifest_sha256': digest(P / 'final-facts-manifest.json')}, indent=2))
