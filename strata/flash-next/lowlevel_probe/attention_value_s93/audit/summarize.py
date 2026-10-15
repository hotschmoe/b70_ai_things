from pathlib import Path
import hashlib, json, re
P = Path(__file__).resolve().parent
R = P.parent
B = R / 'attn-value-main-s93-build/build'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((P / 'selection.json').read_text())
host = json.loads((P / 'host-binding.json').read_text())
resources = {}
for arm, row in selection['selected'].items():
    info = (P / (arm + '.ze_info')).read_text()
    env = info.split('    execution_env:', 1)[1].split('    payload_arguments:', 1)[0]
    vals = {}
    for key in ['barrier_count', 'grf_count', 'simd_size', 'slm_size', 'private_size', 'spill_size', 'scratch_size', 'eu_thread_count']:
        m = re.search(r'^      ' + key + r':\s+(\S+)', env, re.M)
        vals[key] = int(m[1]) if m and m[1].isdigit() else m[1] if m else 'absent'
    scratch = re.search(r'per_thread_memory_buffers:\s*\n\s*- type:\s*scratch\s*\n\s*usage:\s*(\S+)\s*\n\s*size:\s*(\d+)', info)
    vals['per_thread_scratch_buffer'] = {'usage': scratch[1], 'size': int(scratch[2])} if scratch else 'absent'
    reuse = row['reuse_exact_full_image_ISA']
    if reuse:
        asm = Path(reuse['asm_path'])
        assert sha(asm) == reuse['asm_sha256']
    else:
        assert (P / (arm + '-disasm.exit')).read_text().strip() == '0'
        matches = list((P / (arm + '-disasm')).glob('*.asm'))
        assert len(matches) == 1
        asm = matches[0]
    lines = asm.read_text().splitlines()
    ss = [i + 1 for i, line in enumerate(lines) if 'send.ugm' in line and '.ss[a0.0]' in line]
    vals.update(image_sha256=row['image_sha256'], text_bytes=row['text_bytes'],
                asm_path=str(asm), asm_sha256=sha(asm), ss_a32_pattern_sites=len(ss),
                ss_a32_pattern_lines=ss, decode='exact full-image reuse' if reuse else 'selected image newly decoded')
    resources[arm] = vals
todo = (P / 'decode-todo.txt').read_text().strip()
if todo:
    assert (P / 'decoder-input-before.sha256').read_bytes() == (P / 'decoder-input-after.sha256').read_bytes()
assert (B / 'build.exit').read_text().strip() == '0'
image = json.loads((B / 'image.json').read_text())[0]['Id']
assert image == 'sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
facts = {'status': 'STATIC_MAIN_S93_BINDING_RESOURCES_ROOT_CLOSURE_PENDING',
         'binary_sha256_from_root_build_record': selection['binary_sha256_from_build'],
         'source_candidate_sha256': host['source_candidate_sha256'],
         'attention_object_sha256': host['attention_object_sha256'],
         'host': host, 'selected': selection['selected'], 'resources': resources,
         'image': image, 'compiler': (B / 'compiler-version.txt').read_text().strip(),
         'main_build_scope': 'Real main -fsycl link, not the separate CAP ordinary-host-link recipe; no zero-AOT claim.',
         'limits': ['Static actual host/image ownership, no runtime dispatch/count/speed/correctness proof.',
                    'Absent scratch/spill fields are not zero; SS pattern count is not all traffic.',
                    'Only selected original M1, new <1,false,true> and old/new batch merge images decoded or exact-hash reused.',
                    'No whole-main image walk, general ODR, occupancy, physical traffic or cause proof.'],
         'root_closure_written_by_this_audit': False, 'GPU': 0, 'new_builds': 0}
(P / 'facts.json').write_text(json.dumps(facts, indent=2) + '\n', encoding='ascii')
files = {str(p.relative_to(P)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
         for p in sorted(P.iterdir()) if p.is_file() and p.name not in ['final-facts-manifest.json', 'summarize.stdout', 'summarize.stderr']}
(P / 'final-facts-manifest.json').write_text(json.dumps({'status': 'STATIC_MAIN_S93_AUDIT_FINISHED_ROOT_READ_PENDING', 'files': files}, indent=2) + '\n', encoding='ascii')
print(json.dumps({'status': facts['status'], 'resources': resources, 'manifest_sha256': sha(P / 'final-facts-manifest.json')}, indent=2))
