#!/usr/bin/env python3
"""SOURCE ONLY finite alignment/byte-flow check; not compiled GPU proof."""
import json, struct

cases = 0
for base_mod16 in (0, 4, 8, 12):
    base = 64 + base_mod16
    # Eight complete consecutive blocks cover every even start residue mod16.
    buf = bytes(((i * 109) ^ ((i >> 3) * 37)) & 255 for i in range(base + 8 * 210 + 64))
    for block in range(8):
        origin = base + block * 210
        for g in range(8):
            for offset in (16 * g, 128 + 16 * (2 * (g >> 2) + (g & 1))):
                a = origin + offset
                start = a & ~3
                end = start + 20
                assert start % 4 == 0 and a % 2 == 0
                assert start >= base and end <= origin + 210
                # A preceding-block d field is allowed; never before the allocation.
                assert start >= origin - 2
                d = struct.unpack_from('<5I', buf, start)
                actual = [((d[i] >> 16) | ((d[i + 1] << 16) & 0xffffffff)) if a & 2 else d[i] for i in range(4)]
                expected = list(struct.unpack_from('<4I', buf, a))
                assert actual == expected
                # Symbolic byte provenance: check source byte index of every output byte.
                window = list(range(start, end))
                chosen = window[2:18] if a & 2 else window[:16]
                assert chosen == list(range(a, a + 16))
                cases += 1
print(json.dumps({'status': 'PASS_FINITE_SOURCE_BYTE_FLOW_AND_BOUNDS_ONLY', 'cases': cases, 'base_mod16': [0, 4, 8, 12], 'complete_blocks': 8, 'all_even_address_residues': True, 'scalar_reads_are_dword_aligned': True, 'selected_16bytes_exact': True, 'compiled_instruction_or_FP_equivalence': False, 'GPU': 0}))
