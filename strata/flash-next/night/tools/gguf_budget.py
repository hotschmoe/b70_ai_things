#!/usr/bin/env python3
"""Per-tensor byte budget of a GGUF (header only; sizes from data offsets)."""
import struct, sys, re, collections
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def rstr(): n, = rd('Q'); return f.read(n).decode('utf-8', 'replace')
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at, n = rd('IQ'); return [rval(at) for _ in range(n)]
    fm = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
    return rd(fm)[0]
magic = f.read(4); ver, nt, nkv = rd('IQQ')
kv = {}
for _ in range(nkv):
    k = rstr(); t, = rd('I'); v = rval(t)
    kv[k] = v if not isinstance(v, list) or len(v) < 64 else f'<list {len(v)}>'
tensors = []
for _ in range(nt):
    name = rstr(); nd, = rd('I'); dims = rd('Q' * nd); typ, off = rd('IQ'); tensors.append([name, dims, typ, off])
align = kv.get('general.alignment', 32)
data0 = (f.tell() + align - 1) // align * align
import os
fsize = os.path.getsize(sys.argv[1])
tensors.sort(key=lambda t: t[3])
for i, t in enumerate(tensors):
    end = tensors[i + 1][3] if i + 1 < len(tensors) else fsize - data0
    t.append(end - t[3])
for k, v in kv.items():
    if any(s in k for s in ('block_count', 'embedding_length', 'expert', 'head', 'context', 'vocab', 'feed_forward', 'ssm', 'rope', 'layer')):
        print(k, '=', v)
fam = collections.defaultdict(lambda: [0, 0, set()])
for name, dims, typ, off, size in tensors:
    key = re.sub(r'blk\.\d+\.', 'blk.N.', name)
    fam[key][0] += size; fam[key][1] += 1; fam[key][2].add((typ, dims))
tot = sum(t[4] for t in tensors)
print(f'\nTOTAL {tot/2**30:.2f} GiB in {len(tensors)} tensors')
print(f"{'family':45s} {'GiB':>7s} {'n':>4s}  MiB/each  types/dims")
for k, (s, n, td) in sorted(fam.items(), key=lambda x: -x[1][0]):
    tds = sorted(td)[:3]
    print(f'{k:45s} {s/2**30:7.3f} {n:4d} {s/n/2**20:9.2f}  {tds}')
