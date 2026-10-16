from pathlib import Path
import mmap, struct

def bounded_cstring(buf, start):
    end = buf.find(b'\x00', start)
    assert end >= start
    return buf[start:end]

class ELF:

    def __init__(self, path):
        self.path = path
        self._file = path.open('rb')
        self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.sha = None
        so = struct.unpack_from('<Q', self.data, 40)[0]
        se, sn, si = struct.unpack_from('<HHH', self.data, 58)
        self.ss = [struct.unpack_from('<IIQQQQIIQQ', self.data, so + k * se) for k in range(sn)]
        ns = self.ss[si]
        strings = self.data[ns[4]:ns[4] + ns[5]]
        self.names = [bounded_cstring(strings, x[0]).decode() for x in self.ss]
        sy = self.ss[self.names.index('.symtab')]
        st = self.ss[sy[6]]
        strings = self.data[st[4]:st[4] + st[5]]
        self.symbols = []
        for off in range(sy[4], sy[4] + sy[5], sy[9]):
            z = struct.unpack_from('<IBBHQQ', self.data, off)
            self.symbols.append({'name': bounded_cstring(strings, z[0]).decode(), 'bind': z[1] >> 4, 'type': z[1] & 15, 'section': z[3], 'offset': z[4], 'size': z[5]})

    def symbol(self, name):
        xs = [x for x in self.symbols if x['name'] == name]
        assert len(xs) == 1, (self.path, name, len(xs))
        return xs[0]

    def nameptr(self, symbol):
        sec = self.ss[symbol['section']]
        pos = sec[4] + symbol['offset'] - sec[3]
        ptr = struct.unpack_from('<Q', self.data, pos)[0]
        for dst in self.ss:
            if dst[1] != 8 and dst[3] <= ptr < dst[3] + dst[5]:
                at = dst[4] + ptr - dst[3]
                return {'pointer': ptr, 'name': bounded_cstring(self.data, at).decode()}
        raise RuntimeError('Linked name pointer has no loaded section')

def refs_to(obj, targets):
    refs = []
    for sec in obj.ss:
        if sec[1] != 4:
            continue
        for at in range(sec[4], sec[4] + sec[5], sec[9]):
            pos, ri, add = struct.unpack_from('<QQq', obj.data, at)
            z = obj.symbols[ri >> 32]
            rt = ri & 4294967295
            if rt not in [1, 2, 10]:
                continue
            destination = z['offset'] + add + (4 if rt == 2 else 0)
            for target in targets:
                if z['section'] == target['section'] and destination == target['offset']:
                    functions = [x['name'] for x in obj.symbols if x['type'] == 2 and x['section'] == sec[7] and (x['offset'] <= pos < x['offset'] + x['size'])]
                    refs.append({'target_metadata': target['name'], 'target_symbol': z['name'], 'STT_SECTION': z['type'] == 3, 'relocation_type': rt, 'addend': add, 'normalized_destination': destination, 'source_section': obj.names[sec[7]], 'containing_functions': functions})
    return refs

def bind(obj, predicate):
    infos = [x for x in obj.symbols if x['section'] and 'CompileTimeKernelInfoI' in x['name'] and predicate(x['name'])]
    assert len(infos) == 1, infos
    info = infos[0]
    related = [x for x in obj.symbols if x['section'] and 'getDeviceKernelInfo' in x['name'] and predicate(x['name'])]
    assert len(related) == 2, related
    names = []
    for sec in obj.ss:
        if sec[1] != 4 or sec[7] != info['section']:
            continue
        for at in range(sec[4], sec[4] + sec[5], sec[9]):
            pos, ri, add = struct.unpack_from('<QQq', obj.data, at)
            if pos != info['offset']:
                continue
            z = obj.symbols[ri >> 32]
            dst = obj.ss[z['section']]
            names.append(bounded_cstring(obj.data, dst[4] + z['offset'] + add).decode())
    assert len(names) == 1
    refs = refs_to(obj, [info] + related)
    assert any((x['target_metadata'] == info['name'] and x['containing_functions'] for x in refs)), 'No actual hostfunction->selected metadata reference'
    linked = final.symbol(info['name'])
    pointer = final.nameptr(linked)
    assert pointer['name'] == names[0]
    return {'object_info': info, 'object_cache_guard': related, 'actual_host_relocations': refs, 'final_info': linked, 'final_cache_guard': [final.symbol(x['name']) for x in related], 'registered_name_pointer': pointer}

def elf_at(base):
    data = final.data
    assert data[base:base + 6] == b'\x7fELF\x02\x01'
    so = struct.unpack_from('<Q', data, base + 40)[0]
    se, sn, si = struct.unpack_from('<HHH', data, base + 58)
    assert se == 64 and 0 < sn < 1000 and (si < sn)
    ss = [struct.unpack_from('<IIQQQQIIQQ', data, base + so + k * se) for k in range(sn)]
    st = ss[si]
    strings = data[base + st[4]:base + st[4] + st[5]]
    sections = {bounded_cstring(strings, z[0]).decode(): z for z in ss}
    size = max(so + se * sn, max((z[4] + z[5] for z in ss if z[1] != 8)))
    assert base + size <= len(data)
    return (sections, size)
