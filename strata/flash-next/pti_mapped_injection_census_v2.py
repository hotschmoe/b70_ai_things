#!/usr/bin/env python3
"""Explicit Btrfs mapped-file association; no process memory or GPU queries."""
import argparse,hashlib,json,os,stat,re
from pathlib import Path,PurePosixPath

LIB='/pti/lib/libunitrace_tool.so'
def require(ok, message):
    if not ok:
        raise ValueError(message)

def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

def process(text):
    tail = text[text.rfind(')') + 2:].split()
    return {'pid': int(text.split(' ', 1)[0]), 'start_ticks': int(tail[19])}

def select(maps, lib=LIB):
    rows = []
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if Path(path.replace(' (deleted)', '')).name == 'libunitrace_tool.so':
            require(path == lib, 'Foreign/deleted mapped injection path refused')
            region = fields[0].split('-')
            device = fields[3].split(':')
            require(len(region) == len(device) == 2, 'Actual mapping layout required')
            rows.append({'start': int(region[0], 16), 'end': int(region[1], 16), 'permissions': fields[1], 'offset': int(fields[2], 16), 'device_major': int(device[0], 16), 'device_minor': int(device[1], 16), 'inode': int(fields[4]), 'original_container_path': path})
    require(rows and any(('x' in r['permissions'] for r in rows)), 'Actual executable injected mapping missing')
    return rows
def unescape(word):
    return re.sub('\\\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), word)

def covering_mount(text, path):
    rows = []
    p = PurePosixPath(path)
    for line in text.splitlines():
        left, sep, right = line.partition(' - ')
        require(sep, 'Complete mountinfo line required')
        before = left.split()
        after = right.split()
        require(len(before) >= 6 and len(after) >= 3, 'Complete mountinfo fields required')
        mp = PurePosixPath(unescape(before[4]))
        if p == mp or mp in p.parents:
            dev = before[2].split(':')
            require(len(dev) == 2 and all((v.isdigit() for v in dev)), 'Typed mount major/minor required')
            rows.append({'mount_id': int(before[0]), 'parent_id': int(before[1]), 'device_major': int(dev[0]), 'device_minor': int(dev[1]), 'root': unescape(before[3]), 'mountpoint': str(mp), 'mount_options': before[5].split(','), 'filesystem': after[0], 'source': unescape(after[1]), 'super_options': after[2].split(','), 'original_line': line})
    require(rows, 'Actual covering mount required')
    length = max((len(r['mountpoint']) for r in rows))
    chosen = [r for r in rows if len(r['mountpoint']) == length]
    require(len(chosen) == 1, 'Unique actual longest covering mount required')
    return chosen[0]

def associate(rows, path_stat_before, path_stat_after, fd_stat_before, fd_stat_after, mount_before, mount_after, expected_path, expected_bind_root, expected_filesystem_source, actual_elf_sha256, expected_elf_sha256, map_links, expected_mountpoint='/pti'):
    """No process queries/file reads here. Actual producer must supply every row."""
    require(path_stat_before == path_stat_after == fd_stat_before == fd_stat_after, 'Current openedFD/path exact stat5 before/after required')
    s = path_stat_before
    require(type(s) is list and len(s) == 5 and all((type(v) is int and v >= 0 for v in s)) and (s[1] > 0) and (0 < s[2] <= 64 << 20), 'Typed current regular ELF stat5 required')
    require(actual_elf_sha256 == expected_elf_sha256 and re.fullmatch('[0-9a-f]{64}', expected_elf_sha256), 'Actual expected backing ELF SHA required')
    require(mount_before == mount_after, 'Actual covering mount must stay unchanged')
    m = mount_before
    require(m['filesystem'] == 'btrfs' and m['source'] == expected_filesystem_source and (m['root'] == expected_bind_root) and (m['mountpoint'] == expected_mountpoint) and ('ro' in m['mount_options']) and ('subvolid=5' in m['super_options']), 'Exact admitted readonly Btrfs install bind/subvolume required')
    require(rows and len(rows) == len(map_links), 'Complete actual mapped segment/link roster required')
    last = 0
    for row, link in zip(rows, map_links):
        require(all((type(row[n]) is int and row[n] >= 0 for n in ('start', 'end', 'offset', 'inode', 'device_major', 'device_minor'))) and row['start'] >= last and (row['end'] > row['start']), 'Typed ordered mapped ranges required')
        last = row['end']
        require(row['original_container_path'] == expected_path and link == expected_path, 'Exact nondeleted native map/map_files path required')
        require(row['inode'] == s[1] and row['device_major'] == m['device_major'] and (row['device_minor'] == m['device_minor']), 'Mapped inode and Btrfs superblock/mount device must match')
        require(row['offset'] % 4096 == 0 and row['offset'] < s[2] and (row['end'] - row['start'] <= (s[2] - row['offset'] + 4095) // 4096 * 4096), 'Bounded page-aligned file-backed extent required')
        require(re.fullmatch('[r-][w-][x-][ps]', row['permissions']), 'Actual mapping permissions required')
    require(any(('x' in r['permissions'] for r in rows)), 'Actual executable segment required')
    return {'association': 'explicit Btrfs anon-dev stat versus superblock VMA device', 'path_stat5': s, 'mount_device': [m['device_major'], m['device_minor']], 'mapped_segments': rows, 'covering_mount': m, 'source_sha256': expected_elf_sha256, 'process_memory_read': False, 'arbitrary_device_mismatch_accepted': False, 'mapped_memory_bytes_or_symbol_origin_qualified': False, 'actual_producer_or_runtime_qualified': False}

def capture(output,expected_sha256,phase,expected_bind_root,expected_filesystem_source,proc=Path('/proc/1'),library=Path(LIB),expected_mountpoint='/pti'):
 require(phase in('after_READY','after_last_DONE_before_QUIT'),'Declared live census boundary required');output=Path(output);require(output.is_absolute()and not any(p.is_symlink()for p in(output,*output.parents))and output.resolve()==output and not output.exists(),'New nonaliased canonical census output required');output.mkdir(parents=True)
 before=process((proc/'stat').read_text());require(before['pid']==1 and before['start_ticks']>0,'Actual native PID1 identity required');maps=(proc/'maps').read_text();rows=select(maps,str(library));mount_text=(proc/'mountinfo').read_text();mount=covering_mount(mount_text,str(library));require(library.is_file()and not any(p.is_symlink()for p in(library,*library.parents))and library.resolve()==library,'Exact mapped regular backing path required');path_before=signature(library);require(0<path_before[2]<=64<<20,'Bounded mapped ELF backing required');links=[]
 for row in rows:
  target=os.readlink(proc/'map_files'/('%x-%x'%(row['start'],row['end'])));require(target==str(library),'Actual map_files nondeleted backing link differs');links.append(target)
 with os.fdopen(os.open(library,os.O_RDONLY|os.O_NOFOLLOW),'rb')as source:
  s=os.fstat(source.fileno());require(stat.S_ISREG(s.st_mode),'Actual opened regular backing required');fd_before=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];require(fd_before==path_before,'Actual backing openedFD/path stat differs');payload=source.read((64<<20)+1);s=os.fstat(source.fileno());fd_after=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 path_after=signature(library);require(payload[:4]==b'\x7fELF'and len(payload)==path_before[2]and hashlib.sha256(payload).hexdigest()==expected_sha256,'Actual expected backing ELF bytes differ');after=process((proc/'stat').read_text());maps_after=(proc/'maps').read_text();mount_after_text=(proc/'mountinfo').read_text();mount_after=covering_mount(mount_after_text,str(library));require(after==before and select(maps_after,str(library))==rows,'Native process/mappings changed during census');links_after=[os.readlink(proc/'map_files'/('%x-%x'%(r['start'],r['end'])))for r in rows];require(links_after==links and signature(library)==path_before and not any(p.is_symlink()for p in(library,*library.parents)),'Native mapped links/path changed');binding=associate(rows,path_before,path_after,fd_before,fd_after,mount,mount_after,str(library),expected_bind_root,expected_filesystem_source,expected_sha256,expected_sha256,links,expected_mountpoint)
 (output/'mapped-injection.elf').write_bytes(payload);(output/'proc1.maps').write_text(maps,encoding='ascii');(output/'proc1.mountinfo').write_text(mount_text,encoding='ascii');(output/'proc1.mountinfo.after').write_text(mount_after_text,encoding='ascii')
 value={'schema':2,'passed':True,'phase':phase,'process':before,'observer_pid':os.getpid(),'original_container_path':str(library),'mapped_segments':rows,'backing_file_stat':path_before,'opened_fd_stat_before':fd_before,'opened_fd_stat_after':fd_after,'path_stat_after':path_after,'map_file_targets':links,'btrfs_device_domain_association':binding,'covering_mount_before':mount,'covering_mount_after':mount_after,'expected_bind_root':expected_bind_root,'expected_filesystem_source':expected_filesystem_source,'sha256':expected_sha256,'bytes':len(payload),'exported_file':'mapped-injection.elf','maps_file':'proc1.maps','mountinfo_file':'proc1.mountinfo','mountinfo_after_file':'proc1.mountinfo.after','process_memory_bytes_read':False,'LevelZero_handle_proof':False,'versioned_symbol_origin_queried':False,'mapped_memory_bytes_rehashed':False};(output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='ascii');return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--phase',required=True);p.add_argument('--expected-bind-root',required=True);p.add_argument('--expected-filesystem-source',required=True);a=p.parse_args();capture(a.output,a.expected_sha256,a.phase,a.expected_bind_root,a.expected_filesystem_source)
