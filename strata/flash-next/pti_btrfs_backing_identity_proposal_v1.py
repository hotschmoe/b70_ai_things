"""Metadata-only explicit Btrfs device-domain association proposal."""
import posixpath,re
from pathlib import PurePosixPath

def require(ok,msg):
 if not ok:raise ValueError(msg)
def unescape(word):
 return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),word)
def covering_mount(text,path):
 rows=[];p=PurePosixPath(path)
 for line in text.splitlines():
  left,sep,right=line.partition(' - ');require(sep,'Complete mountinfo line required');before=left.split();after=right.split();require(len(before)>=6 and len(after)>=3,'Complete mountinfo fields required');mp=PurePosixPath(unescape(before[4]))
  if p==mp or mp in p.parents:
   dev=before[2].split(':');require(len(dev)==2 and all(v.isdigit()for v in dev),'Typed mount major/minor required');rows.append({'mount_id':int(before[0]),'parent_id':int(before[1]),'device_major':int(dev[0]),'device_minor':int(dev[1]),'root':unescape(before[3]),'mountpoint':str(mp),'mount_options':before[5].split(','),'filesystem':after[0],'source':unescape(after[1]),'super_options':after[2].split(','),'original_line':line})
 require(rows,'Actual covering mount required');length=max(len(r['mountpoint'])for r in rows);chosen=[r for r in rows if len(r['mountpoint'])==length];require(len(chosen)==1,'Unique actual longest covering mount required');return chosen[0]
def associate(rows,path_stat_before,path_stat_after,fd_stat_before,fd_stat_after,mount_before,mount_after,expected_path,expected_bind_root,expected_filesystem_source,actual_elf_sha256,expected_elf_sha256,map_links,expected_mountpoint='/pti'):
 """No process queries/file reads here. Actual producer must supply every row."""
 require(path_stat_before==path_stat_after==fd_stat_before==fd_stat_after,'Current openedFD/path exact stat5 before/after required');s=path_stat_before;require(type(s)is list and len(s)==5 and all(type(v)is int and v>=0 for v in s)and s[1]>0 and 0<s[2]<=64<<20,'Typed current regular ELF stat5 required');require(actual_elf_sha256==expected_elf_sha256 and re.fullmatch('[0-9a-f]{64}',expected_elf_sha256),'Actual expected backing ELF SHA required');require(mount_before==mount_after,'Actual covering mount must stay unchanged');m=mount_before;require(m['filesystem']=='btrfs'and m['source']==expected_filesystem_source and m['root']==expected_bind_root and m['mountpoint']==expected_mountpoint and 'ro'in m['mount_options']and 'subvolid=5'in m['super_options'],'Exact admitted readonly Btrfs install bind/subvolume required');require(rows and len(rows)==len(map_links),'Complete actual mapped segment/link roster required');last=0
 for row,link in zip(rows,map_links):
  require(all(type(row[n])is int and row[n]>=0 for n in('start','end','offset','inode','device_major','device_minor'))and row['start']>=last and row['end']>row['start'],'Typed ordered mapped ranges required');last=row['end'];require(row['original_container_path']==expected_path and link==expected_path,'Exact nondeleted native map/map_files path required');require(row['inode']==s[1]and row['device_major']==m['device_major']and row['device_minor']==m['device_minor'],'Mapped inode and Btrfs superblock/mount device must match');require(row['offset']%4096==0 and row['offset']<s[2]and row['end']-row['start']<=((s[2]-row['offset']+4095)//4096)*4096,'Bounded page-aligned file-backed extent required');require(re.fullmatch('[r-][w-][x-][ps]',row['permissions']),'Actual mapping permissions required')
 require(any('x'in r['permissions']for r in rows),'Actual executable segment required')
 return {'association':'explicit Btrfs anon-dev stat versus superblock VMA device','path_stat5':s,'mount_device':[m['device_major'],m['device_minor']],'mapped_segments':rows,'covering_mount':m,'source_sha256':expected_elf_sha256,'process_memory_read':False,'arbitrary_device_mismatch_accepted':False,'mapped_memory_bytes_or_symbol_origin_qualified':False,'actual_producer_or_runtime_qualified':False}
