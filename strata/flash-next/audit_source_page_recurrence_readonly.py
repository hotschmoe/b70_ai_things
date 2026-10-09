#!/usr/bin/env python3
"""Independent read-only page recurrence readers; no invalidation/source writes/GPU."""
import hashlib,json,mmap,os,shutil,subprocess,tempfile,time
from pathlib import Path

BASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009')
PARENT=BASE/'shard3-cache-recurrence-v1'
C=r'''
#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
 if(argc!=5)return 2;
 int direct=atoi(argv[2]),mode=atoi(argv[3]),pattern=atoi(argv[4]);
 off_t offset=3857879040;size_t size=mode==2?8192:4096;
 if(mode==2)offset-=4096;
 int fd=open(argv[1],O_RDONLY|(direct?O_DIRECT:0));if(fd<0){perror("open");return 3;}
 fprintf(stderr,"pid=%d flags=%d direct=%d mode=%d offset=%lld bytes=%zu\n",getpid(),fcntl(fd,F_GETFL),direct,mode,(long long)offset,size);
 if(mode==3){void*p=mmap(NULL,4096,PROT_READ,MAP_PRIVATE,fd,offset);if(p==MAP_FAILED){perror("mmap");return 4;}if(write(STDOUT_FILENO,p,4096)!=4096)return 5;munmap(p,4096);}
 else {void*p=NULL;if(posix_memalign(&p,16384,size))return 6;memset(p,pattern,size);ssize_t n=pread(fd,p,size,offset);if(n!=(ssize_t)size){perror("pread");return 7;}
 fprintf(stderr,"buffer=%p returned=%zd\n",p,n);if(write(STDOUT_FILENO,(char*)p+(mode==2?4096:0),4096)!=4096)return 8;free(p);}
 close(fd);return 0;
}
'''
def main():
 prior=json.loads((PARENT/'receipt.json').read_text());source=Path(prior['path'])
 out=Path(tempfile.mkdtemp(prefix='shard3-recurrence-independent-readonly-',dir=BASE));(out/'reader.c').write_text(C)
 compile_cmd=['gcc','-O2','-Wall','-Wextra',str(out/'reader.c'),'-o',str(out/'reader')]
 if not shutil.which('gcc'):
  compile_cmd=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
   '--entrypoint','gcc','-v',str(out)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7',
   '-O2','-Wall','-Wextra','/work/reader.c','-o','/work/reader']
 subprocess.run(compile_cmd,check=True)
 observations=[];before=source.stat()
 for label,direct,mode,pattern in [('buffered_a',0,1,0xA5),('direct_a',1,1,0xA5),('direct_two_pages_b',1,2,0x5A),('private_readonly_mmap',0,3,0),('buffered_b',0,1,0x5A),('direct_b',1,1,0x3C)]:
  t=time.time();r=subprocess.run([str(out/'reader'),str(source),str(direct),str(mode),str(pattern)],capture_output=True,check=True)
  assert len(r.stdout)==4096
  (out/(label+'.bin')).write_bytes(r.stdout)
  observations.append(dict(label=label,epoch=t,pid_flags_buffer_read_result=r.stderr.decode('ascii'),bytes=len(r.stdout),sha256=hashlib.sha256(r.stdout).hexdigest(),byte2796=r.stdout[2796],matches_preserved_recurrence=r.stdout==(PARENT/'direct.bin').read_bytes()))
 mappings=[];unreadable=0
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   lines=(proc/'maps').read_text().splitlines();matched=[l for l in lines if source.name in l]
   if matched:mappings.append(dict(pid=int(proc.name),maps=matched,comm=(proc/'comm').read_text().strip()))
  except (PermissionError,FileNotFoundError,ProcessLookupError):unreadable+=1
 commands={}
 for key,cmd in [('mount',['findmnt','-T',str(source),'-o','TARGET,SOURCE,FSTYPE,OPTIONS']),('extent',['filefrag','-v',str(source)]),('kernel',['uname','-a'])]:
  r=subprocess.run(cmd,capture_output=True,text=True);(out/(key+'.txt')).write_text(r.stdout+r.stderr);commands[key]=dict(command=cmd,exit_code=r.returncode,file=str(out/(key+'.txt')))
 after=source.stat()
 def stat(s):return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 receipt=dict(CONFIG='CPU read-only independent C process/buffer readers; no cache invalidation, source write or GPU',COMMAND='python3 strata/flash-next/audit_source_page_recurrence_readonly.py',RESULT=observations,VERDICT='Independent observation only; equal direct/buffered bad bytes do not prove healthy storage or corruption origin',path=str(source),offset=prior['offset'],before=stat(before),after=stat(after),stat_unchanged=stat(before)==stat(after),process_source_mappings=mappings,unreadable_processes=unreadable,commands=commands,parent_receipt_sha256=hashlib.sha256((PARENT/'receipt.json').read_bytes()).hexdigest())
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('RECEIPT',out/'receipt.json')
 for row in observations:print(row['label'],row['sha256'],'byte2796',hex(row['byte2796']))
 print('STAT_UNCHANGED',receipt['stat_unchanged'],'MAPPED_PROCESSES',len(mappings))
if __name__=='__main__':main()
