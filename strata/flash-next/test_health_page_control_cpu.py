#!/usr/bin/env python3
"""Preserved good/bad page admission controls; no current source/cache/GPU changes."""
import json
from pathlib import Path
from health_page_control_contract import validate_boundary,OFFSET

BASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009')


def main():
    good=(BASE/'shard3-cache-discriminator-v1/3857879040.direct.bin').read_bytes()
    bad=(BASE/'shard3-cache-recurrence-v1/direct.bin').read_bytes()
    stat={'device':51,'inode':15212298,'size':49376141504,'mtime_ns':1791502883931301133,'ctime_ns':1791502888380438568}
    assert validate_boundary(good,good,stat,stat,OFFSET,1000,1001)['source_page_original']
    mutated=bytearray(good);mutated[2796]^=0x20;assert bytes(mutated)==bad
    cases=[('preservedbadcached',bad,good,stat,stat,OFFSET,1000,1001),
        ('preservedbaddirect',good,bad,stat,stat,OFFSET,1000,1001),
        ('onebitbothviews',bad,bad,stat,stat,OFFSET,1000,1001),
        ('shortread',good[:-1],good,stat,stat,OFFSET,1000,1001),
        ('wrongoffset',good,good,stat,stat,OFFSET+4096,1000,1001),
        ('changedstat',good,good,stat,{**stat,'ctime_ns':stat['ctime_ns']+1},OFFSET,1000,1001),
        ('stalefullhash',good,good,stat,stat,OFFSET,0,1001),
        ('futurefullhash',good,good,stat,stat,OFFSET,1002,1001)]
    rejected=[]
    for name,*args in cases:
        try:validate_boundary(*args)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Negative boundary admitted: '+name)
    print(json.dumps({'mode':'CPU_PRESERVED_PAGES_ONLY','passed':len(rejected)==8,'negative_controls':rejected,
        'current_source_or_cache_modified':False,'gpu_operations':False,'cause_proven':False},sort_keys=True))


if __name__=='__main__':main()
