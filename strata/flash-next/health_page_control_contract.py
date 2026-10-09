"""CPU-only admission contract for future isolated health/page observations."""
import hashlib

OFFSET=3857879040
PAGE_BYTES=4096
PAGE_SHA='2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e'
STAT_KEYS=('device','inode','size','mtime_ns','ctime_ns')


def validate_boundary(cached,direct,before,after,offset,source_hash_finished_epoch,observed_epoch):
    if offset!=OFFSET or len(cached)!=PAGE_BYTES or len(direct)!=PAGE_BYTES:
        raise ValueError('Health page boundary offset/extent differs')
    if any(before.get(k)!=after.get(k) or type(before.get(k)) is not int for k in STAT_KEYS):
        raise ValueError('Source stat changed or incomplete at health boundary')
    if hashlib.sha256(cached).hexdigest()!=PAGE_SHA or hashlib.sha256(direct).hexdigest()!=PAGE_SHA:
        raise ValueError('Source fidelity invalid at health boundary; preserve both views and stop GPU/model work')
    if not 0<=observed_epoch-source_hash_finished_epoch<=600:
        raise ValueError('Full source hash provenance is stale or after boundary')
    return {'source_page_original':True,'cached_direct_equal':cached==direct,'stat_unchanged':True,
            'source_hash_provenance_fresh':True,'GPU_launch_allowed_by_this_check_only':False}
