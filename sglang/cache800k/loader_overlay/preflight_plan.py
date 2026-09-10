"""Print loader-image pair preflight plan; --run requires both inherited leases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal

HERE=Path(__file__).resolve().parent
HELPER=HERE.parents[1]/'refresh/20260910_main/preflight.py'
HELPER_SHA='85e23da18f6922af3366f83028439fd14eb629843bb13edd52e38ae8ccdd739c'
ROOT=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-loader-fresh-preflight')
IMAGE='sha256:f82a10b2c3d04f10b230ba299dced23455366f307d96bdce54a7143264b41a99'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true')
    args=p.parse_args()
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA,'owned health helper changed; review before running'
    if not args.run:
        print(json.dumps(dict(image=IMAGE,output=str(ROOT),helper=str(HELPER),helper_sha256=HELPER_SHA,
            stages=['strict per-card health (both physical cards)','compiled two-rank P2P0 collective health'],
            ownership='both bin/gpu-run leases; inherited helper verifies/removes only labeled owned containers',
            recovery='helper reset only after owned containers absent; baseline per-card/pair post-recovery health',
            launch=['bin/gpu-run','python3',str(Path(__file__).resolve()),'--run'],
            state='plan only; no GPU launch'),indent=2))
        return 0
    spec=importlib.util.spec_from_file_location('b70_owned_sglang_preflight',HELPER)
    health=importlib.util.module_from_spec(spec);spec.loader.exec_module(health)
    health.ROOT=ROOT
    health.IMAGES=[('sglang-loader-fresh',IMAGE)]
    def interrupted(signum,frame):
        if health.CLEANUP_PENDING:return
        raise KeyboardInterrupt('loader preflight interrupted by signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
    return health.main()

if __name__=='__main__':raise SystemExit(main())
