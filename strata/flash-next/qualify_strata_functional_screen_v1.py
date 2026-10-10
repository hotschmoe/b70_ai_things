#!/usr/bin/env python3
"""Narrow adapter to frozen strengthened batchV7 lifecycle; no lifecycle fork."""
import sys
from pathlib import Path
import strata_functional_screen_v1 as screen
import qualify_batch_numerical_v7 as lifecycle

TEMPLATE_PATH=Path(lifecycle.__file__).resolve()
def main():
 screen_source=screen.read(screen.SOURCE_PLAN);screen.require(screen.sha(TEMPLATE_PATH)==screen_source['files'][str(TEMPLATE_PATH.relative_to(screen.ROOT))],'Frozen shared parent lifecycle changed')
 lifecycle.ctrl=screen;lifecycle.BATCH_NUMERICAL_SHA=screen.sha(Path(screen.__file__));lifecycle.__file__=__file__
 # Parent snapshots this adapter and records functionalkind. Its final proof
 # grants collection/health/source only; batchV7 schema gates reject it asbatch.
 return lifecycle.main()
if __name__=='__main__':sys.exit(main())
