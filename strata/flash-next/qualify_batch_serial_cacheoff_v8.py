#!/usr/bin/env python3
"""CacheOFF serial supplement through unchanged strengthened V7 parent lifecycle."""
from pathlib import Path
import batch_serial_cacheoff_v8 as serial
import qualify_batch_numerical_v7 as shared
TEMPLATE=Path(shared.__file__).resolve()
def main():
 serial.require(serial.sha(TEMPLATE)==serial.read(serial.SOURCE_PLAN)['files'][str(TEMPLATE.relative_to(serial.ROOT))],'Frozen shared V7 lifecycle changed');shared.ctrl=serial;shared.BATCH_NUMERICAL_SHA=serial.sha(Path(serial.__file__));shared.__file__=__file__;return shared.main()
if __name__=='__main__':raise SystemExit(main())
