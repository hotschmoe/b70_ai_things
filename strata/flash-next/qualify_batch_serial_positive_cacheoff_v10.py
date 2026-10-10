"""NEW positive numerical serial adapter to unchanged owned V7 lifecycle."""
from pathlib import Path
import batch_serial_positive_cacheoff_v10 as ctrl
import qualify_batch_numerical_v7 as shared
TEMPLATE=Path(shared.__file__).resolve()
def main():
 ctrl.require(ctrl.sha(TEMPLATE)==ctrl.read(ctrl.SOURCE_PLAN)['files'][str(TEMPLATE.relative_to(ctrl.ROOT))],'Frozen owned V7 parent source changed');shared.ctrl=ctrl;shared.BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__));shared.__file__=__file__;return shared.main()
if __name__=='__main__':raise SystemExit(main())
