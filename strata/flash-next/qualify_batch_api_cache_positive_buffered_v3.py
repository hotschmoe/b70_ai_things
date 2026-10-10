"""NEW buffered observer wrapper through unchanged owned V7 lifecycle."""
from pathlib import Path
import batch_api_cache_positive_buffered_v3 as ctrl
import qualify_batch_numerical_v7 as shared
TEMPLATE=Path(shared.__file__).resolve()
def main():
 ctrl.require(ctrl.sha(TEMPLATE)==ctrl.read(ctrl.SOURCE_PLAN)['files'][str(TEMPLATE.relative_to(ctrl.ROOT))],'Frozen owned parent differs');shared.ctrl=ctrl;shared.BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__));shared.__file__=__file__;return shared.main()
if __name__=='__main__':raise SystemExit(main())
