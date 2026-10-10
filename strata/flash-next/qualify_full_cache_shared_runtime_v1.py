"""NEW parent sharing exact V5 journal/EOF/source lifecycle, no frozen edits."""
from pathlib import Path
import qualify_batch_api_cache_positive_buffered_v5 as template
import qualify_batch_numerical_v7 as original
import full_cache_shared_runtime_v1 as ctrl
def adapted_source():
 source=template.adapted_source();needle='import batch_api_cache_positive_buffered_v5 as ctrl';ctrl.require(source.count(needle)==1,'Exact current strong lifecycle integration failed');return source.replace(needle,'import full_cache_shared_runtime_v1 as ctrl')
def main():
 ctrl.source_binding();source=adapted_source();namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='owned_full_cache_shared_parent_v1',ctrl=ctrl,BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__)));exec(compile(source,str(Path(original.__file__))+'[NEW-fullcache-current-journal-EOF]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
