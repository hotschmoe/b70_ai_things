"""NEW source40 independent fresh49 parent with captured plan bytes."""
from pathlib import Path
import qualify_full_cache_shared_runtime_v3 as template
import qualify_batch_numerical_v7 as original
import full_cache_shared_batch0_fresh40_v3 as ctrl

def main():
 ctrl.source_binding();source=template.adapted_source();needle='import full_cache_shared_runtime_v3 as ctrl';ctrl.require(source.count(needle)==1,'Exact new captured-plan parent integration required');source=source.replace(needle,'import full_cache_shared_batch0_fresh40_v3 as ctrl')
 namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='owned_source40_fresh49_parent_v3',ctrl=ctrl,BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__)));exec(compile(source,str(Path(original.__file__))+'[NEW-source40-fresh49-captured-plan]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
