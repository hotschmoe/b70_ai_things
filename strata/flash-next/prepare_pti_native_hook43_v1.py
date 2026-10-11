"""Source-only opt-in loaded PTI finalizer hook over exact source42 proposal."""
import difflib,hashlib,json
from pathlib import Path
import prepare_host_terminal42_v1 as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
G=parent.G;HEADER='sycl/include/strata/program/pti_native_finalize43.hpp';PATCH=HERE/'patches/0043-opt-in-owned-pti-native-terminal-hook.patch'
def require(ok,message):
    if not ok:raise ValueError(message)
def changes():
    _,source42=parent.changes();old={G:source42[G]};body=old[G]
    require(body.count('#include <cstdio>')==1,'Exact global native cstdio include required');body=body.replace('#include <cstdio>','#include <cstdio>\n#include "strata/program/pti_native_finalize43.hpp"')
    oldclose=parent.REPLACEMENT;newclose=oldclose.replace('        std::fflush(stdout);','''        if (!strata::program::diagnostic_pti_finalize43()) {
            std::fprintf(stderr,"strata diagnostic PTI finalization failed\\n");
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::fflush(stdout);''')
    require(body.count(oldclose)==1,'Exact explicit H36 source42 terminal point required');body=body.replace(oldclose,newclose)
    require(body.replace(newclose,oldclose).replace('#include <cstdio>\n#include "strata/program/pti_native_finalize43.hpp"','#include <cstdio>')==old[G],'Exact native-hook byte undo required')
    return old,{G:body,HEADER:(HERE/'pti_native_finalize43_v1.hpp').read_text()}
def patch_text():
    old,new=changes();return ''.join(''.join(difflib.unified_diff(old.get(name,'').splitlines(True),text.splitlines(True),fromfile='a/'+name if name in old else '/dev/null',tofile='b/'+name))for name,text in new.items())
