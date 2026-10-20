"""CPU reference for Flash-Next pieces that qwen3_5_moe does not have.

PLE n-gram hash and IQ4_NL gather, hyper-connection mix/combine, QSA
indexer selection, and the MTP fc layout. Ground truth is the Strata
night-20261016 headers under
/mnt/vm_8tb/b70/build/strata-ab/src-ep/include/strata/kernels/.
"""

from .geometry import FLASHNEXT, PleConsts, artifact_ple_consts

__all__ = ["FLASHNEXT", "PleConsts", "artifact_ple_consts"]
