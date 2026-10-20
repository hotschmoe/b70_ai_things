"""Import B1 pack_gptq without pulling AutoRound's full package.

auto_round/__init__.py imports accelerate. The bitstream helpers live in
one file that only needs torch. Register stub parents so pack_gptq's
`from auto_round.utils.bit_packing import ...` loads that file alone.
Do not edit vllm/flashnext/quant/.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_AR_ROOT = Path("/mnt/vm_8tb/b70/build/vllm-flashnext/auto-round")
_FLASHNEXT = Path("/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext")


def preload_bit_packing() -> None:
    pkg_root = _AR_ROOT / "auto_round"
    if not (pkg_root / "utils" / "bit_packing.py").is_file():
        raise FileNotFoundError(pkg_root / "utils" / "bit_packing.py")
    if "auto_round" not in sys.modules:
        pkg = types.ModuleType("auto_round")
        pkg.__path__ = [str(pkg_root)]
        pkg.__package__ = "auto_round"
        sys.modules["auto_round"] = pkg
    if "auto_round.utils" not in sys.modules:
        utils = types.ModuleType("auto_round.utils")
        utils.__path__ = [str(pkg_root / "utils")]
        utils.__package__ = "auto_round.utils"
        sys.modules["auto_round.utils"] = utils


def import_pack_gptq():
    preload_bit_packing()
    flash = str(_FLASHNEXT)
    if flash not in sys.path:
        sys.path.insert(0, flash)
    from quant.pack_gptq import (  # noqa: WPS433
        dequant_linear,
        maxq_for,
        pack_linear,
        quantize_sym_codes,
        unpack_qweight,
    )

    return dequant_linear, maxq_for, pack_linear, quantize_sym_codes, unpack_qweight
