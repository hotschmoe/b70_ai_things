#!/usr/bin/env python3
"""Hidden checker for re-vm (verifier container; the original vmrun does not exist there)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from mscore import Milestones, cli_milestones

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
MS = [("m01", "header, --info, command line and exit codes"), ("m02", "stack ops, immediates, halt/exit codes"),
      ("m03", "integer arithmetic and bit ops (wraparound, shifts, div/mod)"), ("m04", "comparisons, jumps, loops"),
      ("m05", "locals and globals"), ("m06", "call/ret, frames, recursion"), ("m07", "strings and conversions"),
      ("m08", "arrays and heap semantics"), ("m09", "stdin/stdout builtins"), ("m10", "runtime traps: exit codes and messages"),
      ("m11", "malformed and corrupt program files"), ("m12", "instruction budget (-b)"),
      ("m13", "obfuscated encodings (flags 1,2,3)"), ("m14", "random programs and stress (limits, depth, heap)")]
M = Milestones()
cli_milestones(M, [f"{SUB}/vmrun"], HID, MS, timeout=30)
M.emit()
