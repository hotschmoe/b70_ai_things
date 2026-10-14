# Warm PTI V5 private command-builder correction

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Warm V4 failed before model creation because copied life.contract still
pointed to warm V3's native_command globals, whose runtime was unset. The V4
failure and original source remain unchanged. V5 rebinds that one contract
callback to its private current actor native_command before constructing ports.
The OFF-only short-read0 builder never enters the runtime mount branch; actual
saved-metadata command construction proves it works without a runtime value.
No short-read0 source change is needed for this correction.

COMMAND: Use the V4 ON retry guide, replacing only run_source43_warm_pti_v4.py
with run_source43_warm_pti_v5.py and validate_source43_warm_pti_v4.py with
validate_source43_warm_pti_v5.py, with a genuinely new ON output directory.
The same original closed OFF component, successful loaded fixture report and
paired lease remain required. No OFF model rerun.

V5 constructs the actual recipe/native command from exact small saved template
and input metadata plus the freshly admitted fixture envelope BEFORE current
model admission, full4 or health. Construction adds no Docker/GPU launch.
The original current admission and per-arm fresh model/source/health/EOF gates
still run afterward. The original construction proof is saved and independently
reconstructed by the public reader. Private callback identity and the actual
fresh readonly /pti install/helper mounts are checked, not merely AST shape.

RESULT: Three bounded CPU tests execute the actual builder and preflight using
saved real metadata/runtime, verify preflight precedes current admission in the
compiled parent, and execute the unaffected OFF construction. No GPU/model or
provider invocation by the author.

VERDICT: Actual ON capture remains pending. No device coverage, alignment,
physical idle, clean serving latency, API or full-model fidelity promotion.
