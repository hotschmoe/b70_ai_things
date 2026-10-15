# S62 recorded CPU reference sources

These exact sources produced the selected diagnostic in
[the dated report](../../../../docs/20261015_strata_grouped_gu_cpu_reference.md).
They use the existing pinned ggml CPU archive and saved S42 fixtures. They do
not launch GPU work or change the failed S42 gate.

The scripts retain their actual absolute evidence paths and expect external
raw inputs. They are recorded experiment sources, not a general reference
framework. For another run, prepare a new external output directory with these
three files and preserve the existing raw results. Wrap heavy CPU work in
`bin/cpu-run`. The host has no g++; the pinned compiler image has icpx and no
/usr/bin/time. Source oneAPI setup before strict shell options.

The raw run records preserve exact compile/run commands, image/source/archive
hashes, FTZ/DAZ/rounding checks and the original setup failures. The first
helper enumerates operand classes; the second exports only selected normal-
scale rows. selected_reference.py computes exact integer dyadic dots and
reports errors without an acceptance tolerance or GPU-operand identity claim.
