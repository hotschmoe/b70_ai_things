# Original executed Half37 embedded SPIRV V4

CONFIG -> unchanged executed helper 4d86d9b8, closed report 5e66575d,
public binding e1789d5c; exact original source/object/link flags and
runtime-after library identity retained. Six symbol-sized images are selected
from the original ELF, not the later save-temps recompile.

COMMAND -> pure Python ELF/SPIRV extraction and read-only admission. The default
original_binding also reruns the complete frozen public runtime/source/model
identity admission; it is not merely a small-code read. Root coordinates those
checks. The parser alone reads at most 8 MiB of regular, nonalias code with
stat/read/reread brackets. It validates the ELF section/symbol extents, unique
nonoverlapping images, SPIRV magic and instruction widths, and exact six
entrypoint names. Publication and admission require the exact six-file roster
plus extraction.json and exact current original bytes and metadata.

RESULT -> ten CPU controls pass, including the authentic original ELF and
mutated names, magic, instruction widths, out-of-range symbol extents, duplicate
entrypoints, foreign image substitution, extra files, symlinks, changed report
reads, schema/scope mutation, and exact supported translation recipes.
No agent extraction invoking complete model identity admission was performed.

VERDICT -> source-only ready for root extraction. Original embedded code cannot
establish actual JIT ISA, IGC behavior, model fidelity, or a conversion-removal
cause. Earlier recompiled post-link translations remain distinct artifacts.

## Root commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next -p 'test_half37_original_elf*_cpu_v4.py' -q
```

With strata/flash-next on PYTHONPATH, call
half37_original_elf_spirv_v4.extract(new_output_directory), then
finalized_binding(new_output_directory). Output must be a new directory.
Both calls require the unchanged closed original runtime/source proof.

half37_original_elf_text_recipe_v4.text_recipes extracts no new code: it returns
six compiler-image Docker recipes using the observed absolute llvm-spirv
--to-text option on the admitted original .spv files. Root executes them under
bin/gpu-run only after the current pair actor retires, preserving command,
inspection, image/tool identities, logs, normal EOF/removal and output hashes.
The recipes are compiler-tool translations, not kernel/model execution.

The separate C388 offline_IGC_inventory_recipe gathers actual ocloc path, ELF,
ldd and help. No unsupported ISA command or device target is guessed. A later
ISA recipe must bind observed supported help/target and actual runtime IGC
libraries from the original report. Offline ISA remains different from the
actual runtime JIT unless a separate genuine runtime witness establishes it.
All original ELF/results and failed/recompiled recipes remain unchanged.
