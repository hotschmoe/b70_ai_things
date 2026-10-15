CONFIG -> Frozen S97 QT17 public image versus saved S98 public image from failed audit.

COMMAND -> Bounded Python ELF section parsing/SHA and byte offsets; retained S97 actual ASM read. No decoder, compiler, GPU, payload, trace or archive-wide read.

RESULT -> Full image 43272 -> 43240 bytes, identity FAIL retained. Target 7488 bytes differs at exactly 7308..7311 and 7324..7327. Bytes 0..7295 are exact, including EOT at 7280. Retained actual ASM: after EOT, L7296 contains MOV-to-null trailer immediates; only E307351B/F3758ADE become DE39F167/976AF49F. Helper text has the same two immediate changes at 92/108. Codebook, all ze_info, options, symbols, relocation sections, compat and strtab are exact. SPIR-V 24212 -> 24180 bytes has different IDs/order; section offsets after it shift -32. Exact section hashes and changed u32 values are in comparison.json.

VERDICT -> No material emitted computational instruction difference before EOT. This is an additive explanation, NOT a changed full-image/raw-text identity PASS. Existing original public decode need not be repeated for this bounded prefix claim. Root decides whether this stronger direct emitted-code/metadata equality suffices for the unchanged public anchor. Actual private/public full output bytes remain future runtime gates.
