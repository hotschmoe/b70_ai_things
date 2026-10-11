#!/usr/bin/env python3
"""Original metadata-only tokenizer decode, no engine/model imports."""
import sys
from pathlib import Path
if '--inside-tokenizer' not in sys.argv:raise ValueError('Only owned metadata decode entry allowed')
from qualify_api_positive_overlap_cpu_screen_v3 import inside_tokenizer
inside_tokenizer(Path(sys.argv[-3]),Path(sys.argv[-2]),Path(sys.argv[-1]))
