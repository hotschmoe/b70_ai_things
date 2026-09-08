#!/usr/bin/env python3
"""API server entry for an explicit, inherited calibration hook."""
import sys
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    sys.argv.insert(1, 'serve')
    main()
