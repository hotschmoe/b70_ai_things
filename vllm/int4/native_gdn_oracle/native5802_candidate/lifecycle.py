#!/usr/bin/env python3
"""Candidate identity gate plus the unchanged owned rolling-contract lifecycle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    original = Path(plan['original_lifecycle'])
    assert hashlib.sha256(original.read_bytes()).hexdigest() == plan['frozen_files'][str(original)]
    if args.run:
        assert Path(plan['pair_preflight_pass']).is_file()
        assert Path(plan['pair_preflight_pass']).parent in Path(plan['pair_preflight_image_inspect']).parents
        inspected = json.loads(Path(plan['pair_preflight_image_inspect']).read_text())
        assert len(inspected) == 1 and inspected[0]['Id'] == plan['image'], 'candidate pair health image mismatch'
        assert plan['contract'] == 'unchanged original rolling6; all 12 checks; 128 steps'
    spec = importlib.util.spec_from_file_location('original_owned_gdn_lifecycle', original)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    # Its main retains signal handling, lease checks, labeled cleanup, and
    # strict selected-card post-health, including numerical failure paths.
    return original_module.main()


if __name__ == '__main__':
    raise SystemExit(main())
