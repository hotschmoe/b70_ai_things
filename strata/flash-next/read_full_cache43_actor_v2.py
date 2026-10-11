#!/usr/bin/env python3
"""Concrete root public command; whole-suite/fresh comparisons stay explicit."""
import argparse,json
from pathlib import Path
from validate_full_cache43_runtime_v2 import finalized_binding
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--fresh-control',type=Path,action='append');p.add_argument('--output',type=Path,required=True);a=p.parse_args();value=finalized_binding(a.root,fresh_control_roots=a.fresh_control);a.output.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
