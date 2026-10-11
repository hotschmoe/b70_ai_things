#!/usr/bin/env python3
"""Root CPU-only fresh43 intake; cannot substitute missing runtime gates."""
import argparse,json
from pathlib import Path
from source43_num10_intake_v1 import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--upload-receipt',type=Path,required=True);p.add_argument('--oracle-receipt',type=Path,required=True);p.add_argument('--pack-receipt',type=Path,required=True);p.add_argument('--masked-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.template,a.inputs,a.upload_receipt,a.oracle_receipt,a.pack_receipt,a.masked_root,a.model_identity,a.output)
