#!/usr/bin/env python3
"""Root CPU-only original fixture and actual fresh41 upload prerequisite."""
import argparse,json
from pathlib import Path
from indexer_materialized41_masked_inputs_v1 import prepare
from admit_source_upload41_v1 import finalized_binding
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--own-producer-root',type=Path,required=True);p.add_argument('--fixture-output',type=Path,required=True);p.add_argument('--upload-receipt',type=Path,required=True);p.add_argument('--oracle-receipt',type=Path,required=True);p.add_argument('--pack-receipt',type=Path,required=True);p.add_argument('--prerequisite-output',type=Path,required=True);a=p.parse_args()
 if a.fixture_output.exists()or a.prerequisite_output.exists():raise ValueError('Fresh source fixture/prerequisite directories required')
 proof=finalized_binding(a.upload_receipt,a.oracle_receipt,a.pack_receipt)
 prepare(a.own_producer_root,a.fixture_output);a.prerequisite_output.mkdir();record={'upload_receipt':str(a.upload_receipt.resolve()),'oracle_receipt':str(a.oracle_receipt.resolve()),'pack_receipt':str(a.pack_receipt.resolve()),'actual_fresh41_upload_binding':proof,'model_baseline_or_constructor_qualified':False};(a.prerequisite_output/'prerequisite.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
