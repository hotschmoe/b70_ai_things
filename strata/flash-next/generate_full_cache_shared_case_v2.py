"""Authenticate root-owned CPU tokenizer receipts into a NEW source-only case."""
import argparse
from pathlib import Path
from full_cache_shared_admission_v2 import generate

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.fixture_root,a.output)
if __name__=='__main__':main()
