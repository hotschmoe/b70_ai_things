#!/usr/bin/env python3
"""Root CPU-only extraction from the closed independently owned projection fixture."""
import argparse
from owned_indexer_half_control_v2 import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original-own-fixture',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.original_own_fixture,a.output)
