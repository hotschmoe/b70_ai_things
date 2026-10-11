#!/usr/bin/env python3
"""Root CPU-only exact independently owned raw/gamma fixture extraction."""
import argparse
from indexer_materialized40_control_v1 import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--own-producer-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.own_producer_root,a.output)
