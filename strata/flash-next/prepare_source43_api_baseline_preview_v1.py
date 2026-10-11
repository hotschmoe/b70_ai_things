#!/usr/bin/env python3
"""Reviewable configuration preview only; no API launch or global alias edit."""
import argparse
from pathlib import Path
from source43_api_baseline_profile_v1 import preview
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--normal-plan',type=Path,required=True);p.add_argument('--normal-control',type=Path,required=True);p.add_argument('--port',type=int,default=18083);p.add_argument('--output',type=Path,required=True);a=p.parse_args();preview(a.normal_plan,a.normal_control,a.port,a.output)
