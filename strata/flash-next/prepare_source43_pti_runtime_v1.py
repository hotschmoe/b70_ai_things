#!/usr/bin/env python3
"""ROOT intake from actual GPU0 property capability; never fills absent proof."""
import argparse,json
from pathlib import Path
import source43_pti_runtime_v1 as runtime
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device-properties',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();runtime.require(not a.output.exists(),'Fresh PTI runtime intake required');a.output.write_text(json.dumps(runtime.intake(a.device_properties),indent=2)+'\n');runtime.binding(a.output)
