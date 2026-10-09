#!/usr/bin/env python3
"""Operator-only explicit zero-staged-PLE MODIFIED original reference exploration.
Same V9/source admission and postfull4 gates; no exact/native/fullmath claim.
"""
from pathlib import Path
import explore_full48_original_prefix1_v1 as original_driver
from full48_zero_staged_ple_ablation_v1 import ZeroStagedPleComposition,PROVENANCE


def main():
 old_write=original_driver.write;old_binding=original_driver.dependency_binding;old_composition=original_driver.Full48OwnedComposition
 def labelled_write(path,value):
  if Path(path).name=='report.json':
   value['reference_ablation']=dict(PROVENANCE);value['modified_reference_driver_sha256']=original_driver.sha(Path(__file__));value['modified_reference_adapter_sha256']=original_driver.sha(Path(__file__).with_name('full48_zero_staged_ple_ablation_v1.py'))
  old_write(path,value)
 def binding():
  value=old_binding();value.update({str(Path(__file__)):original_driver.sha(Path(__file__)),str(Path(__file__).with_name('full48_zero_staged_ple_ablation_v1.py')):original_driver.sha(Path(__file__).with_name('full48_zero_staged_ple_ablation_v1.py'))});return value
 # New entry point selects an explicit source subclass in this private process;
 # frozen source files and the original entry point are never edited.
 original_driver.Full48OwnedComposition=ZeroStagedPleComposition
 original_driver.write=labelled_write;original_driver.dependency_binding=binding
 try:return original_driver.main()
 finally:
  original_driver.Full48OwnedComposition=old_composition;original_driver.write=old_write;original_driver.dependency_binding=old_binding
if __name__=='__main__':raise SystemExit(main())
