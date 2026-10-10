"""Hardpinned actual closed V7 target prerequisite; captured fields target-only."""
from pathlib import Path
import native_rms_device_ops_proposal_v1 as ops
from serial37_canonical_json_v3 import canonical,read_unique
PRIOR_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/native-rms-rsqrt37-owned-v7-run-v1')
EXTERNAL=PRIOR_ROOT.parent/'native-rms-rsqrt37-owned-v7-readonly-binding-v1.json'
REPORT_SHA='58da27b488ba02c77d9717977d68152a25fd6cff21f278a2336149daa56b462c'
EXTERNAL_SHA='c45571e440fb72e1a9a863c35331fa4b79800b15275692dcdfee2a05344c2ba1'
def header(root):
 root=Path(root).resolve();ops.require(root==PRIOR_ROOT,'Exact actual closed V7 root required');ops.require(ops.sha((root/'report.json').read_bytes())==REPORT_SHA and ops.sha(EXTERNAL.read_bytes())==EXTERNAL_SHA,'Original successful V7 report/public binding changed');return root
def admit_prior(root):
 root=header(root);proof=ops.prior_binding(root);ops.require(proof['finalized_binding']['report_sha256']==REPORT_SHA and canonical(proof['finalized_binding'])==canonical(read_unique(EXTERNAL)),'Actual current V7 public admission contradicts original independent binding');report=read_unique(root/'report.json');proof['fixture_binding']=report['fixture_binding'];proof['external_binding_sha256']=EXTERNAL_SHA;return proof
def match_fixture(prior,fixture):ops.require(canonical(prior['fixture_binding'])==canonical(fixture),'Device operations must use exact original closed V7 fixture')
