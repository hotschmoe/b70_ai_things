"""Prepare a reviewed immutable Python-only build context; NEVER build it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE='sha256:bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf'
BASE_TAG='b70/sglang-loader-base:'+BASE.split(':')[1]
RUNNER_SHA='6159576a0506b508da274097d603d9ec25e52c439d90915f2ed54c48c4857aae'
OLD_DIGEST='d53fb6565c485658b345d0b48aaf3ea4f5e0a48748e09a24bc6a29234c026a82'
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--reviewed-digest',required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    review=json.loads(args.review.read_bytes())
    assert review['review_passed'] is True
    assert review['artifact_sha256']==args.reviewed_digest==sha(args.artifact)
    assert args.reviewed_digest!=OLD_DIGEST
    assert len(args.reviewed_digest)==64 and all(c in '0123456789abcdef' for c in args.reviewed_digest)
    original=args.source/'python/sglang/srt/model_executor/model_runner.py'
    assert sha(original)==RUNNER_SHA,'unexpected original model_runner'
    assert not args.out.exists(),'refusing to overwrite build context'
    loader=HERE.parent/'calibrated_kv'
    args.out.mkdir(parents=True)
    overlay=args.out/'overlay';overlay.mkdir()
    working=args.out/'patch-work';target=working/'python/sglang/srt/model_executor/model_runner.py'
    target.parent.mkdir(parents=True);shutil.copy2(original,target)
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-d',str(working),'-i',
                    str(loader/'sglang-main-scale-loader.patch')],check=True)
    ast.parse(target.read_text());shutil.copy2(target,overlay/'model_runner.py')
    package=overlay/'b70_calibrated_kv';package.mkdir()
    (package/'__init__.py').write_text('')
    text=(loader/'scale_loader.py').read_text()
    needle="EXPECTED_ARTIFACT_SHA256 = '"+OLD_DIGEST+"'"
    assert text.count(needle)==1,'accepted loader digest changed before deliberate update'
    replacement="EXPECTED_ARTIFACT_SHA256 = '"+args.reviewed_digest+"'"
    (package/'scale_loader.py').write_text(text.replace(needle,replacement))
    shutil.copy2(loader/'scale_plan.py',package/'scale_plan.py')
    for path in package.glob('*.py'):ast.parse(path.read_text())
    shutil.rmtree(working)
    shutil.copy2(args.artifact,args.out/'fresh-scales.json')
    shutil.copy2(args.review,args.out/'artifact-review.json')
    shutil.copy2(HERE/'native_identity.py',args.out/'native_identity.py')
    (args.out/'Dockerfile').write_text(
        'FROM '+BASE_TAG+'\n'
        'LABEL b70.calibrated_kv.artifact_sha256="'+args.reviewed_digest+'"\n'
        'COPY overlay/ /opt/venv/lib/python3.12/site-packages/sglang/srt/model_executor/\n'
        'COPY fresh-scales.json artifact-review.json /opt/b70/calibrated-kv/\n')
    manifest=dict(base_image=BASE,reviewed_artifact_sha256=args.reviewed_digest,
        original_runner_sha256=RUNNER_SHA,
        inputs={str(path.relative_to(args.out)):sha(path) for path in args.out.rglob('*') if path.is_file()},
        allowed_changed_python_files=['model_runner.py','b70_calibrated_kv/__init__.py',
            'b70_calibrated_kv/scale_loader.py','b70_calibrated_kv/scale_plan.py'],
        base_tag_command=['docker','tag',BASE,BASE_TAG],
        required_base_tag_image_id=BASE,
        build=['docker','build','--pull=false','--network=none','--iidfile',str(args.out/'image.id'),str(args.out)],
        status='prepared only; no build or native identity check executed')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
