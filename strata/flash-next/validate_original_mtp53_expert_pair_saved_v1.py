"""Read-only reconstruction of actual pair-reuse model evidence; no GPU/model IO."""
import argparse
import hashlib
import json
from pathlib import Path
import run_original_mtp53_expert_pair_model_v3 as reader

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def normalize(value):
    # Parsed segment maps are tuples; saved JSON represents them as arrays.
    return json.loads(json.dumps(value))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    producer = json.loads((root / 'report.json').read_bytes())
    require(producer['passed'] and not producer['errors'] and producer['full64_equivalence']
            and producer['resolved_residency_equal'], 'Actual successful producer required')
    component = json.loads((root / 'historical-OFF-component.json').read_bytes())
    reports = [Path(name) for name in component['consumed'] if Path(name).name == 'report.json']
    require(len(reports) == 1, 'Unique consumed historical producer required')
    off_root = reports[0].parent
    old = reader.historical_off(off_root, component['original_report_sha256'],
                                producer['compile_receipt_sha256'], producer['elf_sha256'])['actual']
    require(normalize(old) == producer['arm_results'][0]['actual'], 'Historical OFF reconstruction differs')
    recipe = json.loads((root / 'ON/recipe.json').read_bytes())
    profile = next(Path(src) for src, dst in recipe['mounts'].items() if dst == '/profiles/expert-profile.bin')
    text = (root / 'ON/native.stdout').read_text(encoding='ascii')
    actual = reader.controls(text, 54)
    actual['host_phases'] = reader.phase_rows(text, actual)
    actual['resolved_residency'] = reader.residency((root / 'ON/native.stderr').read_text(encoding='ascii'), profile)
    on = producer['arm_results'][1]
    require(normalize(actual) == on['actual'], 'Actual ON reconstruction differs')
    require(actual['ids'] == old['ids'] and actual['natural_eos'] == old['natural_eos'], 'Complete IDs/EOS differ')
    require(normalize(actual['resolved_residency']) == normalize(old['resolved_residency']), 'Residency differs')
    for name, want in on['raw_sha256'].items():
        require(sha(root / 'ON' / name) == want, 'Actual ON raw digest differs')
    terminal = json.loads((root / 'ON/terminal.json').read_bytes())
    require(isinstance(terminal, dict) and terminal['State']['ExitCode'] == 0
            and not terminal['State']['Running'] and on['owned_removed'], 'Normal owned terminal required')
    for name in ('pre-full5.json', 'post-full5.json'):
        rows = json.loads((root / name).read_bytes())
        require(isinstance(rows, list) and len(rows) == 5 and len({r['path'] for r in rows}) == 5,
                'Complete five-file identity roster required')
        require(all(r['passed'] and r['sha256'] == r['expected_sha256'] and r['before'] == r['after'] for r in rows),
                'Actual stable complete identity scan required')
    off_ms = float(old['summary']['decode_wall_ms'])
    on_ms = float(actual['summary']['decode_wall_ms'])
    report = dict(passed=True, producer_report_sha256=sha(root / 'report.json'), reader_sha256=sha(Path(__file__)),
                  all64_IDs_EOS_equal=True, residency_profile_mirror_equal=True, OFF_decode_wall_ms=off_ms,
                  ON_decode_wall_ms=on_ms, elapsed_reduction_percent=100 * (1 - on_ms / off_ms),
                  separated_run_diagnostic_scope=True, model_speed_qualified=False, original_fullmath_qualified=False)
    require(args.output.is_absolute() and not args.output.exists(), 'New external public output required')
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='ascii')
    print(json.dumps(report, ensure_ascii=True))

if __name__ == '__main__':
    main()
