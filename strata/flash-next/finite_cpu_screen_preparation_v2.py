"""Root CPU-metadata preparation; exact frozen recipe plus a genuine new fixture."""
import argparse
import copy
from pathlib import Path
import produce_api_positive_overlap_corpus_v2 as export
import prepare_api_positive_overlap_cpu_screen_v2 as requests
from serial37_canonical_json_v3 import canonical, read_unique
HERE = Path(__file__).resolve().parent
RECIPE = HERE / 'api-positive-overlap-cpu-screen-source-plan-v2.json'


def source_recipe():
    recipe = read_unique(RECIPE)
    export.require(recipe['schema'] == 'api-positive-overlap-finite-screen-source-v2'
                   and recipe['actual_model_inference'] is False,
                   'Exact unexecuted finite source recipe required')
    for path, digest in recipe['source_bindings'].items():
        export.require(export.sha(path) == digest, 'Current finite screen source changed ' + path)
    export.require(recipe['fixture_root'] is None and recipe['preregistered_screen_plan'] is None,
                   'Frozen source recipe cannot fabricate a future runtime fixture')
    return recipe


def expected_plan(recipe, fixture_root, fixture, fixture_binding, preregistered_path):
    expected_requests = requests.screen_plan(fixture, fixture_binding, export.read(export.SEED))
    actual_requests = read_unique(preregistered_path)
    export.require(canonical(actual_requests) == canonical(expected_requests),
                   'Exact actual new fixture/all-eight preregistration changed')
    result = copy.deepcopy(recipe)
    result.update(schema='api-positive-overlap-finite-prepared-screen-v2',
                  fixture_root=str(Path(fixture_root).resolve()),
                  authentic_fixture_binding=fixture_binding,
                  frozen_recipe_source_sha256=export.sha(RECIPE),
                  preregistered_screen_plan={'path': str(Path(preregistered_path).resolve()),
                                            'sha256': export.sha(preregistered_path)})
    export.require(result['corpus_messages'] == [r['messages'] for r in fixture['fixtures']['warm'] + fixture['fixtures']['target']],
                   'Actual finite corpus differs from frozen eight tasks')
    return result


def admit_prepared(path):
    original = Path(path)
    export.require(original.is_file() and not original.is_symlink(), 'Original regular prepared plan required')
    raw = original.read_bytes()
    plan = read_unique(original)
    recipe = source_recipe()
    fixture, binding = export.finalized_binding(plan['fixture_root'])
    expected = expected_plan(recipe, plan['fixture_root'], fixture, binding,
                             plan['preregistered_screen_plan']['path'])
    export.require(canonical(plan) == canonical(expected), 'Prepared plan has undeclared recipe/identity/guard changes')
    export.require(original.read_bytes() == raw and source_recipe() == recipe,
                   'Prepared/source recipe changed during admission')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export.require(not args.output.exists() and not args.preregistration.exists(), 'Both prepared files must be new')
    recipe = source_recipe()
    fixture, binding = export.finalized_binding(args.fixture)
    export.write(args.preregistration, requests.screen_plan(fixture, binding, export.read(export.SEED)))
    export.write(args.output, expected_plan(recipe, args.fixture, fixture, binding, args.preregistration))
    admit_prepared(args.output)
    print('Prepared authentic finite all16 CPU screen; no inference or prior-result transfer')


if __name__ == '__main__':
    main()
