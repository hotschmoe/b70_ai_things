"""No Docker or GPU execution: exercise sequencing and fail-closed post-health."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import oracle_campaign as campaign

for card in (0, 1):
    command = campaign.oracle_command(card)
    assert command[0] == 'docker'  # Must resolve via the ownership wrapper PATH.
    assert '--memory' in command and command[command.index('--memory') + 1] == '8g'
    assert 'ZE_AFFINITY_MASK=' + str(card) in command
    assert str(campaign.SOURCE) + ':/source:ro' in command
    assert campaign.IMAGE in command


def exercise(fail_oracle=False, preflight_failure=False):
    events = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'campaign'
        def preflight():
            events.append('preflight')
            root.mkdir()
            if preflight_failure:
                return 1
            (root / 'PASS').write_text('health only')
            return 0
        def run(command, out, timeout, owned=False):
            assert owned and timeout == 420
            card = '0' if 'ZE_AFFINITY_MASK=0' in command else '1'
            events.append('oracle' + card)
            out.write_text('mocked')
            return 1 if fail_oracle else 0
        def parse(_):
            manifest = json.loads((root / 'input-manifest.json').read_text())
            return {'source_hashes': manifest['source_hashes'],
                    'loader_source_sha256': manifest['inputs']['calibrated_kv/scale_loader.py']}
        def post(_):
            events.append('post')
            return {'mock': True}, True
        with patch.object(campaign, 'ROOT', root), \
             patch.object(campaign.health, 'main', preflight), \
             patch.object(campaign.health, 'run', run), \
             patch.object(campaign, 'parse_oracle', parse), \
             patch.object(campaign, 'post_health', post):
            rc = campaign.execute()
        if preflight_failure:
            assert rc == 1 and events == ['preflight']
        elif fail_oracle:
            assert rc == 1 and events == ['preflight', 'oracle0', 'post']
            assert not json.loads((root / 'OUTCOME.json').read_text())['passed']
        else:
            assert rc == 0 and events == ['preflight', 'oracle0', 'oracle1', 'post']
            assert json.loads((root / 'OUTCOME.json').read_text())['passed']


exercise()
exercise(fail_oracle=True)
exercise(preflight_failure=True)
print('PASS two pinned 8GiB commands, one preflight, sequential oracles, failure post-health, preflight stop')
