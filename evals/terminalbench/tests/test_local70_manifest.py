from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals.terminalbench.local70_manifest import (
    DEFAULT_MANIFEST,
    DEFAULT_TASK_ROOT,
    EXCLUDED_TASKS,
    ManifestError,
    build_manifest,
    render_manifest,
    validate_manifest,
)


class Local70ManifestTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        names = [f"local-task-{index:02d}" for index in range(70)]
        names.extend(EXCLUDED_TASKS)
        for name in reversed(names):
            task_dir = root / name
            task_dir.mkdir()
            (task_dir / "task.toml").write_text(
                f'[task]\nname = "terminal-bench/{name}"\n', encoding="ascii"
            )

    def test_build_is_deterministic_and_shards_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_fixture(root)
            first = build_manifest(root)
            second = build_manifest(root)

        self.assertEqual(first, second)
        self.assertEqual(first["all_task_count"], 74)
        self.assertEqual(first["local_task_count"], 70)
        self.assertEqual(
            [entry["name"] for entry in first["excluded_tasks"]],
            list(EXCLUDED_TASKS),
        )
        self.assertEqual(len(first["shards"]), 14)
        self.assertEqual({len(shard["tasks"]) for shard in first["shards"]}, {5})
        flattened = [task for shard in first["shards"] for task in shard["tasks"]]
        self.assertEqual(flattened, [entry["name"] for entry in first["tasks"]])

    def test_validation_fails_after_task_toml_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                render_manifest(build_manifest(root)), encoding="ascii"
            )
            validate_manifest(root, manifest_path)
            changed = root / "local-task-00" / "task.toml"
            changed.write_text(changed.read_text(encoding="ascii") + "# changed\n")
            with self.assertRaisesRegex(ManifestError, "does not match"):
                validate_manifest(root, manifest_path)

    def test_missing_h100_exclusion_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_fixture(root)
            missing = root / EXCLUDED_TASKS[0]
            (missing / "task.toml").unlink()
            missing.rmdir()
            replacement = root / "unexpected-local-task"
            replacement.mkdir()
            (replacement / "task.toml").write_text("[task]\n", encoding="ascii")
            with self.assertRaisesRegex(ManifestError, "missing required H100"):
                build_manifest(root)

    def test_tracked_manifest_matches_current_tb3_source(self) -> None:
        manifest = validate_manifest(DEFAULT_TASK_ROOT, DEFAULT_MANIFEST)
        self.assertEqual(manifest["local_task_count"], 70)
        self.assertEqual(len(manifest["excluded_tasks"]), 4)
        self.assertTrue(
            all(4 <= len(shard["tasks"]) <= 6 for shard in manifest["shards"])
        )
        encoded = DEFAULT_MANIFEST.read_text(encoding="ascii")
        self.assertEqual(json.loads(encoded), manifest)


if __name__ == "__main__":
    unittest.main()
