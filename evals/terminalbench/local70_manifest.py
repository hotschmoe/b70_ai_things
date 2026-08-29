#!/usr/bin/env python3
"""Generate or validate the deterministic Terminal-Bench 3 local-70 lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DATASET = "terminal-bench/terminal-bench"
DATASET_VERSION = "3.0.0"
CAMPAIGN = "TB3-local-70"
EXPECTED_TOTAL = 74
EXPECTED_LOCAL = 70
SHARD_SIZE = 5
EXCLUDED_TASKS = (
    "exam-pdf-eval",
    "fp8-rmsnorm-gemm",
    "jax-speedrun-gpu",
    "math-eval-grader",
)
DEFAULT_TASK_ROOT = Path("/mnt/vm_8tb/b70/evals/terminal-bench")
DEFAULT_MANIFEST = Path(__file__).with_name("tb3_local70_manifest.json")


class ManifestError(ValueError):
    """The source tasks or lock manifest violate the local-70 contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry_digest(entries: list[dict[str, str]]) -> str:
    payload = b"".join(
        entry["name"].encode("ascii")
        + b"\0"
        + entry["task_toml_sha256"].encode("ascii")
        + b"\n"
        for entry in entries
    )
    return sha256_bytes(payload)


def discover_tasks(task_root: Path) -> list[dict[str, str]]:
    if not task_root.is_dir():
        raise ManifestError(f"task root is not a directory: {task_root}")

    paths = sorted(task_root.glob("*/task.toml"), key=lambda path: path.parent.name)
    entries = [
        {
            "name": path.parent.name,
            "task_toml_sha256": sha256_bytes(path.read_bytes()),
        }
        for path in paths
    ]
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ManifestError("duplicate task directory names")
    if len(entries) != EXPECTED_TOTAL:
        raise ManifestError(
            f"expected {EXPECTED_TOTAL} task.toml files, found {len(entries)}"
        )
    missing = sorted(set(EXCLUDED_TASKS) - set(names))
    if missing:
        raise ManifestError(f"missing required H100 exclusions: {', '.join(missing)}")
    return entries


def build_manifest(task_root: Path) -> dict[str, Any]:
    all_entries = discover_tasks(task_root)
    excluded_names = set(EXCLUDED_TASKS)
    excluded = [entry for entry in all_entries if entry["name"] in excluded_names]
    local = [entry for entry in all_entries if entry["name"] not in excluded_names]
    if len(excluded) != len(EXCLUDED_TASKS):
        raise ManifestError("the H100 exclusion set is not exact")
    if len(local) != EXPECTED_LOCAL:
        raise ManifestError(f"expected {EXPECTED_LOCAL} local tasks, found {len(local)}")

    shards = []
    for index in range(0, len(local), SHARD_SIZE):
        shard_entries = local[index : index + SHARD_SIZE]
        if not 4 <= len(shard_entries) <= 6:
            raise ManifestError(
                f"shard {index // SHARD_SIZE + 1} has invalid size {len(shard_entries)}"
            )
        shards.append(
            {
                "id": f"tb3-local-70-{index // SHARD_SIZE + 1:02d}",
                "tasks": [entry["name"] for entry in shard_entries],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "campaign": CAMPAIGN,
        "hash_algorithm": "sha256",
        "hash_scope": "task.toml bytes only",
        "all_task_count": len(all_entries),
        "local_task_count": len(local),
        "all_tasks_digest": entry_digest(all_entries),
        "local_tasks_digest": entry_digest(local),
        "excluded_tasks": excluded,
        "tasks": local,
        "shards": shards,
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError(f"manifest is not a JSON object: {path}")
    return data


def validate_manifest(task_root: Path, manifest_path: Path) -> dict[str, Any]:
    expected = build_manifest(task_root)
    observed = load_manifest(manifest_path)
    if observed != expected:
        raise ManifestError(
            f"manifest does not match task.toml source; regenerate {manifest_path}"
        )
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="write the deterministic manifest instead of validating it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.generate:
            manifest = build_manifest(args.task_root)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(render_manifest(manifest), encoding="ascii")
            action = "generated"
        else:
            manifest = validate_manifest(args.task_root, args.manifest)
            action = "validated"
    except ManifestError as error:
        print(f"local70 manifest error: {error}", file=sys.stderr)
        return 1

    print(
        f"{action} {args.manifest}: local={manifest['local_task_count']} "
        f"excluded={len(manifest['excluded_tasks'])} "
        f"shards={len(manifest['shards'])} "
        f"digest={manifest['local_tasks_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
