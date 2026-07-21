"""Shared helpers for the quant eval orchestrator.

Everything that needs to be CONSISTENT across tiers lives here: the OpenAI client wiring,
endpoint health checks, full run provenance (so a result is self-describing months later),
result IO, and the retention math. Keep tier modules thin; keep policy here.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - surfaced at runtime with a clear hint
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "evals" / "results"


# ----------------------------------------------------------------------------- config
def load_models_config(path: str | Path) -> dict:
    if yaml is None:
        sys.exit("PyYAML missing: pip install -r evals/requirements.txt")
    with open(path) as fh:
        return yaml.safe_load(fh)


def find_reference(cfg: dict) -> dict | None:
    for m in cfg.get("models", []):
        if m.get("reference"):
            return m
    return None


# ----------------------------------------------------------------------------- openai client
def make_client(base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai sdk missing: pip install -r evals/requirements.txt")
    # api_key is unused by vLLM but the sdk requires a non-empty value.
    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=600.0)


def check_endpoint(base_url: str) -> dict:
    """Return {ok, models, error}. Hits /v1/models (cheap, no generation)."""
    url = base_url.rstrip("/") + "/models"
    try:
        hdrs = {}
        key = os.environ.get("OPENAI_API_KEY", "")
        if key and key != "EMPTY":
            hdrs["Authorization"] = "Bearer " + key
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        ids = [m.get("id") for m in data.get("data", [])]
        return {"ok": True, "models": ids, "error": None}
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        return {"ok": False, "models": [], "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------- provenance
def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclasses.dataclass
class RunContext:
    """One (model, quant) evaluation run. Carries everything needed for provenance + IO."""
    model_id: str          # served-model-id sent in API requests
    quant: str             # label, e.g. w8a8
    endpoint: str
    reference_model_id: str | None
    reference_endpoint: str | None
    sampling: dict
    started: str = dataclasses.field(default_factory=utc_stamp)
    git_sha: str = dataclasses.field(default_factory=get_git_sha)
    notes: str = ""

    @property
    def out_dir(self) -> Path:
        d = RESULTS_DIR / f"{self.started}__{self.model_id.replace('/', '_')}__{self.quant}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_config(self, extra: dict | None = None) -> None:
        cfg = dataclasses.asdict(self)
        cfg["endpoint_check"] = check_endpoint(self.endpoint)
        if extra:
            cfg.update(extra)
        write_json(self.out_dir / "config.json", cfg)


# ----------------------------------------------------------------------------- IO
def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- math
def retention_pct(value: float, reference: float, higher_is_better: bool = True) -> float | None:
    """Express a metric as % of the reference (the bf16 ceiling). None if reference is ~0."""
    if reference is None or value is None or abs(reference) < 1e-12:
        return None
    if higher_is_better:
        return 100.0 * value / reference
    # lower-is-better (e.g. perplexity): retention = how close we stayed, >100 means worse
    return 100.0 * reference / value
