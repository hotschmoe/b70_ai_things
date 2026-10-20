"""Collect existing Strata greedy outputs for the vLLM harness.

Reads strata/flash-next, docs/, and the hotschmoe-dd result dir. Does not
call the live server. UD-Q4 campaign files are kept and labelled as a
different stack.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/mnt/vm_8tb/github/b70_ai_things")
OUT = ROOT / "vllm/flashnext/harness/strata_reference.json"
ROOTS = (
    ROOT / "strata/flash-next",
    ROOT / "docs",
    Path("/mnt/vm_8tb/b70/results/hotschmoe-dd"),
)
MAX_READ = 4_000_000
MAX_TEXT = 2000
MAX_RECORDS = 160


def _stack(path: str) -> str:
    low = path.lower()
    if "udq4" in low or "ud-q4" in low:
        return "ud-q4-paused"
    if "hotschmoe-dd" in low or "flash-next" in low:
        return "ista-iq3-dd"
    return "unknown"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_of(request: dict) -> str | None:
    messages = request.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(f"{msg.get('role', '')}: {content}")
            elif isinstance(content, list):
                bits = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        bits.append(item["text"])
                if bits:
                    parts.append(f"{msg.get('role', '')}: {' '.join(bits)}")
        if parts:
            return "\n".join(parts)
    if isinstance(request.get("prompt"), str):
        return request["prompt"]
    return None


def _clip(text: str) -> dict:
    if len(text) <= MAX_TEXT:
        return {"text": text, "text_len": len(text), "truncated": False}
    return {
        "text": text[:MAX_TEXT],
        "text_len": len(text),
        "truncated": True,
        "sha256_full": _sha(text),
    }


def _from_obj(obj: dict, path: str) -> dict | None:
    request = obj.get("request") if isinstance(obj.get("request"), dict) else obj
    text = obj.get("text")
    reasoning = None
    if text is None and isinstance(obj.get("choices"), list) and obj["choices"]:
        choice = obj["choices"][0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                text = msg["content"]
                if isinstance(msg.get("reasoning_content"), str):
                    reasoning = msg["reasoning_content"]
    sha = obj.get("output_sha256")
    if not isinstance(text, str) and not isinstance(sha, str):
        return None
    prompt = _prompt_of(request) if isinstance(request, dict) else None
    if prompt is None and not isinstance(text, str):
        return None
    temp = None
    if isinstance(request, dict):
        temp = request.get("temperature")
    model = None
    if isinstance(request, dict):
        model = request.get("model")
    model = model or obj.get("model")
    think = None
    if isinstance(request, dict):
        kwargs = request.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
            think = kwargs.get("enable_thinking")
    rec = {
        "path": path,
        "stack": _stack(path),
        "model": model,
        "temperature": temp,
        "prompt": prompt,
        "enable_thinking": think,
        "sha256_stored": sha if isinstance(sha, str) else None,
    }
    if isinstance(reasoning, str) and reasoning:
        rec["reasoning_len"] = len(reasoning)
        rec["reasoning_content"] = reasoning if len(reasoning) <= 500 else reasoning[:500]
    usage = obj.get("usage")
    if isinstance(usage, dict):
        rec["usage"] = {
            k: usage.get(k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    timings = obj.get("timings")
    if isinstance(timings, dict):
        rec["predicted_n"] = timings.get("predicted_n")
        rec["draft_n"] = timings.get("draft_n")
        rec["draft_n_accepted"] = timings.get("draft_n_accepted")
    if isinstance(prompt, str) and len(prompt) > 400:
        rec["prompt_len"] = len(prompt)
        rec["prompt_sha256"] = _sha(prompt)
        rec["prompt"] = prompt[:400]
        rec["prompt_truncated"] = True
    if isinstance(text, str):
        rec.update(_clip(text))
        digest = _sha(text)
        rec["sha256_of_text"] = digest
        if isinstance(sha, str):
            rec["sha256_match"] = digest == sha
    else:
        rec["text"] = None
        rec["sha_only"] = True
    return rec


def _walk_json(obj, path: str, sink: list, seen: set) -> None:
    if len(sink) >= MAX_RECORDS:
        return
    if isinstance(obj, list):
        for item in obj:
            _walk_json(item, path, sink, seen)
            if len(sink) >= MAX_RECORDS:
                return
        return
    if not isinstance(obj, dict):
        return
    if isinstance(obj.get("request"), dict) and isinstance(obj.get("response"), dict):
        merged = dict(obj["response"])
        merged["request"] = obj["request"]
        wrapped = _from_obj(merged, path)
        if wrapped is not None and wrapped.get("text"):
            key = (
                wrapped.get("prompt"),
                wrapped.get("sha256_of_text") or wrapped.get("sha256_stored"),
                wrapped.get("stack"),
                wrapped.get("temperature"),
            )
            if key not in seen:
                seen.add(key)
                sink.append(wrapped)
            return
    rec = _from_obj(obj, path)
    if rec is not None and (rec.get("text") or rec.get("sha_only")):
        # A chat-completion object also contains nested choices. Keep the
        # parent record and do not walk into stream chunks.
        if "request" in obj or "choices" in obj or "output_sha256" in obj:
            key = (
                rec.get("prompt"),
                rec.get("sha256_of_text") or rec.get("sha256_stored"),
                rec.get("stack"),
                rec.get("temperature"),
            )
            if key not in seen:
                seen.add(key)
                sink.append(rec)
            return
    for value in obj.values():
        if isinstance(value, (dict, list)):
            _walk_json(value, path, sink, seen)


def _scan_file(path: Path, sink: list, notes: list, seen: set) -> None:
    if len(sink) >= MAX_RECORDS:
        return
    try:
        size = path.stat().st_size
    except OSError as exc:
        notes.append(f"{path}: {exc}")
        return
    name = path.name.lower()
    if size > MAX_READ and "functional" not in str(path):
        notes.append(f"skip {path} size={size}")
        return
    if path.suffix.lower() == ".jsonl":
        _scan_jsonl(path, sink, notes, seen)
        return
    if path.suffix.lower() != ".json":
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"{path}: {type(exc).__name__}")
        return
    before = len(sink)
    _walk_json(data, str(path), sink, seen)
    if name.endswith(".json") and len(sink) == before and size < 200_000:
        # Keep a pointer when the file names a greedy run but stores no text.
        if "hotschmoe-dd" in str(path) and ("pair" in name or "functional" in str(path)):
            notes.append(f"no text in {path}")


def _scan_jsonl(path: Path, sink: list, notes: list, seen: set) -> None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        notes.append(f"{path}: {exc}")
        return
    for line in lines[:400]:
        line = line.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        _walk_json(data, str(path), sink, seen)
        if len(sink) >= MAX_RECORDS:
            return


def _scan_docs(notes: list) -> list:
    """Only lines that quote a fixed prompt or its greedy answer."""
    hits = []
    docs = ROOT / "docs"
    if not docs.is_dir():
        return hits
    needles = (
        "17 * 19",
        "23 * 29",
        "capital of France",
        "Reply only with the city",
    )
    for path in docs.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if "flashnext" not in str(path).lower() and "2026101" not in path.name:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if any(n in line for n in needles):
                hits.append({"path": str(path), "line": i, "text": line.strip()[:240]})
                if len(hits) >= 20:
                    return hits
    return hits


def _scan_engine(notes: list) -> dict:
    path = Path("/mnt/vm_8tb/b70/results/hotschmoe-dd/engine.log")
    info = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info["bytes"] = path.stat().st_size
    # Bounded scan. The engine log is traces, not chat completions.
    needles = (b"Paris", b'"content"', b"completion_tokens", b"323")
    found = {n.decode(): 0 for n in needles}
    try:
        with path.open("rb") as handle:
            scanned = 0
            while scanned < 2_000_000:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                scanned += len(chunk)
                for needle in needles:
                    found[needle.decode()] += chunk.count(needle)
        info["scanned_bytes"] = scanned
        info["counts"] = found
    except OSError as exc:
        notes.append(f"engine.log: {exc}")
    return info


def main() -> int:
    sink: list = []
    notes: list = []
    seen: set = set()
    n_files = 0
    for root in ROOTS:
        if not root.exists():
            notes.append(f"missing {root}")
            continue
        if root.is_file():
            _scan_file(root, sink, notes, seen)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules"}]
            for name in filenames:
                if not name.endswith((".json", ".jsonl")):
                    continue
                n_files += 1
                _scan_file(Path(dirpath) / name, sink, notes, seen)
                if len(sink) >= MAX_RECORDS:
                    notes.append("record cap hit")
                    break
            if len(sink) >= MAX_RECORDS:
                break
    # Prefer temperature 0 with text. Drop sha-only duplicates of a stored prompt.
    dedup = []
    seen = set()
    for rec in sink:
        key = (rec.get("prompt"), rec.get("sha256_of_text") or rec.get("sha256_stored"), rec.get("stack"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rec)
    text_recs = [r for r in dedup if r.get("text")]
    sha_only = [r for r in dedup if r.get("sha_only")]
    temp0 = [r for r in text_recs if r.get("temperature") in (0, 0.0)]
    doc_hits = _scan_docs(notes)
    engine = _scan_engine(notes)
    payload = {
        "purpose": "Fixed-prompt greedy references for C7. Collected from existing logs. No live server call.",
        "n_json_files_seen": n_files,
        "n_text_records": len(text_recs),
        "n_temp0": len(temp0),
        "n_sha_only": len(sha_only),
        "records": temp0 + [r for r in text_recs if r not in temp0] + sha_only[:20],
        "doc_mentions": doc_hits,
        "engine_log": engine,
        "notes": notes[:40],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {OUT}")
    print(f"files={n_files} text={len(text_recs)} temp0={len(temp0)} sha_only={len(sha_only)}")
    for rec in temp0[:12]:
        prompt = (rec.get("prompt") or "")[:60].replace("\n", " ")
        text = (rec.get("text") or "")[:40].replace("\n", " ")
        print(f"  t0 {rec.get('model')} sha_ok={rec.get('sha256_match')} {prompt!r} -> {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
