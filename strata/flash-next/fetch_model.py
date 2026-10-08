#!/usr/bin/env python3
"""Download the pinned Flash-Next files and verify publisher identities (CPU only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name("model-lock.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    destination = REPO / lock["destination"]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"repo": lock["repo"], "revision": lock["revision"],
               "destination": str(destination), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "status": "running", "verified": []}

    def save():
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.receipt)

    save()
    try:
        if not args.verify_only:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
            from huggingface_hub import snapshot_download
            print("Downloading pinned model to", destination, flush=True)
            snapshot_download(repo_id=lock["repo"], revision=lock["revision"],
                              allow_patterns=[f["path"] for f in lock["files"]],
                              local_dir=destination, max_workers=2)
        for entry in lock["files"]:
            path = destination / entry["path"]
            if path.is_symlink() or path.stat().st_size != entry["size"]:
                raise ValueError("File type or size mismatch: " + entry["path"])
            digest = hashlib.sha256() if "sha256" in entry else hashlib.sha1()
            if "git_blob" in entry:
                digest.update(("blob " + str(entry["size"]) + "\0").encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            expected = entry.get("sha256", entry.get("git_blob"))
            if digest.hexdigest() != expected:
                raise ValueError("Hash mismatch: " + entry["path"])
            receipt["verified"].append({"path": entry["path"], "digest": digest.hexdigest(), "size": entry["size"]})
            save()
            print("Verified", entry["path"], flush=True)
        receipt["status"] = "verified"
    except Exception as error:
        # Exception type is sufficient here; do not persist signed URLs or credentials.
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        print("Intake failed:", type(error).__name__, flush=True)
        raise SystemExit(1) from None
    finally:
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
