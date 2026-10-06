"""Copy the Collated v2 split manifests into this workspace and lock them.

Every image referenced by the manifests must exist and match its recorded
SHA-256. The result is recorded in manifests/FROZEN.json; later scripts refuse
to run if the frozen files no longer match that record.

    python freeze_manifests.py            # first freeze
    python freeze_manifests.py --check    # verify the frozen copy is intact
    python freeze_manifests.py --force    # re-freeze after upstream changes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import (FROZEN_RECORD, MANIFEST_DIR, MANIFEST_FILES, SOURCE_ROOT,
                    UPSTREAM_MANIFEST_DIR)

IMAGE_MANIFESTS = ("hellostreetcat_stage2.csv", "pawsadaan_stage3.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def manifest_hashes(folder: Path) -> dict[str, str]:
    return {name: sha256(folder / name) for name in MANIFEST_FILES}


def verify_images(folder: Path) -> dict:
    problems: list[str] = []
    summary = {}
    for name in IMAGE_MANIFESTS:
        rows = read_rows(folder / name)
        for row in rows:
            image = SOURCE_ROOT / row["image_path"]
            if not image.is_file():
                problems.append(f"missing: {row['image_path']}")
            elif sha256(image) != row["sha256"]:
                problems.append(f"changed: {row['image_path']}")
        summary[name] = {
            "rows": len(rows),
            "identities": len({row["identity"] for row in rows}),
            "rows_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
            "extensions": dict(sorted(Counter(Path(row["image_path"]).suffix.lower()
                                              for row in rows).items())),
        }
    if problems:
        print(f"{len(problems)} image problem(s):", file=sys.stderr)
        for problem in problems[:50]:
            print("  " + problem, file=sys.stderr)
        sys.exit(1)
    return summary


def check() -> None:
    if not FROZEN_RECORD.is_file():
        sys.exit("No frozen manifests yet; run: python freeze_manifests.py")
    record = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))
    if manifest_hashes(MANIFEST_DIR) != record["manifest_sha256"]:
        sys.exit("Frozen manifests were modified after freezing.")
    verify_images(MANIFEST_DIR)
    upstream = manifest_hashes(UPSTREAM_MANIFEST_DIR)
    if upstream != record["manifest_sha256"]:
        print("Note: upstream manifests have changed since freezing "
              "(the frozen copy is still intact).")
    print("Frozen manifests and all referenced images verified.")


def freeze(force: bool) -> None:
    upstream = manifest_hashes(UPSTREAM_MANIFEST_DIR)
    if FROZEN_RECORD.is_file():
        record = json.loads(FROZEN_RECORD.read_text(encoding="utf-8"))
        if record["manifest_sha256"] == upstream:
            print("Already frozen and identical to upstream; nothing to do.")
            return
        if not force:
            sys.exit("Upstream manifests differ from the frozen copy. Re-freezing "
                     "changes the splits under every existing result; pass --force "
                     "only if that is intended.")

    summary = verify_images(UPSTREAM_MANIFEST_DIR)
    MANIFEST_DIR.mkdir(exist_ok=True)
    for name in MANIFEST_FILES:
        shutil.copy2(UPSTREAM_MANIFEST_DIR / name, MANIFEST_DIR / name)
    if manifest_hashes(MANIFEST_DIR) != upstream:
        sys.exit("Upstream manifests changed while copying; run again.")

    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "upstream": UPSTREAM_MANIFEST_DIR.relative_to(SOURCE_ROOT).as_posix(),
        "manifest_sha256": upstream,
        "summary": summary,
    }
    FROZEN_RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the frozen copy")
    mode.add_argument("--force", action="store_true", help="overwrite a differing frozen copy")
    args = parser.parse_args()
    check() if args.check else freeze(args.force)


if __name__ == "__main__":
    main()
