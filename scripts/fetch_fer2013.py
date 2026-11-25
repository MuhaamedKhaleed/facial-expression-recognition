"""Download the pinned FER2013 challenge CSV mirror; verify bytes before use.

Fetches ``icml_face_data.csv`` from a pinned Hugging Face dataset revision and
checks the full SHA-256 before writing the output marker. This is the original
ICML 2013 challenge CSV (``emotion, Usage, pixels``), not the cleaned CSV this
repository's CLI accepts; run ``scripts/prepare_fer2013.py`` next. No training,
no import side effects, refuses an existing output directory.
"""

from __future__ import annotations

REPOSITORY = "DerrickUnleashed/FER-2013"
REVISION = "5b1d92d29dea477a4c13af5f928b2baee3744951"
FILENAME = "icml_face_data.csv"
EXPECTED_BYTES = 301072768
EXPECTED_SHA256 = "2c7ce224ef03ec2e2e522d10b93a069900b26fa1c028174d91d5f1f8013a2884"
URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"


def main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory.")
    import hashlib
    import json
    import urllib.request

    args.output.mkdir(exist_ok=False)
    target = args.output / FILENAME
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp, target.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    actual = digest.hexdigest()
    if total != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        target.unlink(missing_ok=True)
        raise SystemExit(
            f"Source checksum mismatch: {FILENAME} "
            f"(bytes {total}, sha256 {actual}); "
            f"expected {EXPECTED_BYTES} bytes, sha256 {EXPECTED_SHA256}. "
            "The pinned mirror changed or the download was corrupted; "
            "do not train on it."
        )
    print(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "file": FILENAME,
                "bytes": total,
                "sha256": actual,
                "output": str(target),
            }
        )
    )


if __name__ == "__main__":
    main()
