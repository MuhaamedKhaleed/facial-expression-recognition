"""Reviewed preparation of the raw FER2013 challenge CSV for this repository.

Reads the pinned raw ``icml_face_data.csv`` (columns ``emotion, Usage, pixels``,
with the original header spacing), validates every row under the same contract
as ``fer_research.core.read``, removes exact-duplicate pixel rows by keeping the
first occurrence, and writes a normalized ``emotion,pixels,Usage`` CSV that the
CLI accepts. Every removal is counted in a JSON ledger; conflicting duplicates
(same pixels, different label or partition) are reported separately, never
silently dropped. Raw file and ledger stay local; no faces are redistributed.
"""

from __future__ import annotations

PARTITIONS = ("Training", "PublicTest", "PrivateTest")


def main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory.")
    if not args.input.is_file():
        p.error("Raw CSV input required.")
    import csv
    import hashlib
    import json

    kept_rows: list[tuple[str, str, str]] = []
    seen: dict[bytes, tuple[str, str]] = {}
    usage_counts = {k: 0 for k in PARTITIONS}
    removed_same = 0
    removed_conflict = 0
    total = 0
    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        header = [h.strip() for h in next(rows)]
        if sorted(header) != ["Usage", "emotion", "pixels"]:
            raise SystemExit(
                f"Unexpected raw header {header}; expected emotion/Usage/pixels."
            )
        cols = [header.index(name) for name in ("emotion", "pixels", "Usage")]
        for i, row in enumerate(rows, 2):
            total += 1
            if len(row) != 3:
                raise SystemExit(f"Malformed row{i}.")
            emotion, pixels, usage = (row[c].strip() for c in cols)
            if usage not in PARTITIONS:
                raise SystemExit(f"Unknown Usage at row{i}; no random fallback split.")
            try:
                label = int(emotion)
                pix = [int(x) for x in pixels.split()]
            except ValueError as exc:
                raise SystemExit(f"Noninteger label/pixels at row{i}.") from exc
            if (
                not 0 <= label < 7
                or len(pix) != 2304
                or any(v < 0 or v > 255 for v in pix)
            ):
                raise SystemExit(
                    f"Need label0..6 and2304 integer pixels0..255 at row{i}."
                )
            key = hashlib.sha256(bytes(pix)).digest()
            if key in seen:
                if seen[key] == (emotion, usage):
                    removed_same += 1
                else:
                    removed_conflict += 1
                continue
            seen[key] = (emotion, usage)
            usage_counts[usage] += 1
            kept_rows.append((emotion, pixels, usage))
    args.output.mkdir(exist_ok=False)
    target = args.output / "fer2013.csv"
    with target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerows((e, p, u) for e, p, u in kept_rows)
    in_digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    out_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    ledger = {
        "input": str(args.input),
        "input_sha256": in_digest,
        "output": str(target),
        "output_sha256": out_digest,
        "rows_total": total,
        "rows_kept": len(kept_rows),
        "duplicates_removed": removed_same + removed_conflict,
        "duplicates_same_label_partition": removed_same,
        "duplicates_conflicting_label_or_partition": removed_conflict,
        "policy": "keep first occurrence; all removals counted here",
        "usage_counts": usage_counts,
    }
    (args.output / "ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(ledger))


if __name__ == "__main__":
    main()
