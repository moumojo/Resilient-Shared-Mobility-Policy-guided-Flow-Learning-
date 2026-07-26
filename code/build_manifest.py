from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

try:
    from .run_test import discover_checkpoints
except ImportError:
    from run_test import discover_checkpoints


FIELDS = [
    "checkpoint_id",
    "family",
    "variant",
    "city",
    "availability",
    "ablation",
    "checkpoint_bytes",
    "index_bytes",
    "data_bytes",
    "checkpoint_sha256",
    "index_sha256",
    "data_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.stat().st_size, sha256(path)


def build_rows(result_root: Path):
    rows = []
    for spec in discover_checkpoints(result_root):
        state_path = spec.prefix.parent / "checkpoint"
        index_path = spec.prefix.with_suffix(".index")
        data_path = spec.prefix.parent / (spec.prefix.name + ".data-00000-of-00001")
        state_size, state_hash = file_info(state_path)
        index_size, index_hash = file_info(index_path)
        data_size, data_hash = file_info(data_path)
        rows.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "family": spec.family,
                "variant": spec.variant,
                "city": spec.city,
                "availability": spec.availability,
                "ablation": spec.ablation,
                "checkpoint_bytes": state_size,
                "index_bytes": index_size,
                "data_bytes": data_size,
                "checkpoint_sha256": state_hash,
                "index_sha256": index_hash,
                "data_sha256": data_hash,
            }
        )
    return rows


def render_csv(rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parse_args():
    package_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate or verify the released checkpoint manifest.")
    parser.add_argument("--result-root", default=str(package_root / "result"))
    parser.add_argument("--output", default=str(package_root / "CHECKPOINTS.csv"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(Path(args.result_root))
    rendered = render_csv(rows)
    output = Path(args.output)
    if args.check:
        if not output.is_file():
            raise SystemExit(f"Manifest does not exist: {output}")
        if output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Checkpoint manifest is stale or a released file has changed")
        print(f"Verified {len(rows)} checkpoint entries in {output}")
        return 0
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write(rendered)
    print(f"Wrote {len(rows)} checkpoint entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
