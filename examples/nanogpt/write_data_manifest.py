from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--output", default="manifest.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name == args.output:
            continue
        size = path.stat().st_size
        entry = {"name": path.name, "path": str(path), "bytes": size, "sha256": sha256_file(path)}
        if path.suffix == ".bin":
            entry["dtype"] = "uint16"
            entry["tokens"] = size // 2
        files.append(entry)
    manifest = {"path": str(data_dir), "files": files}
    output = Path(args.output)
    if not output.is_absolute():
        output = data_dir / output
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
