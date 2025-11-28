#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def collect_jsons(src_dir: Path, dst_dir: Path, preserve_structure: bool, overwrite: bool) -> int:
    if not src_dir.is_dir():
        raise SystemExit(f"Input directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for recording in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        json_files = list(recording.glob("*.json"))
        if not json_files:
            continue

        if preserve_structure:
            target_base = dst_dir / recording.name
            target_base.mkdir(parents=True, exist_ok=True)
            for jf in json_files:
                target = target_base / jf.name
                if target.exists() and not overwrite:
                    raise SystemExit(f"Refusing to overwrite existing file: {target}")
                shutil.copy2(jf, target)
                copied += 1
        else:
            for jf in json_files:
                target_name = f"{recording.name}_{jf.name}"
                target = dst_dir / target_name
                if target.exists() and not overwrite:
                    raise SystemExit(f"Refusing to overwrite existing file: {target}")
                shutil.copy2(jf, target)
                copied += 1

    return copied

def main():
    parser = argparse.ArgumentParser(description="Collect recording JSON files into one folder.")
    parser.add_argument("-i", "--input", default="recordings", help="Source recordings directory")
    parser.add_argument("-o", "--output", default="recordings_json", help="Destination directory")
    parser.add_argument("--preserve-structure", action="store_true",
                        help="Keep per-recording subfolders instead of flattening with prefixes")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting destination files")
    args = parser.parse_args()

    copied = collect_jsons(Path(args.input), Path(args.output),
                           preserve_structure=args.preserve_structure,
                           overwrite=args.overwrite)
    print(f"Copied {copied} JSON file(s) into {args.output}")

if __name__ == "__main__":
    main()
