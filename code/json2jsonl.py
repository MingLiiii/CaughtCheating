#!/usr/bin/env python3
"""
json_to_jsonl.py
================
Convert a standard JSON file to a JSONL (JSON Lines) file.

Usage
-----
python json_to_jsonl.py --input_file input.json --output_file output.jsonl

Arguments
---------
--input_file   Path to the input JSON file.
--output_file  Path where the generated JSONL file will be saved.
"""

import argparse
import json
import pathlib
import sys


def json_to_jsonl(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Read a JSON file at *src* and write its contents to *dst* in JSONL format."""
    # 1) Load the entire JSON file
    try:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Failed to read {src}: {e}")

    # 2) If the top-level element is a single object, wrap it in a list
    if isinstance(data, dict):
        data = [data]

    # 3) The top-level element must be iterable (list/tuple)
    if not isinstance(data, (list, tuple)):
        sys.exit("❌ The top-level element of the input JSON must be an object or an array!")

    # 4) Write each object on its own line in the output file
    with dst.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON ➜ JSONL converter")
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to the output JSONL file",
    )
    args = parser.parse_args()

    src = pathlib.Path(args.input_file).expanduser().resolve()
    dst = pathlib.Path(args.output_file).expanduser().resolve()

    if not src.exists():
        sys.exit(f"❌ Input file not found: {src}")

    json_to_jsonl(src, dst)
    print(f"✅ Conversion complete. Output written to {dst}")


if __name__ == "__main__":
    main()
