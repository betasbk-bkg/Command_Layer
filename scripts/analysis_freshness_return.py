"""Compatibility entry point for the v1.5 freshness-return analysis.

The old flat-bootstrap implementation was retired. This script now regenerates
`reports/freshness_return_cellpreserving_v15.json` using the same
cell-preserving relay-frontier routine exposed by `analyze_relay_frontier_v15.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_relay_frontier_v15 import compute_freshness, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the v1.5 cell-preserving freshness-return artifact.")
    parser.add_argument("data", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--b", type=int, default=2000)
    args = parser.parse_args()

    payload = compute_freshness(args.data, b=args.b)
    output = args.output or (args.reports / "freshness_return_cellpreserving_v15.json")
    write_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
