from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_bundle, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the authorized Pyramid futures evidence bundle")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization-json", type=Path, required=True,
                        help="explicit private-research authorization evidence JSON")
    parser.add_argument("--expected-raw-inventory-json", type=Path, required=True)
    parser.add_argument("--expected-corrected-raw-aggregate-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    authorization = json.loads(args.authorization_json.read_text(encoding="utf-8"))
    result = (preflight(args.source_root, args.out, authorization) if args.preflight_only else
              build_bundle(args.source_root, args.out, authorization, args.batch_size,
                           json.loads(args.expected_raw_inventory_json.read_text(encoding="utf-8")),
                           args.expected_corrected_raw_aggregate_sha256))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
