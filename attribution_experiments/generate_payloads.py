"""Generate the frozen payload files shared by every ablation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from attribution_experiments.common.analysis_payloads import (
    DEFAULT_PAYLOAD_ROOT,
    DEFAULT_RANDOM_SEED,
    generate_payloads,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    manifest = generate_payloads(args.output_root.resolve(), args.seed)
    print(json.dumps({"tasks": len(manifest["tasks"]), "groups": manifest["groups"]}, indent=2))


if __name__ == "__main__":
    main()
