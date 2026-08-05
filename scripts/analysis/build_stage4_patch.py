from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphptc.patch_controller import write_stage4_patch_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a validated offline Stage 4 local patch version."
    )
    parser.add_argument("graph_path", type=Path)
    parser.add_argument("proposal_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = write_stage4_patch_report(
        args.graph_path,
        args.proposal_path,
        args.output_path,
    )
    print(
        json.dumps(
            {
                "episode_id": report["episode_id"],
                "prompt_variant": report["prompt_variant"],
                "original_version_id": report["application"]["original"]["id"],
                "patched_version_id": report["application"]["patched"]["id"],
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
