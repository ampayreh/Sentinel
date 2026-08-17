#!/usr/bin/env python3
"""Download the NVIDIA PhysicalAI Autonomous Vehicles dataset (camera +
LiDAR + radar + ego-motion).

STATUS IN THIS BUILD: NOT RUN. NVIDIA PhysicalAI datasets are distributed
through NGC (NVIDIA GPU Cloud) / Hugging Face and require an NVIDIA
account plus, for several PhysicalAI releases, acceptance of a dataset
EULA before any file is servable — there is no anonymous bulk-download
endpoint. This build environment cannot complete an interactive
account/EULA flow, so this script does not attempt a network pull.

MANUAL FALLBACK:
  1. Create/sign in to an NVIDIA NGC account (or Hugging Face account, if
     the specific PhysicalAI AV release you need is mirrored there).
  2. Search the current NGC catalog or huggingface.co/nvidia for
     "PhysicalAI Autonomous Vehicles" — get the exact current dataset
     slug; NVIDIA has published multiple PhysicalAI dataset drops and the
     slug/version you want depends on which sensor suite (camera+LiDAR
     vs. +radar) you need.
  3. Use `ngc registry resource download-version <org>/<dataset>` (NGC
     CLI) or `huggingface-cli download <repo>` into
     `data/manual_downloads/physicalai/`.
  4. Inspect the actual downloaded file layout before writing/trusting a
     reader against it — sentinel/pipeline/readers/physicalai.py's
     docstring documents the expected schema as INFERRED FROM NVIDIA'S
     PUBLISHED DOCS, not verified against a real downloaded file in this
     build; treat column/field names there as provisional until you
     confirm them against what you actually download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/manual_downloads/physicalai")
    args = parser.parse_args()
    Path(args.dest).mkdir(parents=True, exist_ok=True)
    print(
        "NVIDIA PhysicalAI AV dataset requires an authenticated NGC/HF pull "
        "and EULA acceptance this script cannot complete non-interactively. "
        "See this file's MANUAL FALLBACK docstring.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
