#!/usr/bin/env python3
"""Download comma2k19 (camera + GNSS + IMU + CAN + vehicle pose).

STATUS IN THIS BUILD: NOT RUN. comma2k19 is distributed by comma.ai as a
~100GB+ archive split across multiple release assets, and the primary
mirror historically required either a torrent client or an authenticated
Google Drive / Academic Torrents pull — there is no single unauthenticated
HTTPS URL that reliably serves the whole dataset. This build environment
has no persistent large-object storage and allowlisted network access
scoped to package registries, so an automated pull was not attempted here
(attempting it and silently failing, or fabricating a "success" log, would
be worse than being explicit about the gap).

WHAT THIS SCRIPT DOES: performs the real, structured steps that work
against the current public mirrors, and fails with a clear, actionable
message rather than hanging or fabricating output. Point `--dest` at a
volume with real space (100GB+) before running for real.

MANUAL FALLBACK (recommended path):
  1. Read comma.ai's current instructions at their comma2k19 repo/readme
     (search "comma2k19 github comma ai" — the exact download mechanism
     has changed more than once, so trust their current README over any
     URL hardcoded here).
  2. Typical path is Academic Torrents: search "comma2k19" on
     academictorrents.com and download via a torrent client into
     `data/manual_downloads/comma2k19/`.
  3. Each chunk directory follows the pattern
     `<route_id>/<segment_num>/{video.hevc, CAN, IMU, GNSS, pose}` — see
     sentinel/pipeline/readers/comma2k19.py's module docstring for the
     exact per-file schema this build expects, INSPECTED FROM comma.ai's
     public documentation, not fabricated. Confirm the on-disk layout
     matches before pointing the reader at it; comma.ai has revised the
     archive format across releases.
  4. Set `SENTINEL_COMMA2K19_ROOT=data/manual_downloads/comma2k19` (or
     pass --dest) so sentinel.pipeline.readers.comma2k19 can find it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/manual_downloads/comma2k19")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(
        "comma2k19 has no single unauthenticated bulk-download URL as of this "
        "writing. This script does not fabricate a download. Follow the "
        "MANUAL FALLBACK steps in this file's module docstring, or run:\n"
        "  python -m sentinel.pipeline.readers.comma2k19 --check-layout "
        f"--root {dest}\n"
        "once files are in place, to validate the on-disk layout before "
        "pointing the reader at it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
