#!/usr/bin/env python3
"""Download California DMV Autonomous Vehicle Disengagement Reports.

STATUS IN THIS BUILD: NOT RUN. The CA DMV disengagement report pages
(dmv.ca.gov) returned HTTP 403 to this build environment's outbound
network access (general web fetch is not allowlisted here — only package
registries are). This is a real, verified result from this session, not
an assumption: `curl -I https://www.dmv.ca.gov/...` returned 403 Forbidden
when actually attempted.

MANUAL FALLBACK:
  1. Visit dmv.ca.gov and search "autonomous vehicle disengagement
     reports" — CA DMV publishes one ZIP/CSV bundle per reporting year
     (Dec 1 - Nov 30), one file per manufacturer with an active testing
     permit.
  2. Download the CSVs for the years you want into
     `data/manual_downloads/dmv_disengagement/<year>/`.
  3. Run `python scripts/download_dmv_disengagement.py --parse-only
     --root data/manual_downloads/dmv_disengagement` to run the taxonomy
     clustering in sentinel/pipeline/readers/dmv_taxonomy.py against the
     real files.

COLUMN NAME HONESTY NOTE: sentinel/pipeline/readers/dmv_taxonomy.py's
parser is written against CA DMV's disengagement report schema as
documented publicly (columns commonly named similarly to: Manufacturer,
Permit Number, DATE, VIN NUMBER, VEHICLE IS CAPABLE OF OPERATING IN
AUTONOMOUS MODE, DRIVER PRESENT, DISENGAGEMENT INITIATED BY, LOCATION,
DESCRIPTION OF FACTS CAUSING DISENGAGEMENT) — but this was NOT verified
against a freshly downloaded file in this session, since the download
itself failed (see above). The parser is written defensively (fuzzy
column-name matching, see `_COLUMN_ALIASES`) specifically because of this
uncertainty, and will raise a clear error naming the columns it found vs.
expected rather than silently guessing, if you point it at a real file
whose columns don't match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/manual_downloads/dmv_disengagement")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="skip the (currently non-functional) download step and just run the taxonomy parser",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if not args.parse_only:
        print(
            "dmv.ca.gov returned 403 to this environment's outbound network "
            "access when tested. See this file's MANUAL FALLBACK docstring.",
            file=sys.stderr,
        )
        return 1

    from sentinel.pipeline.readers.dmv_taxonomy import cluster_directory

    csv_files = list(root.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {root}. Download real files first (see docstring).", file=sys.stderr)
        return 1

    result = cluster_directory(csv_files)
    print(result.summary_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
