#!/usr/bin/env python3
"""Download the NHTSA Standing General Order (SGO) ADS/ADAS incident
reporting dataset, and honest findings on Seattle/Pacific-Northwest-
specific AV data.

RESEARCH FINDING (this is the deliverable the spec explicitly asked for —
"investigate and report what actually exists before assuming"):

  - Washington State DOT publishes AV TESTING SELF-CERTIFICATIONS
    (companies declaring intent/readiness to test), not disengagement or
    incident data. These are NOT equivalent to CA's disengagement
    taxonomy and were not used as a taxonomy source in this build.
  - There is no clean, standalone "Seattle AV disengagement/incident"
    dataset comparable to CA's DMV reports. The best fusable NATIONAL
    source that includes real incidents and is filterable by city/state
    is NHTSA's Standing General Order (SGO) ADS/ADAS incident reporting
    dataset, which NHTSA publishes as downloadable CSVs and which
    includes fields for city, state, weather, and lighting condition.
  - Schema reconciliation with the CA taxonomy is NOT automatic: SGO
    records are incident reports (crashes/near-crashes with a structured
    narrative + coded fields), while CA's disengagement reports are
    proactive disengagement events with a free-text cause description.
    They describe different populations (SGO: manufacturer-reported
    incidents, often crash-triggered; CA: any disengagement, including
    zero-consequence precautionary ones) — a naive union would conflate
    two different base rates. Reconciliation approach implemented in
    sentinel/pipeline/readers/dmv_taxonomy.py's classify_description():
    the same keyword-based FailureClass taxonomy is applied to SGO's
    narrative field, but SGO/CA records should be reported as SEPARATE
    counts by failure class, not summed, unless you've verified the two
    populations are comparable for your use case.

STATUS IN THIS BUILD: NOT RUN, for the same reason as the other download
scripts — this sandbox's outbound network access is not allowlisted for
arbitrary sites (only package registries), and NHTSA's SGO portal was not
reachable to verify a current bulk-download URL. See MANUAL FALLBACK.

MANUAL FALLBACK:
  1. Visit NHTSA's Standing General Order website (search "NHTSA Standing
     General Order ADS ADAS incident reporting" for the current URL — it
     has moved between nhtsa.gov paths before).
  2. Download the "SGO 2021-Present" (or current) CSV/XLSX bundle into
     `data/manual_downloads/nhtsa_sgo/`.
  3. Filter to City/State fields matching Seattle/WA if you specifically
     want Pacific Northwest coverage; note the resulting sample size may
     be small — report it honestly rather than treating a handful of
     records as a representative distribution.
  4. Run the same `classify_description`-based clustering used for CA
     DMV data, but keep the counts in a separate column/table (see
     sentinel/pipeline/readers/dmv_taxonomy.py's ClusterResult) rather
     than merging into one taxonomy table.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    dest = Path("data/manual_downloads/nhtsa_sgo")
    dest.mkdir(parents=True, exist_ok=True)
    print(__doc__, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
