"""Sentinel: runtime assurance layer for AI-native autonomous systems.

Sentinel observes an autonomy stack from the outside and continuously
decides how much authority that stack is currently allowed to exercise.
It never drives; it never plans; it only monitors, decides, constrains,
records, and lets you replay.

Import boundary rule: `sentinel.trust`, `sentinel.policy`, and
`sentinel.evidence` (the decision path) must be importable without
pulling in torch. Anything that needs torch lives behind
`sentinel.monitors.m1_perception` and is imported lazily.
"""

__version__ = "0.1.0"
