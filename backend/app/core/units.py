"""Canonical unit conventions — SI everywhere inside the codebase.

- distance: metres (m)
- speed: metres/second (m/s)
- acceleration: metres/second^2 (m/s^2)
- jerk: metres/second^3 (m/s^3)
- time / timestamps: seconds (s), monotonic within a log file
- TTC: seconds (s)

Ingestion converts external units (e.g. km/h in demo CSVs) at the boundary and
records the conversion; nothing downstream re-converts.
"""

KMH_TO_MPS: float = 1.0 / 3.6
MPS_TO_KMH: float = 3.6
MS_TO_S: float = 1e-3
