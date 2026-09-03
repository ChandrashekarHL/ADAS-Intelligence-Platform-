---
title: Supplier Fusion Calibration Parameters (restricted)
document_type: manual
vehicle_platform: DEMO-P1
feature: AEB
version: 0.9
valid_from: 2025-11-10
access_level: restricted
related_signal_names: object_confidence
---

# Supplier Fusion Calibration Parameters

SYNTHETIC demo content marked restricted. This document exists to prove that the
retrieval layer never returns restricted chunks to a caller without restricted access.

## Track hold parameters

Track hold time after confidence drop: 180 ms. Confidence hysteresis: valid above 0.50,
invalid below 0.45. Re-association gate: 1.5 m longitudinal, 0.8 m lateral.
