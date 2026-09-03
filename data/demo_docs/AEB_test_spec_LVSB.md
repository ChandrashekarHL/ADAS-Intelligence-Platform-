---
title: AEB Test Specification - Lead Vehicle Sudden Braking
document_type: test_spec
vehicle_platform: DEMO-P1
feature: AEB
version: 1.1
valid_from: 2026-06-01
access_level: internal
related_signal_names: brake_command, relative_distance_m, object_confidence
related_scenario_ids: SCN-AEB-LVSB-01
---

# AEB Test Specification: Lead Vehicle Sudden Braking (LVSB)

SYNTHETIC demo content.

## Scenario SCN-AEB-LVSB-01

Ego and lead vehicle travel at 50 km/h with a 30 m gap on a straight dry road. At T0 the
lead vehicle brakes at 6 m/s^2 to a standstill. Weather: clear. Duration 10 s, sampling
50 Hz.

### TC-AEB-001 Risk declaration timing

Verify that TTC computed from relative_distance_m and relative_velocity_mps falls to 2.0 s
approximately 1.76 s after lead brake onset. Pass: the TTC threshold crossing is detected
within +/- 2 samples of the analytical value.

### TC-AEB-003 Brake command latency

Verify REQ-AEB-003. Measure braking_latency_s as the time from the TTC threshold crossing
to the brake_command rising edge. Pass: braking_latency_s <= 0.30 s. Fail: any larger
value. Record object_confidence over the risk phase to distinguish controller latency
from perception-induced delay.

### TC-AEB-008 Collision avoidance

Verify REQ-AEB-008. Pass: relative_distance_m never falls below 1.0 m and no collision
event is logged. Fail: collision or minimum gap below 1.0 m.

### TC-AEB-011 Confidence dropout robustness

Inject an object_confidence dropout of 150 ms and of 400 ms during the risk phase. Pass:
the 150 ms dropout does not delay brake_command by more than one sample; the 400 ms
dropout is reported as target lost and brake_command follows within 300 ms of
re-acquisition.

## Data quality preconditions

A test run is valid only if the log passes the data-quality gates: all REQ-AEB-020 signals
present, strictly increasing timestamps, no gap longer than 2.5 sample intervals, and no
more than 5 percent missing samples in any critical signal.
