---
title: AEB Issue History
document_type: issue
vehicle_platform: DEMO-P1
feature: AEB
version: 2026-08
valid_from: 2026-08-20
access_level: internal
related_signal_names: object_confidence, brake_command
---

# AEB Issue History

SYNTHETIC demo content. Past incidents relevant to AEB late braking.

## INC-2041 Late AEB intervention after fusion confidence collapse

Symptom: brake_command asserted 0.7 to 0.9 s after the TTC threshold crossing in three
lead-vehicle-braking runs. Root cause: radar-camera fusion re-associated the lead track
when the lead vehicle's brake lights changed the camera exposure, dropping
object_confidence from about 0.9 to below 0.3 for roughly 0.5 to 1.2 s. With
object_confidence below the 0.50 validity threshold, REQ-AEB-010 prevented the brake
request until the track was re-acquired.

Evidence pattern: object_confidence step-down during the risk phase; brake_command rising
edge coincides with confidence recovery; ego kinematics nominal before the drop.

Fix: fusion track-hold tuning (release 2026.07). Status: closed, regression test TC-AEB-011.

## INC-1877 Unnecessary AEB braking on overhead sign

Symptom: brake_command asserted with no lead vehicle; relative_distance_m reported 25 m to
a stationary object. Root cause: radar ghost target from an overhead gantry, camera
confidence 0.55 (barely above threshold). Fix: raised camera weight in fusion for
stationary elevated returns. Status: closed.

## INC-1990 Timestamp gap in logger

Symptom: a 0.6 s gap in timestamp_s during a braking run made braking latency
unmeasurable. Root cause: logger buffer overflow at 50 Hz with video enabled. Fix: logger
firmware 4.2. Status: closed. Note: the data-quality gate for timestamp continuity was
added after this incident.
