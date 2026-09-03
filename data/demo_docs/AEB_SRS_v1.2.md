---
title: AEB System Requirements Specification
document_type: requirement
vehicle_platform: DEMO-P1
feature: AEB
version: 1.2
valid_from: 2026-05-15
access_level: internal
related_signal_names: ego_speed_mps, relative_distance_m, relative_velocity_mps, object_confidence, brake_command, aeb_state
related_scenario_ids: SCN-AEB-LVSB-01
---

# AEB System Requirements Specification (SRS) v1.2

This document is SYNTHETIC demo content for the ADAS Intelligence Platform. It is not a
real OEM specification. All numeric thresholds are illustrative.

## 1. Scope

The Automatic Emergency Braking (AEB) function shall detect an imminent frontal collision
with a lead vehicle and apply autonomous braking to avoid or mitigate the collision. This
specification covers the lead-vehicle-sudden-braking use case at speeds from 10 km/h to
80 km/h on dry and wet roads.

## 2. Functional requirements

### REQ-AEB-001 Risk detection by time-to-collision

The AEB function shall compute time-to-collision (TTC) as relative_distance divided by
closing speed whenever the ego vehicle is closing on the lead vehicle. A collision risk
shall be declared when TTC is less than or equal to 2.0 s.

Rationale: 2.0 s leaves sufficient margin for a jerk-limited full braking intervention
from 50 km/h against a decelerating lead vehicle.

### REQ-AEB-002 Forward collision warning

The AEB function shall issue a forward collision warning (aeb_state = WARNING) when TTC is
less than or equal to 2.8 s and a valid target is present.

### REQ-AEB-003 Brake command latency

When a collision risk is declared (REQ-AEB-001) and a valid target is present
(REQ-AEB-010), the AEB function shall assert brake_command within 300 ms of the TTC
threshold crossing.

Verification: TC-AEB-003. Metric: braking_latency_s = brake_command_time_s -
ttc_threshold_crossing_s. Pass criterion: braking_latency_s <= 0.30 s.

### REQ-AEB-004 Deceleration build-up

After brake_command is asserted, the ego vehicle deceleration shall reach at least
6.0 m/s^2 within 500 ms, subject to the jerk limit in REQ-AEB-005.

### REQ-AEB-005 Jerk limit

The longitudinal jerk during AEB intervention shall not exceed 30 m/s^3 while the vehicle
is moving. The jerk limit does not apply at standstill.

### REQ-AEB-006 Maximum deceleration

The AEB function shall not command a deceleration exceeding 10.0 m/s^2.

### REQ-AEB-007 Brake latch

Once asserted, brake_command shall remain asserted until the ego vehicle has come to a
standstill or the collision risk has been cleared for at least 1.0 s.

### REQ-AEB-008 Collision avoidance performance

For the lead-vehicle-sudden-braking scenario SCN-AEB-LVSB-01 (ego 50 km/h, lead 50 km/h,
initial gap 30 m, lead deceleration 6 m/s^2), the AEB function shall avoid the collision
with a minimum relative distance of at least 1.0 m.

## 3. Perception interface requirements

### REQ-AEB-010 Valid target

A lead object shall be treated as a valid AEB target only while object_confidence is
greater than or equal to 0.50. Braking shall not be requested on a target whose confidence
is below this threshold.

### REQ-AEB-011 Confidence dropout handling

If object_confidence of a previously valid target drops below 0.50 for less than 200 ms,
the AEB function shall retain the target using the last valid kinematic estimate. If the
dropout lasts 200 ms or longer, the target shall be considered lost and REQ-AEB-003 timing
restarts from re-acquisition.

Note: this requirement is the known weak point for perception-induced late braking. A
confidence dropout longer than 200 ms during the risk phase directly delays the brake
command. See issue INC-2041.

### REQ-AEB-012 Confidence quality reporting

The perception stack shall report object_confidence in the range 0.0 to 1.0 at the same
sample rate as the kinematic signals (nominally 50 Hz).

## 4. Data logging requirements

### REQ-AEB-020 Logged signals

The following signals shall be logged at 50 Hz with a common monotonic timestamp:
timestamp_s, ego_speed_mps, ego_acceleration_mps2, relative_distance_m,
relative_velocity_mps, object_class, object_confidence, brake_command, aeb_state.

### REQ-AEB-021 Timestamp continuity

Logged timestamps shall be strictly increasing. Gaps longer than 2.5 sample intervals
shall be flagged in the log metadata.
