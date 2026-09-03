---
title: AEB Telemetry Signal Definitions
document_type: dbc
vehicle_platform: DEMO-P1
feature: AEB
version: 3.0
valid_from: 2026-04-01
access_level: internal
related_signal_names: timestamp_s, ego_speed_mps, ego_acceleration_mps2, relative_distance_m, relative_velocity_mps, object_class, object_confidence, brake_command, aeb_state, weather
---

# AEB Telemetry Signal Definitions v3.0

SYNTHETIC demo content: a DBC-like description of the logged signals.

## timestamp_s

Monotonic log time in seconds since log start. Source: logger clock domain. Resolution
0.001 s. Nominal sample interval 0.02 s (50 Hz).

## ego_speed_mps

Ego vehicle longitudinal speed in metres per second. Source: wheel speed fusion. Some
export tools write this column as ego_speed_kmh in km/h; ingestion must convert once.

## ego_acceleration_mps2

Ego vehicle longitudinal acceleration in m/s^2, negative when braking. Source: IMU,
low-pass filtered at 10 Hz.

## relative_distance_m

Longitudinal gap from ego front bumper to the rear of the selected lead object in metres.
Source: radar-camera fusion. Zero indicates contact.

## relative_velocity_mps

Lead object speed minus ego speed in m/s. Negative values mean the ego is closing on the
lead. Closing speed = -relative_velocity_mps. Exported as relative_velocity_kmh by some
tools.

## object_class

Classification of the selected lead object: vehicle, pedestrian, cyclist, unknown.

## object_confidence

Fusion confidence that the selected lead object exists and its kinematics are valid,
range 0.0 to 1.0. Values below 0.50 mark the target as not valid for AEB (REQ-AEB-010).
Confidence collapses are typically caused by camera exposure changes, radar multipath, or
fusion track re-association.

## brake_command

AEB autonomous brake request: 0 = not requested, 1 = requested. Rising edge marks the
brake command time used for braking latency.

## aeb_state

AEB state machine: 0 = IDLE, 1 = WARNING (TTC <= 2.8 s), 2 = BRAKING.

## weather

Categorical weather tag for the run: clear, rain, fog, snow.
