"""Deterministic 1-D lead-vehicle-sudden-braking generator for AEB diagnostics.

Physics (per sample, explicit Euler, fixed dt):

* The lead vehicle cruises at ``lead_speed_mps`` and brakes at ``lead_brake_onset_s``
  with ``lead_decel_mps2`` until it stops.
* The ego vehicle cruises at ``ego_speed_mps`` until the AEB controller issues a brake
  command, then decelerates along a jerk-limited ramp to ``ego_max_decel_mps2``.
* The AEB controller enters WARNING when true TTC <= ``warning_ttc_s`` and requests
  braking when true TTC <= ``trigger_ttc_s`` **and** perception currently reports the
  lead above ``detection_confidence_threshold``. The brake command appears
  ``system_latency_s`` after the request and latches until the ego stops.
* In the LATE_BRAKING variant, perception confidence collapses for a window after the
  lead starts braking, so the trigger condition cannot be honoured until confidence
  recovers — the brake command arrives late relative to the risk-threshold crossing.

Reported signals carry seeded Gaussian measurement noise; ground truth is computed
from the noise-free physics. The generator is a pure function of its config.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.signals import (
    COL_AEB_STATE,
    COL_BRAKE_CMD,
    COL_EGO_ACCEL,
    COL_EGO_SPEED,
    COL_OBJECT_CLASS,
    COL_OBJECT_CONF,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_TIMESTAMP,
    COL_WEATHER,
    FRAME_COLUMNS,
)
from app.synthetic.schemas import (
    AebScenarioConfig,
    AebState,
    FaultInjection,
    ScenarioGroundTruth,
    ScenarioVariant,
)

GENERATOR_VERSION = "aeb-1d-0.1.0"
OBJECT_CLASS_VEHICLE = "vehicle"


@dataclass(frozen=True)
class SyntheticScenario:
    """A generated scenario: config, noise-free truth and the (noisy, possibly faulted) frame."""

    config: AebScenarioConfig
    ground_truth: ScenarioGroundTruth
    frame: pd.DataFrame


def _true_ttc(gap_m: float, closing_speed_mps: float) -> float:
    """TTC in seconds; +inf when not closing. Matches the spec §9.4 definition."""
    if closing_speed_mps <= 0.0 or gap_m <= 0.0:
        return math.inf
    return gap_m / closing_speed_mps


def _perception_confidence(cfg: AebScenarioConfig, t: float) -> float:
    if cfg.variant is ScenarioVariant.LATE_BRAKING:
        start = cfg.lead_brake_onset_s + cfg.confidence_drop_start_after_onset_s
        if start <= t < start + cfg.confidence_drop_duration_s:
            return cfg.dropped_confidence
    return cfg.nominal_confidence


def generate_aeb_scenario(cfg: AebScenarioConfig) -> SyntheticScenario:
    """Generate one scenario. Same config (incl. seed) → identical output."""
    dt = cfg.dt_s
    n = int(round(cfg.duration_s * cfg.sample_rate_hz))
    rng = np.random.default_rng(cfg.seed)

    # State
    ego_x, ego_v, ego_a = 0.0, cfg.ego_speed_mps, 0.0
    lead_x, lead_v = cfg.initial_gap_m, cfg.lead_speed_mps
    brake_request_t: float | None = None
    brake_cmd_active = False
    collided = False

    # Truth trackers
    risk_crossing_t: float | None = None
    perception_valid_at_risk_t: float | None = None
    brake_cmd_t: float | None = None
    collision_t: float | None = None
    ego_stopped_t: float | None = None
    min_ttc = math.inf
    min_gap = math.inf
    max_decel = 0.0

    ts: list[float] = []
    speeds: list[float] = []
    accels: list[float] = []
    gaps: list[float] = []
    rel_vs: list[float] = []
    confs: list[float] = []
    brakes: list[int] = []
    states: list[int] = []

    for i in range(n):
        t = round(i * dt, 6)
        gap = lead_x - ego_x
        closing = ego_v - lead_v
        ttc = _true_ttc(gap, closing)
        conf_true = _perception_confidence(cfg, t)
        perceived = conf_true >= cfg.detection_confidence_threshold

        # --- controller decision at time t (acts on true kinematics + perception state)
        if risk_crossing_t is None and ttc <= cfg.trigger_ttc_s:
            risk_crossing_t = t
        if risk_crossing_t is not None and perception_valid_at_risk_t is None and perceived:
            perception_valid_at_risk_t = t
        if brake_request_t is None and ttc <= cfg.trigger_ttc_s and perceived:
            brake_request_t = t
        if (
            not brake_cmd_active
            and brake_request_t is not None
            and t + 1e-9 >= brake_request_t + cfg.system_latency_s
        ):
            brake_cmd_active = True
            brake_cmd_t = t

        if brake_cmd_active:
            state = AebState.BRAKING
        elif ttc <= cfg.warning_ttc_s and perceived:
            state = AebState.WARNING
        else:
            state = AebState.IDLE

        # --- truth bookkeeping for this sample
        min_gap = min(min_gap, gap)
        if ttc < min_ttc:
            min_ttc = ttc
        if gap <= 0.0 and not collided:
            collided = True
            collision_t = t
        if ego_stopped_t is None and ego_v <= 1e-6 and i > 0:
            ego_stopped_t = t

        # --- record noisy observations
        ts.append(t)
        speeds.append(max(0.0, ego_v + rng.normal(0.0, cfg.speed_noise_mps)))
        accels.append(ego_a + rng.normal(0.0, cfg.accel_noise_mps2))
        gaps.append(max(0.0, gap + rng.normal(0.0, cfg.distance_noise_m)))
        rel_vs.append(lead_v - ego_v + rng.normal(0.0, cfg.speed_noise_mps))
        confs.append(float(np.clip(conf_true + rng.normal(0.0, cfg.confidence_noise), 0.0, 1.0)))
        brakes.append(1 if brake_cmd_active else 0)
        states.append(int(state))

        # --- integrate to t + dt
        if brake_cmd_active and ego_v > 0.0:
            ego_a = max(-cfg.ego_max_decel_mps2, ego_a - cfg.ego_jerk_limit_mps3 * dt)
        elif ego_v <= 0.0:
            ego_a = 0.0
        max_decel = max(max_decel, -ego_a)
        new_ego_v = max(0.0, ego_v + ego_a * dt)
        ego_x += 0.5 * (ego_v + new_ego_v) * dt
        ego_v = new_ego_v

        lead_a = -cfg.lead_decel_mps2 if t + dt > cfg.lead_brake_onset_s and lead_v > 0 else 0.0
        new_lead_v = max(0.0, lead_v + lead_a * dt)
        lead_x += 0.5 * (lead_v + new_lead_v) * dt
        lead_v = new_lead_v

        if collided:
            # Vehicles are in contact; freeze the gap at zero rather than letting the ego
            # drive "through" the lead.
            lead_x = ego_x

    frame = pd.DataFrame(
        {
            COL_TIMESTAMP: ts,
            COL_EGO_SPEED: speeds,
            COL_EGO_ACCEL: accels,
            COL_REL_DISTANCE: gaps,
            COL_REL_VELOCITY: rel_vs,
            COL_OBJECT_CLASS: [OBJECT_CLASS_VEHICLE] * n,
            COL_OBJECT_CONF: confs,
            COL_BRAKE_CMD: brakes,
            COL_AEB_STATE: states,
            COL_WEATHER: [cfg.weather] * n,
        },
        columns=list(FRAME_COLUMNS),
    )
    frame = apply_faults(frame, cfg.faults)

    truth = ScenarioGroundTruth(
        lead_brake_onset_s=cfg.lead_brake_onset_s,
        risk_threshold_crossing_s=risk_crossing_t,
        perception_valid_at_risk_s=perception_valid_at_risk_t,
        brake_command_s=brake_cmd_t,
        braking_latency_s=(
            round(brake_cmd_t - risk_crossing_t, 6)
            if brake_cmd_t is not None and risk_crossing_t is not None
            else None
        ),
        min_ttc_s=None if math.isinf(min_ttc) else min_ttc,
        min_gap_m=max(0.0, min_gap),
        max_deceleration_mps2=max_decel,
        collision=collided,
        collision_time_s=collision_t,
        ego_stopped_s=ego_stopped_t,
    )
    return SyntheticScenario(config=cfg, ground_truth=truth, frame=frame)


def apply_faults(frame: pd.DataFrame, faults: FaultInjection) -> pd.DataFrame:
    """Apply data-quality faults to a clean frame. Returns a new frame; input untouched."""
    out = frame.copy()
    t = out[COL_TIMESTAMP]
    if faults.timestamp_gap is not None:
        g = faults.timestamp_gap
        keep = ~((t >= g.start_s) & (t < g.start_s + g.duration_s))
        out = out.loc[keep].reset_index(drop=True)
        t = out[COL_TIMESTAMP]
    if faults.nan_burst is not None:
        b = faults.nan_burst
        if b.column in out.columns:
            mask = (t >= b.start_s) & (t < b.start_s + b.duration_s)
            if pd.api.types.is_numeric_dtype(out[b.column]):
                # int columns cannot hold NaN; widen once instead of relying on upcasting.
                out[b.column] = out[b.column].astype("float64")
            out.loc[mask, b.column] = np.nan
    if faults.drop_columns:
        out = out.drop(columns=[c for c in faults.drop_columns if c in out.columns])
    return out
