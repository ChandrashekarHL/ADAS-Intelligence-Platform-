"""The locked AEB scenario suite.

Changing anything here changes ``suite_sha256()``; results and baselines carry that hash and
refuse to be compared across different suites. Pin new expectations from the generator's
ground truth (``ScenarioMetadata.ground_truth``), not from the metrics code.
"""

import hashlib
import json

from app.agents.schemas import FailureClass
from app.eval.schemas import EvalCase, ExpectedOutcome
from app.metrics.aeb import (
    M_BRAKING_LATENCY,
    M_COLLISION,
    M_CONF_DROPOUT,
    M_MIN_CONF_IN_RISK,
    M_MIN_TTC,
)
from app.metrics.schemas import EventType
from app.quality.report import QualityVerdict
from app.synthetic.schemas import (
    AebScenarioConfig,
    FaultInjection,
    NanBurst,
    ScenarioVariant,
    TimestampGap,
)
from app.verification.schemas import ReportConfidence

SUITE_NAME = "aeb-locked-v1"

# The late-braking generator delays detection by collapsing object confidence during the
# risk phase. That is a perception-layer failure; "fusion_error" is accepted because the
# confidence signal is the fused track's and the matching past incident (INC-2041) is
# filed as a fusion confidence collapse. Blaming the planner, controller or calibration
# is wrong: the planner braked within 0.16 s of the target becoming valid again.
_CONFIDENCE_COLLAPSE = frozenset({FailureClass.PERCEPTION_ERROR, FailureClass.FUSION_ERROR})

_COLLISION_EVENTS = (
    EventType.TTC_THRESHOLD_CROSSING,
    EventType.AEB_BRAKE_COMMAND,
    EventType.COLLISION,
)
_NO_COLLISION_EVENTS = (EventType.TTC_THRESHOLD_CROSSING, EventType.AEB_BRAKE_COMMAND)

_LATE_WITH_COLLISION = {
    M_BRAKING_LATENCY: False,
    M_COLLISION: False,
    M_MIN_TTC: False,
    M_CONF_DROPOUT: False,
    M_MIN_CONF_IN_RISK: False,
}
_NOMINAL_VERDICTS = {
    M_BRAKING_LATENCY: True,
    M_COLLISION: True,
    M_MIN_TTC: True,
    M_CONF_DROPOUT: True,
    M_MIN_CONF_IN_RISK: True,
}


def _late(seed: int, rationale: str) -> EvalCase:
    return EvalCase(
        case_id=f"late_braking_seed{seed}",
        description=f"Demo late-braking scenario, noise seed {seed}: confidence collapse "
        "delays the brake command 0.8 s after the TTC threshold; collision at ~6.5 s.",
        config=AebScenarioConfig(variant=ScenarioVariant.LATE_BRAKING, seed=seed),
        expected_outcome=ExpectedOutcome.FAULT,
        expected_quality=QualityVerdict.PASS,
        expected_events=_COLLISION_EVENTS,
        expected_metric_verdicts=_LATE_WITH_COLLISION,
        acceptable_failure_classes=_CONFIDENCE_COLLAPSE,
        rationale=rationale,
    )


def _nominal(case_id: str, config: AebScenarioConfig, description: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        description=description,
        config=config,
        expected_outcome=ExpectedOutcome.NOMINAL,
        expected_quality=QualityVerdict.PASS,
        expected_events=_NO_COLLISION_EVENTS,
        forbidden_events=(EventType.COLLISION,),
        expected_metric_verdicts=_NOMINAL_VERDICTS,
        rationale="Ground truth: brake 0.16 s after the TTC threshold (limit 0.30 s), no "
        "collision, confidence never below 0.5 in the risk phase. A correct agent reports "
        "no violated requirement; any fault hypothesis above low confidence is a false alarm.",
    )


LOCKED_SUITE: tuple[EvalCase, ...] = (
    _late(
        42,
        "Ground truth (generator): brake_command 5.56 s, latency 0.80 s > 0.30 s, "
        "perception valid again at 5.40 s, collision 6.54 s.",
    ),
    _late(7, "Same physics as seed 42 with different measurement noise; latency 0.82 s."),
    _late(2026, "Same physics as seed 42 with different measurement noise; latency 0.80 s."),
    EvalCase(
        case_id="late_braking_near_miss",
        description="Shorter confidence collapse (0.9 s): brake command 0.52 s late, "
        "min TTC ~0.55 s, no collision.",
        config=AebScenarioConfig(
            variant=ScenarioVariant.LATE_BRAKING, seed=11, confidence_drop_duration_s=0.9
        ),
        expected_outcome=ExpectedOutcome.FAULT,
        expected_quality=QualityVerdict.PASS,
        expected_events=_NO_COLLISION_EVENTS,
        forbidden_events=(EventType.COLLISION,),
        expected_metric_verdicts={
            M_BRAKING_LATENCY: False,
            M_COLLISION: True,
            M_CONF_DROPOUT: False,
            M_MIN_CONF_IN_RISK: False,
        },
        acceptable_failure_classes=_CONFIDENCE_COLLAPSE,
        rationale="Ground truth: latency 0.52 s > 0.30 s with collision avoided. The agent "
        "must still find the latency violation and its perception cause; a near miss is "
        "not a pass. min_ttc_s sits at ~0.53 s against a 0.5 s limit and is not locked.",
    ),
    _nominal(
        "nominal_seed42",
        AebScenarioConfig(variant=ScenarioVariant.NOMINAL, seed=42),
        "Demo nominal scenario: same approach, perception stays valid, brake at 4.92 s.",
    ),
    _nominal(
        "nominal_seed7",
        AebScenarioConfig(variant=ScenarioVariant.NOMINAL, seed=7),
        "Nominal scenario with different measurement noise; latency 0.18 s.",
    ),
    _nominal(
        "nominal_60kmh",
        AebScenarioConfig(
            variant=ScenarioVariant.NOMINAL,
            seed=3,
            ego_speed_mps=60 / 3.6,
            lead_speed_mps=60 / 3.6,
            initial_gap_m=35.0,
        ),
        "Nominal scenario at 60 km/h with a 35 m gap: brake at 5.12 s, min TTC ~1.0 s.",
    ),
    EvalCase(
        case_id="blocked_missing_ego_speed",
        description="Late-braking log with the ego_speed_mps column removed.",
        config=AebScenarioConfig(
            variant=ScenarioVariant.LATE_BRAKING,
            seed=5,
            faults=FaultInjection(drop_columns=("ego_speed_mps",)),
        ),
        expected_outcome=ExpectedOutcome.BLOCKED,
        expected_quality=QualityVerdict.BLOCKED,
        rationale="ego_speed_mps is a critical AEB signal; the required_signals gate must "
        "block analysis before any metric or agent runs (no pretend diagnosis).",
    ),
    EvalCase(
        case_id="degraded_confidence_nan_burst",
        description="Late-braking log with a 0.4 s NaN burst in object_confidence at 1.0 s, "
        "outside the risk phase.",
        config=AebScenarioConfig(
            variant=ScenarioVariant.LATE_BRAKING,
            seed=5,
            faults=FaultInjection(
                nan_burst=NanBurst(column="object_confidence", start_s=1.0, duration_s=0.4)
            ),
        ),
        expected_outcome=ExpectedOutcome.FAULT,
        expected_quality=QualityVerdict.DEGRADED,
        expected_events=_COLLISION_EVENTS,
        expected_metric_verdicts=_LATE_WITH_COLLISION,
        acceptable_failure_classes=_CONFIDENCE_COLLAPSE,
        max_report_confidence=ReportConfidence.MEDIUM,
        rationale="The fault is real (latency 0.82 s, collision) but the quality verdict is "
        "DEGRADED, so the verifier must cap report confidence at medium.",
    ),
    EvalCase(
        case_id="degraded_timestamp_gap_nominal",
        description="Nominal log with a 0.3 s timestamp gap at 1.0 s.",
        config=AebScenarioConfig(
            variant=ScenarioVariant.NOMINAL,
            seed=5,
            faults=FaultInjection(timestamp_gap=TimestampGap(start_s=1.0, duration_s=0.3)),
        ),
        expected_outcome=ExpectedOutcome.NOMINAL,
        expected_quality=QualityVerdict.DEGRADED,
        expected_events=_NO_COLLISION_EVENTS,
        forbidden_events=(EventType.COLLISION,),
        expected_metric_verdicts=_NOMINAL_VERDICTS,
        max_report_confidence=ReportConfidence.MEDIUM,
        rationale="Nothing is wrong with the vehicle; the log is merely degraded. No fault "
        "hypothesis above low confidence, and the degraded cap applies.",
    ),
)


def _canonical(value: object) -> object:
    """Make the dump independent of set iteration order (PYTHONHASHSEED): any list made
    only of strings is sorted. Event lists lose their order in the hash, which is fine
    for an identity."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_canonical(v) for v in value]
        strings = [v for v in items if isinstance(v, str)]
        return sorted(strings) if len(strings) == len(items) else items
    return value


def suite_sha256(cases: tuple[EvalCase, ...] = LOCKED_SUITE) -> str:
    """Content hash of the case definitions; results are comparable only when it matches.
    Stable across processes and Python hash seeds."""
    payload = json.dumps(
        [_canonical(c.model_dump(mode="json")) for c in cases],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_cases(case_ids: list[str] | None) -> tuple[EvalCase, ...]:
    """Subset of the suite by id, in suite order. Unknown ids raise ``KeyError``."""
    if not case_ids:
        return LOCKED_SUITE
    known = {c.case_id for c in LOCKED_SUITE}
    unknown = sorted(set(case_ids) - known)
    if unknown:
        raise KeyError(f"unknown eval case(s): {unknown}; known: {sorted(known)}")
    wanted = set(case_ids)
    return tuple(c for c in LOCKED_SUITE if c.case_id in wanted)
