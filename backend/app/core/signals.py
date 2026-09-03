"""Canonical telemetry signal vocabulary shared by every stage.

Column names encode their SI unit so a frame is self-describing. Ingestion maps
external headers (e.g. ``ego_speed_kmh``) onto these names, converting exactly once.
"""

COL_TIMESTAMP = "timestamp_s"
COL_EGO_SPEED = "ego_speed_mps"
COL_EGO_ACCEL = "ego_acceleration_mps2"
COL_REL_DISTANCE = "relative_distance_m"
COL_REL_VELOCITY = "relative_velocity_mps"
COL_OBJECT_CLASS = "object_class"
COL_OBJECT_CONF = "object_confidence"
COL_BRAKE_CMD = "brake_command"
COL_AEB_STATE = "aeb_state"
COL_WEATHER = "weather"

FRAME_COLUMNS: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_EGO_SPEED,
    COL_EGO_ACCEL,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_OBJECT_CLASS,
    COL_OBJECT_CONF,
    COL_BRAKE_CMD,
    COL_AEB_STATE,
    COL_WEATHER,
)

# Unit each canonical column is stored in. Non-physical columns get a descriptive tag.
CANONICAL_UNITS: dict[str, str] = {
    COL_TIMESTAMP: "s",
    COL_EGO_SPEED: "m/s",
    COL_EGO_ACCEL: "m/s^2",
    COL_REL_DISTANCE: "m",
    COL_REL_VELOCITY: "m/s",
    COL_OBJECT_CLASS: "category",
    COL_OBJECT_CONF: "ratio",
    COL_BRAKE_CMD: "bool",
    COL_AEB_STATE: "enum",
    COL_WEATHER: "category",
}

NUMERIC_COLUMNS: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_EGO_SPEED,
    COL_EGO_ACCEL,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_OBJECT_CONF,
    COL_BRAKE_CMD,
    COL_AEB_STATE,
)

# Physical-quantity base names (without unit suffix) → canonical column.
SIGNAL_BASE_NAMES: dict[str, str] = {
    "timestamp": COL_TIMESTAMP,
    "time": COL_TIMESTAMP,
    "ego_speed": COL_EGO_SPEED,
    "ego_acceleration": COL_EGO_ACCEL,
    "relative_distance": COL_REL_DISTANCE,
    "relative_velocity": COL_REL_VELOCITY,
    "object_class": COL_OBJECT_CLASS,
    "object_confidence": COL_OBJECT_CONF,
    "brake_command": COL_BRAKE_CMD,
    "aeb_state": COL_AEB_STATE,
    "weather": COL_WEATHER,
}

# AEB late-braking diagnostics cannot run without these (spec §26 AEB inputs).
CRITICAL_AEB_SIGNALS: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_EGO_SPEED,
    COL_REL_DISTANCE,
    COL_REL_VELOCITY,
    COL_OBJECT_CONF,
    COL_BRAKE_CMD,
)

# Useful but not blocking; absence downgrades confidence rather than stopping analysis.
OPTIONAL_AEB_SIGNALS: tuple[str, ...] = (
    COL_EGO_ACCEL,
    COL_OBJECT_CLASS,
    COL_AEB_STATE,
    COL_WEATHER,
)
