"""Domain errors.

Data-quality failures block or downgrade downstream analysis; they are raised,
never silently swallowed.
"""


class AipError(Exception):
    """Base class for all AIP domain errors."""


class DataQualityError(AipError):
    """Raised when a data-quality gate fails hard (e.g. missing critical signal)."""


class EvidenceResolutionError(AipError):
    """Raised when a cited evidence ID does not resolve to a stored artifact."""


class ProviderError(AipError):
    """Raised when an LLM/embedding provider call fails after retries."""


class IngestionError(AipError):
    """Raised when a telemetry file cannot be read or its columns cannot be resolved."""
