"""Project-level exception types."""


class VitLabError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(VitLabError):
    """A configuration value is invalid or inconsistent with another value."""


class CheckpointError(VitLabError):
    """A checkpoint file is missing or does not hold the expected content."""


class ResultsError(VitLabError):
    """A results file is missing or cannot be read."""
