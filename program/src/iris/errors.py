"""Custom exceptions for the iris project."""


class IrisPipelineError(Exception):
    """Base exception for known pipeline failures."""


class ConfigError(IrisPipelineError):
    """Raised when configuration is invalid or missing."""


class InputDataError(IrisPipelineError):
    """Raised when expected input data is not available."""

