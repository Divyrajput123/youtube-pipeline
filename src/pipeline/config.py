"""Pipeline-wide configuration and mode detection.

This module provides utilities for detecting the pipeline's operational mode
(development vs production) and controlling fallback behavior.
"""

import os
from enum import Enum


class PipelineMode(Enum):
    """Pipeline operational mode."""
    
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_pipeline_mode() -> PipelineMode:
    """Detect the current pipeline mode from the PIPELINE_MODE environment variable.
    
    Returns:
        PipelineMode.PRODUCTION if PIPELINE_MODE is set to "production" (case-insensitive),
        otherwise PipelineMode.DEVELOPMENT (default).
        
    Examples:
        >>> os.environ['PIPELINE_MODE'] = 'production'
        >>> get_pipeline_mode()
        <PipelineMode.PRODUCTION: 'production'>
        
        >>> os.environ['PIPELINE_MODE'] = 'development'
        >>> get_pipeline_mode()
        <PipelineMode.DEVELOPMENT: 'development'>
        
        >>> del os.environ['PIPELINE_MODE']
        >>> get_pipeline_mode()
        <PipelineMode.DEVELOPMENT: 'development'>
    """
    mode_str = os.environ.get("PIPELINE_MODE", "development").lower().strip()
    
    if mode_str == "production":
        return PipelineMode.PRODUCTION
    
    return PipelineMode.DEVELOPMENT


def is_production_mode() -> bool:
    """Check if the pipeline is running in production mode.
    
    Returns:
        True if PIPELINE_MODE=production, False otherwise.
    """
    return get_pipeline_mode() == PipelineMode.PRODUCTION


def is_development_mode() -> bool:
    """Check if the pipeline is running in development mode.
    
    Returns:
        True if PIPELINE_MODE is not set to production, False otherwise.
    """
    return get_pipeline_mode() == PipelineMode.DEVELOPMENT


def require_production_config(
    service_name: str,
    config_value: str | None,
    error_message: str | None = None,
) -> None:
    """Raise an error if a required configuration is missing in production mode.
    
    In development mode, this function does nothing (allows fallbacks).
    In production mode, raises ValueError if config_value is None or empty.
    
    Args:
        service_name: Name of the service (for error message).
        config_value: The configuration value to check.
        error_message: Optional custom error message.
        
    Raises:
        ValueError: If in production mode and config_value is missing.
        
    Examples:
        >>> os.environ['PIPELINE_MODE'] = 'production'
        >>> require_production_config('ElevenLabs', None)
        Traceback (most recent call last):
            ...
        ValueError: Production mode requires ElevenLabs configuration
        
        >>> os.environ['PIPELINE_MODE'] = 'development'
        >>> require_production_config('ElevenLabs', None)  # No error in dev mode
    """
    if not is_production_mode():
        return  # Allow missing config in development mode
    
    if not config_value or (isinstance(config_value, str) and not config_value.strip()):
        if error_message:
            raise ValueError(error_message)
        else:
            raise ValueError(
                f"Production mode requires {service_name} configuration. "
                f"Please set the required environment variables or switch to "
                f"development mode (PIPELINE_MODE=development)."
            )


__all__ = [
    "PipelineMode",
    "get_pipeline_mode",
    "is_production_mode",
    "is_development_mode",
    "require_production_config",
]
