"""Configuration helpers with Streamlit secrets support."""

from __future__ import annotations

import os
from typing import Any, Optional


def _coerce_to_string(value: Any) -> Optional[str]:
    """Convert secret/environment values to trimmed strings."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _read_streamlit_secret(key: str) -> Optional[str]:
    """Return the requested key from ``st.secrets`` when available."""
    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError:
        return None

    try:
        secrets = st.secrets  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None

    try:
        raw_value = secrets[key]
    except KeyError:
        return None

    return _coerce_to_string(raw_value)


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch configuration values from Streamlit secrets or environment variables."""
    secret_value = _read_streamlit_secret(key)
    if secret_value is not None:
        return secret_value

    env_value = _coerce_to_string(os.environ.get(key))
    if env_value is not None:
        return env_value

    return default


__all__ = ["get_setting"]
