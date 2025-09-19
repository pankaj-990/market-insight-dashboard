"""Compatibility shim for the legacy Streamlit entry point."""

from __future__ import annotations

import warnings

from main import main as _main

warnings.warn(
    "'streamlit_app.py' is deprecated; use 'streamlit run main.py' instead.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    _main()
