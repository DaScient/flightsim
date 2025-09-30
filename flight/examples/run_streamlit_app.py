"""
Launch the interactive Streamlit application.

This convenience script locates the Streamlit app bundled with the
`flight` package and invokes Streamlit to run it.  It requires the
``streamlit`` package to be installed.  Use this script to explore
simulation settings and visualise trajectories in your browser.

Usage::

    PYTHONPATH=src python examples/run_streamlit_app.py

Alternatively, you can run the app directly with::

    streamlit run src/flight/web/app.py
"""

from __future__ import annotations

import subprocess
import sys

try:
    import importlib.resources as resources
except ImportError:  # pragma: no cover
    # Python <3.9 fallback
    import importlib_resources as resources  # type: ignore


def main() -> None:
    # Ensure Streamlit is installed
    try:
        import streamlit  # noqa: F401  # pragma: no cover
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required to run the web app. Install it with `pip install streamlit`."
        ) from exc
    # Locate the app.py file packaged in flight.web
    try:
        with resources.files("flight.web").joinpath("app.py") as path:  # type: ignore[attr-defined]
            app_path = str(path)
    except Exception:
        # Fallback: assume relative path if importlib.resources fails
        app_path = "src/flight/web/app.py"
    # Run the Streamlit command
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    main()