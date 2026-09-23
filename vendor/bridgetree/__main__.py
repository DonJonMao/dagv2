"""Module entry point for ``python -m bridgetree``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    sys.exit(main())
