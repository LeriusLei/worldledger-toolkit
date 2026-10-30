"""Regenerate the repository figures (compatibility entry point)."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("create_visuals.py")), run_name="__main__")
