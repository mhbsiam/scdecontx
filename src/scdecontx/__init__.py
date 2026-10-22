"""DecontX for scverse: remove ambient RNA contamination from single-cell data."""

from importlib.metadata import version as _version

__version__ = _version("scdecontx")

from ._run import run

__all__ = ["run"]
