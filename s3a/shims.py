"""
All API discrepancies between different systems, python versions, etc. should be
resolved here if remotely possible
"""
import sys

__all__ = ["entry_points", "typing_extensions"]

if sys.version_info < (3, 8):
    import typing_extensions
    from importlib_metadata import entry_points
else:
    import typing as typing_extensions
    from importlib.metadata import entry_points
