"""
All API discrepancies between different systems, python versions, etc. should be resolved here if remotely possible
"""
import sys

__all__ = ['entry_points', 'typing_extensions']

if sys.version_info < (3, 8):
    from importlib_metadata import entry_points
    import typing_extensions
else:
    from importlib.metadata import entry_points
    import typing as typing_extensions