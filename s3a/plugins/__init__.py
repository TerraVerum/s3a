import pydoc
import sys
import typing
import warnings
from functools import lru_cache
from importlib.metadata import entry_points

from .base import ParameterEditorPlugin


def INTERNAL_PLUGINS():
    # Special case: FilePlugin is ommitted since it is added explicitly by
    # the main window
    from .edit import EditPlugin
    from .help import HelpPlugin
    from .mainimage import MainImagePlugin
    from .multipred import MultiPredictionsPlugin
    from .table import ComponentTablePlugin
    from .tablefield import VerticesPlugin
    from .tools import ToolsPlugin
    from .usermetrics import UserMetricsPlugin

    return [
        EditPlugin,
        MultiPredictionsPlugin,
        VerticesPlugin,
        MainImagePlugin,
        ComponentTablePlugin,
        ToolsPlugin,
        HelpPlugin,
        UserMetricsPlugin,
    ]


_nonEntryPointExternalPlugins = []


@lru_cache()
def EXTERNAL_PLUGINS():
    # Account for warnings in py3.10
    ep = entry_points()
    if sys.version_info >= (3, 10):
        discoveredPlgs = ep.select(group="s3a.plugins")
    else:
        discoveredPlgs = ep.get("s3a.plugins", [])
    externPlgs = _nonEntryPointExternalPlugins.copy()

    def fallback():
        warnings.warn(
            f"'{ep.value}' did not expose a callable. No plugins were found.",
            UserWarning,
            stacklevel=2,
        )
        return []

    for ep in discoveredPlgs:
        # ALL_PLUGINS is usually exposed here
        member = pydoc.locate(ep.value)
        if member:
            # Avoid "pydoc returns object" warning by getattr access to callable
            # Also defaults to no plugins if a bogus entry point was specified
            plgs = getattr(member, "__call__", fallback)()
            for plg in plgs:
                if plg not in externPlgs:
                    externPlgs.append(plg)
    return externPlgs


def addExternalPlugin(pluginClass: typing.Type[ParameterEditorPlugin]):
    if pluginClass not in _nonEntryPointExternalPlugins:
        _nonEntryPointExternalPlugins.append(pluginClass)
