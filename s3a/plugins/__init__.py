import pydoc
import typing
import warnings
from functools import lru_cache

from utilitys import ParamEditorPlugin

from ..shims import entry_points


def INTERNAL_PLUGINS():
    from .misc import (
        RandomToolsPlugin,
    )
    from .edit import EditPlugin
    from .comptable import CompTablePlugin
    from .mainimage import MainImagePlugin
    from .help import HelpPlugin
    from .multipred import MultiPredictionsPlugin
    from .usermetrics import UserMetricsPlugin
    from .file import FilePlugin
    from .tablefield import VerticesPlugin

    return [
        FilePlugin,
        EditPlugin,
        MultiPredictionsPlugin,
        VerticesPlugin,
        MainImagePlugin,
        CompTablePlugin,
        RandomToolsPlugin,
        HelpPlugin,
        UserMetricsPlugin,
    ]


_nonEntryPointExternalPlugins = []


@lru_cache()
def EXTERNAL_PLUGINS():
    discoveredPlgs = entry_points().get("s3a.plugins", [])
    externPlgs = _nonEntryPointExternalPlugins.copy()

    def fallback():
        warnings.warn(
            f"'{ep.value}' did not expose a callable. No plugins were found.",
            UserWarning,
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


def addExternalPlugin(pluginClass: typing.Type[ParamEditorPlugin]):
    if pluginClass not in _nonEntryPointExternalPlugins:
        _nonEntryPointExternalPlugins.append(pluginClass)
