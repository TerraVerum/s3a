import pydoc
import typing
from functools import lru_cache
from s3a.shims import entry_points
from utilitys import ParamEditorPlugin


def INTERNAL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin, MultiPredictionsPlugin
  from .usermetrics import UserMetricsPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [FilePlugin, EditPlugin, MultiPredictionsPlugin, VerticesPlugin, MainImagePlugin, CompTablePlugin,
          RandomToolsPlugin, HelpPlugin, UserMetricsPlugin]

_nonEntryPointExternalPlugins = []

@lru_cache()
def EXTERNAL_PLUGINS():
  discoveredPlgs = entry_points().get('s3a.plugins', [])
  externPlgs = _nonEntryPointExternalPlugins.copy()
  # noinspection PyTypeChecker
  for ep in discoveredPlgs:
    # ALL_PLUGINS is usually exposed here
    member = pydoc.locate(ep.value)
    if member:
      plgs = member()
      for plg in plgs:
        if plg not in externPlgs:
          externPlgs.append(plg)
  return externPlgs

def addExternalPlugin(pluginClass: typing.Type[ParamEditorPlugin]):
  _nonEntryPointExternalPlugins.append(pluginClass)