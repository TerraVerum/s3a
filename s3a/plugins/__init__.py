import pydoc
import sys
from functools import lru_cache

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

def INTERNAL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin, MultiPredictionsPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [FilePlugin, EditPlugin, MultiPredictionsPlugin, VerticesPlugin, MainImagePlugin, CompTablePlugin,
          RandomToolsPlugin, HelpPlugin]

@lru_cache()
def EXTERNAL_PLUGINS():
  discoveredPlgs = entry_points(group='s3a.plugins')
  externPlgs = []
  for ep in discoveredPlgs:
    member = pydoc.locate(ep.value)
    if member:
      plgs = member()
      for plg in plgs:
        if plg not in externPlgs:
          externPlgs.append(plg)
  return externPlgs