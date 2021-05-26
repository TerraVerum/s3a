import pydoc
from functools import lru_cache
from s3a.shims import entry_points

def INTERNAL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin, MultiPredictionsPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [FilePlugin, EditPlugin, MultiPredictionsPlugin, VerticesPlugin, MainImagePlugin, CompTablePlugin,
          RandomToolsPlugin, HelpPlugin]

@lru_cache()
def EXTERNAL_PLUGINS():
  discoveredPlgs = entry_points().get('s3a.plugins', [])
  externPlgs = []
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