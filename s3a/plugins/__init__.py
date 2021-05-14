def ALL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin, MultiPredictionsPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [FilePlugin, EditPlugin, MultiPredictionsPlugin, VerticesPlugin, MainImagePlugin, CompTablePlugin, RandomToolsPlugin, HelpPlugin]