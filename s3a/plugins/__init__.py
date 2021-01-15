def ALL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin, GlobalPredictionsPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [VerticesPlugin, GlobalPredictionsPlugin, EditPlugin, MainImagePlugin, CompTablePlugin, RandomToolsPlugin, HelpPlugin]