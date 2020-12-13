def ALL_PLUGINS():
  from .misc import MainImagePlugin, CompTablePlugin, RandomToolsPlugin, EditPlugin, HelpPlugin
  from .file import FilePlugin
  from .tablefield import VerticesPlugin

  return [VerticesPlugin, FilePlugin, EditPlugin, MainImagePlugin, CompTablePlugin, RandomToolsPlugin, HelpPlugin]