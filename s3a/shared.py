from .plugins.settings import SettingsPlugin
from .plugins.shortucts import ShortcutsPlugin


class SharedAppSettings:
    def __init__(self):
        self.settingsPlugin = SettingsPlugin()
        self.colorScheme = self.settingsPlugin.colorScheme
        self.generalProperties = self.settingsPlugin.generalProperties

        self.shortcutsPlugin = ShortcutsPlugin()
        self.shortcuts = self.shortcutsPlugin.shortcuts
        self.quickLoader = self.shortcutsPlugin.quickLoader
