from qtextras import ParameterEditor

from .constants import GENERAL_PROPERTIES_DIR, SCHEMES_DIR
from .parameditors.quickloader import QuickLoaderEditor
from .plugins.base import ParameterEditorPlugin


class SettingsPlugin(ParameterEditorPlugin):
    createProcessMenu = True
    createDock = True

    def __init__(self):
        super().__init__(name="Settings")
        self.generalProperties = ParameterEditor(
            name="App Settings", directory=GENERAL_PROPERTIES_DIR, suffix=".genprops"
        )
        self.colorScheme = ColorSchemePlugin(
            name="Color Scheme", directory=SCHEMES_DIR, suffix=".scheme"
        )
        self.extraDocks = []

    def attachToWindow(self, window):
        super().attachToWindow(window)
        # Remove "show settings" action from the menu
        self.menu.removeAction(self.menu.actions()[0])
        for editor in self.generalProperties, self.colorScheme:
            dock, menu = editor.createWindowDock(window, createProcessMenu=False)
            self.extraDocks.append(dock)
            self.menu.addActions(menu.actions())


class ColorSchemePlugin(ParameterEditorPlugin):
    createDock = True


class SharedAppSettings:
    def __init__(self):
        self.settingsPlugin = SettingsPlugin()
        self.generalProperties = self.settingsPlugin.generalProperties
        self.colorScheme = self.settingsPlugin.colorScheme

        self.quickLoader = QuickLoaderEditor()
