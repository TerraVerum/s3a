from qtextras import ParameterEditor

from .constants import GENERAL_PROPERTIES_DIR, SCHEMES_DIR
from .parameditors.quickloader import QuickLoaderEditor
from .plugins.base import ParameterEditorPlugin


class GeneralPropertiesPlugin(ParameterEditorPlugin):
    pass


class ColorSchemePlugin(ParameterEditorPlugin):
    pass


class SharedAppSettings:
    def __init__(self):
        self.generalProperties = GeneralPropertiesPlugin(
            name="App Settings", directory=GENERAL_PROPERTIES_DIR, suffix=".genprops"
        )
        self.colorScheme = ColorSchemePlugin(
            name="Color Scheme", directory=SCHEMES_DIR, suffix=".scheme"
        )

        self.quickLoader = QuickLoaderEditor()
