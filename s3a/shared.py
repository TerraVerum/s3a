from qtextras import ParameterEditor

from .constants import GENERAL_PROPERTIES_DIR, SCHEMES_DIR
from .parameditors.quickloader import QuickLoaderEditor


class SharedAppSettings:
    def __init__(self):
        self.generalProperties = ParameterEditor(
            name="App Settings", directory=GENERAL_PROPERTIES_DIR, suffix=".genprops"
        )
        self.colorScheme = ParameterEditor(
            name="Color Scheme", directory=SCHEMES_DIR, suffix=".scheme"
        )

        self.quickLoader = QuickLoaderEditor()
