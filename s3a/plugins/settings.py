from __future__ import annotations

from utilitys import ParamEditor, ParamEditorPlugin

from ..constants import GENERAL_PROPERTIES_DIR, SCHEMES_DIR


class SettingsPlugin(ParamEditorPlugin):
    name = "Settings"

    def __initEditorParams__(self, **kwargs):
        super().__initEditorParams__(**kwargs)

        self.generalProperties = ParamEditor(
            saveDir=GENERAL_PROPERTIES_DIR, fileType="genprops", name="App Settings"
        )
        self.colorScheme = ParamEditor(
            saveDir=SCHEMES_DIR, fileType="scheme", name="Color Scheme"
        )

        self.dock.addEditors([self.generalProperties, self.colorScheme])
