from __future__ import annotations

from utilitys import ParamEditorPlugin, ParamEditor

from ..constants import GEN_PROPS_DIR, SCHEMES_DIR


class SettingsPlugin(ParamEditorPlugin):
    name = "Settings"

    def __initEditorParams__(self, **kwargs):
        super().__initEditorParams__(**kwargs)

        self.generalProps = ParamEditor(
            saveDir=GEN_PROPS_DIR, fileType="genprops", name="App Settings"
        )
        self.colorScheme = ParamEditor(
            saveDir=SCHEMES_DIR, fileType="scheme", name="Color Scheme"
        )

        self.dock.addEditors([self.generalProps, self.colorScheme])
