from __future__ import annotations

from utilitys import ParamEditorPlugin, ShortcutParameter

from ..constants import SHORTCUTS_DIR
from ..parameditors.quickloader import QuickLoaderEditor


class ShortcutsPlugin(ParamEditorPlugin):
    name = "Shortcuts"

    def __initEditorParams__(self, **kwargs):
        super().__initEditorParams__(**kwargs)

        self.shortcuts = ShortcutParameter.setRegistry(
            createIfNone=True, saveDir=SHORTCUTS_DIR
        )
        self.quickLoader = QuickLoaderEditor()

        self.dock.addEditors([self.shortcuts, self.quickLoader])
