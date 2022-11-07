from __future__ import annotations

import typing as t

from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from qtextras import ParameterEditor

from .constants import GENERAL_PROPERTIES_DIR, SCHEMES_DIR
from .parameditors.quickloader import QuickLoaderEditor
from .plugins.base import ParameterEditorPlugin


class SortableGroupParameter(GroupParameter):
    sortKey: t.Callable[[Parameter], t.Any] | None

    def insertChild(self, pos, child, autoIncrementName=None, existOk=False):
        if isinstance(child, dict):
            child = Parameter.create(**child)
        if self.sortKey is not None:
            pos = self._findInsertPos(child)
        return super().insertChild(pos, child, autoIncrementName, existOk)

    def _findInsertPos(self, child):
        key = self.sortKey(child)
        for i, p in enumerate(self.children()):
            if self.sortKey(p) > key:
                return i
        return len(self.children())


class SettingsPlugin(ParameterEditorPlugin):
    createProcessMenu = False
    createDock = False

    def __init__(self):
        super().__init__(name="Settings")
        self.generalProperties = ParameterEditor(
            name="App Settings", directory=GENERAL_PROPERTIES_DIR, suffix=".genprops"
        )
        self.colorScheme = ColorSchemePlugin(
            name="Color Scheme", directory=SCHEMES_DIR, suffix=".scheme"
        )
        for editor in self.generalProperties, self.colorScheme:
            editor.rootParameter = SortableGroupParameter(**editor.rootParameter.opts)
            editor.rootParameter.sortKey = lambda p: p.name()
            editor.tree.setParameters(editor.rootParameter, showTop=False)
        self.extraDocks = []


class ColorSchemePlugin(ParameterEditorPlugin):
    createDock = True


class SharedAppSettings:
    def __init__(self):
        self.settingsPlugin = SettingsPlugin()
        self.generalProperties = self.settingsPlugin.generalProperties
        self.colorScheme = self.settingsPlugin.colorScheme

        self.quickLoader = QuickLoaderEditor()
