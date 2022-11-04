import functools
import warnings
from pathlib import Path
from typing import List, Union

from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtCore, QtWidgets
from qtextras import (
    ParameterEditor,
    PopupLineEditor,
    attemptFileLoad,
    getParameterChild,
)

from . import MetaTreeParameterEditor
from ..constants import QUICK_LOAD_DIR
from ..generalutils import lowerNoSpaces
from ..logger import getAppLogger


class EditorListModel(QtCore.QAbstractListModel):
    def __init__(
        self, editorList: List[ParameterEditor], parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.displayFormat = "{stateName} | {editor.name}"
        self.parameterStates: List[str] = []
        self.editorList: List[ParameterEditor] = []
        self.uniqueEditors: List[ParameterEditor] = []

        self.addEditors(editorList)

    def addEditors(self, editorList: List[ParameterEditor]):
        editorList = [
            e
            for e in editorList
            if e not in self.uniqueEditors and e.stateManager.directory is not None
        ]
        self.uniqueEditors.extend(editorList)
        self.layoutAboutToBeChanged.emit()
        for editor in editorList:
            for stateName in self.getParameterStateFiles(
                editor.stateManager.directory, editor.stateManager.suffix
            ):
                self.parameterStates.append(stateName)
                self.editorList.append(editor)
            editor.stateManager.signals.created.connect(
                lambda name, e=editor: self.addOptionForEditor(e, name)
            )
        self.layoutChanged.emit()

    def addOptionForEditor(self, editor: ParameterEditor, name: str):
        if (
            self.displayFormat.format(editor=editor, stateName=name)
            in self.displayedData
        ):
            return
        self.layoutAboutToBeChanged.emit()
        self.parameterStates.append(name)
        self.editorList.append(editor)
        self.layoutChanged.emit()

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ):
        row = index.row()
        paramState = self.parameterStates[row]
        editor = self.editorList[row]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.displayFormat.format(stateName=paramState, editor=editor)
        elif role == QtCore.Qt.ItemDataRole.EditRole:
            return paramState, editor
        else:
            return

    def stringList(self):
        return self.displayedData

    @property
    def displayedData(self):
        return [
            self.displayFormat.format(stateName=name, editor=editor)
            for name, editor in zip(self.parameterStates, self.editorList)
        ]

    def rowCount(self, paren=QtCore.QModelIndex()) -> int:
        return len(self.parameterStates)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        return "Parameter State List"

    @staticmethod
    def getParameterStateFiles(directory: Path, suffix: str) -> List[str]:
        if directory is None:
            return []
        files = directory.glob(f"*{suffix}")
        return [file.stem for file in files]


class QuickLoaderEditor(MetaTreeParameterEditor):
    def __init__(self, editorList: List[ParameterEditor] = None):
        if editorList is None:
            editorList = []
        self.listModel = EditorListModel(editorList)
        self.addNewParamState = PopupLineEditor(
            model=self.listModel, clearOnComplete=False
        )

        super().__init__(
            name="Quick Loader", directory=QUICK_LOAD_DIR, suffix=".loader"
        )
        # Now that `self` is initialized, it can be used as a parent
        for widget in self.listModel, self.addNewParamState:
            widget.setParent(self)

        # self.addNewParamState.completer().activated.connect(self.addFromLineEdit)
        self.addNewParamState.returnPressed.connect(self.addFromLineEdit)

    def _guiChildren(self) -> list:
        return [self.addNewParamState, *super()._guiChildren()]

    def saveParameterValues(
        self, saveName: str = None, parameterState: dict = None, **kwargs
    ):
        kwargs.pop("includeDefaults", None)
        return super().saveParameterValues(
            saveName, parameterState, **kwargs, includeDefaults=True
        )

    def buildFromStartupParameters(self, startupSource: dict):
        # If quick loader is given along with other params, use the quick loader as the
        # base and apply other settings on top of it
        errSettings = []
        # Ignore case and spacing on input keys
        startupSource = {lowerNoSpaces(kk): vv for kk, vv in startupSource.items()}

        for editor in [self] + self.listModel.uniqueEditors:  # type: ParameterEditor
            paramStateInfo: Union[dict, str] = startupSource.get(
                lowerNoSpaces(editor.name), None
            )
            try:
                if isinstance(paramStateInfo, dict):
                    editor.loadParameterValues(self.stateName, paramStateInfo)
                elif paramStateInfo is not None:
                    editor.loadParameterValues(paramStateInfo)
            except Exception as ex:
                errSettings.append(f"{editor.name}: {ex}")
        if len(errSettings) > 0:
            warnings.warn(
                "The following settings could not be loaded (shown as [setting]: "
                "[exception])\n" + "\n\n".join(errSettings),
                UserWarning,
                stacklevel=3,
            )
        return startupSource

    def addEditor(self, editor: ParameterEditor):
        self.listModel.addEditors([editor])

    def loadParameterValues(
        self,
        stateName: Union[str, Path] = None,
        stateDict: dict = None,
        useDefaults=True,
        **kwargs,
    ):
        if stateDict is None:
            stateDict = attemptFileLoad(self.stateManager.formatFileName(stateName))
        if useDefaults:
            self.rootParameter.clearChildren()
        if len(stateDict):
            for editorName, shcOpts in stateDict.items():
                matches = [
                    e for e in self.listModel.uniqueEditors if e.name == editorName
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f'Exactly one editor name must match "{editorName}" but '
                        f"{len(matches)} were found"
                    )
                editor = matches[0]
                for state, shcValue in shcOpts.items():
                    self.addActionForEditor(editor, state, shcValue)
        return super().loadParameterValues(
            stateName, stateDict, useDefaults=False, candidateParameters=[]
        )

    def addFromLineEdit(self):
        try:
            selectionIdx = self.listModel.displayedData.index(
                self.addNewParamState.text()
            )
        except ValueError:
            selectionIdx = None
        if selectionIdx is None:
            return
        qtSelectionIdx = self.listModel.index(selectionIdx)
        # selectionIdx = completer.popup().currentIndex()
        # if not selectionIdx.isValid():
        #   selectionIdx = completer.currentIndex()
        paramState, editor = qtSelectionIdx.data(QtCore.Qt.ItemDataRole.EditRole)
        self.addActionForEditor(editor, paramState)
        # self.addNewParamState.clear()

    def addActionForEditor(
        self, editor: ParameterEditor, parameterState: str, shortcut: str = None
    ):
        """
        Ensures the specified editor shortcut will exist in the quickloader parameter
        tree. The action can either be None (if no shortcut should be defaulted) or the
        starting shortcut value.
        """
        act = getParameterChild(
            self.rootParameter,
            editor.name,
            parameterState,
            groupOpts=dict(removable=True),
            childOpts=dict(
                name=parameterState,
                value=shortcut or "",
                removable=True,
                type="keysequence",
            ),
        )

        # Ensure the value matches this new action in the event it already existed
        # Also set `removable` in case this was added through a different execution
        # path
        act.setOpts(removable=True, value=shortcut or "")
        if not (qShortcut := act.opts.get("shortcut")):
            qShortcut = QtWidgets.QShortcut(
                act.value(),
                context=QtCore.Qt.ShortcutContext.ApplicationShortcut,
            )
            qShortcut.activated.connect(
                functools.partial(
                    self._safeLoadParameterValues, act, editor, parameterState
                )
            )
            act.setOpts(shortcut=qShortcut)
        else:
            qShortcut.setKey(act.value())

    def _safeLoadParameterValues(
        self, action: Parameter, editor: ParameterEditor, parameterState: str
    ):
        """
        It is possible for the quick loader to refer to a parameter state that no longer
        exists. When this happens, failure should be graceful and the action should be
        deleted
        """
        try:
            editor.loadParameterValues(parameterState)
        except FileNotFoundError:
            action.opts["shortcut"].deleteLater()
            del action.opts["shortcut"]
            action.remove()
            # Wait until end of process cycle to raise error
            formattedState = self.listModel.displayFormat.format(
                editor=editor, stateName=parameterState
            )
            getAppLogger(__name__).critical(
                f"Attempted to load {formattedState} but the setting was not found."
            )
