from __future__ import annotations
import warnings
from pathlib import Path
from typing import List, Union

from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtCore, QtWidgets
from qtextras import ParameterEditor, PopupLineEditor
from qtextras.shims import ActionGroupParameter

from ..constants import QUICK_LOAD_DIR
from ..generalutils import lowerNoSpaces
from ..logger import getAppLogger
from ..plugins import ParameterEditorPlugin


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
            if e not in self.uniqueEditors and e.directory is not None
        ]
        self.uniqueEditors.extend(editorList)
        self.layoutAboutToBeChanged.emit()
        for editor in editorList:
            for stateName in self.getParameterStateFiles(
                editor.directory, editor.suffix
            ):
                self.parameterStates.append(stateName)
                self.editorList.append(editor)
            editor.stateManager.signals.created.connect(
                lambda name, e=editor: self.addOptionsForEditor(e, name)
            )
        self.layoutChanged.emit()

    def addOptionsForEditor(self, editor: ParameterEditor, names: list[str]):
        self.layoutAboutToBeChanged.emit()
        for name in names:
            if (
                self.displayFormat.format(editor=editor, stateName=name)
                in self.displayedData
            ):
                return
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


class QuickLoaderEditor(ParameterEditor):
    def __init__(self, editorList: List[ParameterEditor] = None):
        if editorList is None:
            editorList = []
        self.listModel = EditorListModel(editorList)
        self.addNewEditorState = PopupLineEditor(
            model=self.listModel, clearOnComplete=False
        )
        self.applyButton = QtWidgets.QPushButton("Load All First States")
        super().__init__(
            name="Quick Loader", directory=QUICK_LOAD_DIR, suffix=".loader"
        )
        # Now that `self` is initialized, it can be used as a parent
        for widget in self.listModel, self.addNewEditorState:
            widget.setParent(self)

        # self.addNewEditorState.completer().activated.connect(self.addFromLineEdit)
        self.addNewEditorState.returnPressed.connect(self.addFromLineEdit)
        self.applyButton.clicked.connect(self.loadFirstStateFromEachEditor)
        self.applyButton.setToolTip(self.loadFirstStateFromEachEditor.__doc__)

    def _guiChildren(self) -> list:
        return [self.addNewEditorState, self.applyButton, *super()._guiChildren()]

    def saveParameterValues(
        self, saveName: str = None, stateDict: dict = None, **kwargs
    ):
        kwargs.pop("addDefaults", None)
        return super().saveParameterValues(
            saveName, stateDict, **kwargs, addDefaults=True
        )

    def buildFromStartupParameters(self, startupSource: dict):
        # If quick loader is given along with other params, use the quick loader as the
        # base and apply other settings on top of it
        errSettings = []
        # Ignore case and spacing on input keys
        startupSource = {lowerNoSpaces(kk): vv for kk, vv in startupSource.items()}

        for editor in self.listModel.uniqueEditors:  # type: ParameterEditor
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
                stacklevel=2,
            )
        return startupSource

    def addEditor(self, editor: ParameterEditor):
        self.listModel.addEditors([editor])

    def addPlugin(self, plugin: ParameterEditorPlugin):
        self.listModel.addEditors(plugin.registeredEditors)

    def loadParameterValues(
        self,
        stateName: Union[str, Path] = None,
        stateDict: dict = None,
        addDefaults=True,
        **kwargs,
    ):
        stateDict = self.stateManager.loadState(stateName, stateDict)
        if addDefaults:
            # Default is an empty state
            self.rootParameter.clearChildren()
        for editorName, options in stateDict.items():
            self.loadEditorDict(editorName, options)

        return super().loadParameterValues(
            stateName, stateDict, addDefaults=False, candidateParameters=[]
        )

    def loadFirstStateFromEachEditor(self):
        """
        Load the first state from each present editor group. This is useful for
        quickly changing multiple aspects of the program at once.
        """
        for grp in filter(Parameter.hasChildren, self.rootParameter.children()):
            act: ActionGroupParameter = next(iter(grp))
            act.activate()

    def loadEditorDict(self, editorName: str, options: dict):
        matches = [e for e in self.listModel.uniqueEditors if e.name == editorName]
        if len(matches) != 1:
            raise ValueError(
                f'Exactly one editor name must match "{editorName}" but '
                f"{len(matches)} were found"
            )
        editor = matches[0]
        for state, shortcut in options.items():
            # Shortcut is nested under a "shortcut" key when set using the gui
            if isinstance(shortcut, dict):
                shortcut = shortcut["shortcut"]
            self.addActionForEditor(editor, state, shortcut)

    def addFromLineEdit(self):
        try:
            selectionIdx = self.listModel.displayedData.index(
                self.addNewEditorState.text()
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
        # self.addNewEditorState.clear()

    def addActionForEditor(
        self, editor: ParameterEditor, stateDict: str, shortcut: str = None
    ):
        """
        Ensures the specified editor shortcut will exist in the quickloader parameter
        tree. The action can either be None (if no shortcut should be defaulted) or the
        starting shortcut value.
        """
        if shortcut is None:
            shortcut = ""
        groupOpts = dict(name=editor.name, type="group", editor=editor)
        actionOpts = dict(
            name=stateDict,
            removable=True,
            type="_actiongroup",
            button=dict(visible=True, title="load"),
            expanded=False,
        )
        seqOpts = dict(
            name="shortcut", value=shortcut, removable=True, type="keysequence"
        )
        action = self.rootParameter.addChild(groupOpts, existOk=True).addChild(
            actionOpts, existOk=True
        )
        # For some reason, the action doesn't do well with an initial shortcut,
        # so set explicitly here
        action.setButtonOpts(shortcut=shortcut)
        param = action.addChild(seqOpts, existOk=True)

        param.sigValueChanged.connect(
            self.onKeySequenceChanged, QtCore.Qt.ConnectionType.UniqueConnection
        )
        action.sigActivated.connect(
            self.onActionActivated, QtCore.Qt.ConnectionType.UniqueConnection
        )

    def onActionActivated(self, action: Parameter):
        state = action.name()
        editor = action.parent().opts["editor"]
        self._safeLoadParameterValues(action, editor, state)

    def onKeySequenceChanged(self, keyParam, value):
        if value is None:
            value = ""
        assert isinstance(parent := keyParam.parent(), ActionGroupParameter)
        parent.setButtonOpts(shortcut=value)

    def _safeLoadParameterValues(
        self, action: Parameter, editor: ParameterEditor, stateDict: str
    ):
        """
        It is possible for the quick loader to refer to a parameter state that no longer
        exists. When this happens, failure should be graceful and the action should be
        deleted
        """
        try:
            editor.loadParameterValues(stateDict)
        except FileNotFoundError:
            action.remove()
            # Wait until end of process cycle to raise error
            formattedState = self.listModel.displayFormat.format(
                editor=editor, stateName=stateDict
            )
            getAppLogger(__name__).critical(
                f"Attempted to load {formattedState} but the setting was not found."
            )
