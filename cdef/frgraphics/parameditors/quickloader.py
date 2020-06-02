from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter

from cdef.projectvars import QUICK_LOAD_DIR
from .genericeditor import FRParamEditor
from ..graphicsutils import FRPopupLineEditor

Slot = QtCore.pyqtSlot

class FREditorListModel(QtCore.QAbstractListModel):
  def __init__(self, editorList: List[FRParamEditor], parent: QtWidgets.QWidget=None):
    super().__init__(parent)
    self.displayFormat = '{stateName} | {editor.name}'
    self.paramStatesLst: List[str] = []
    self.editorList: List[FRParamEditor] = []
    self.addEditors(editorList)

  def addEditors(self, editorList: List[FRParamEditor]):
    self.layoutAboutToBeChanged.emit()
    for editor in editorList:
      for stateName in self.getParamStateFiles(editor.saveDir, editor.fileType):
        self.paramStatesLst.append(stateName)
        self.editorList.append(editor)
      editor.sigParamStateCreated.connect(lambda name, e=editor:
                                          self.addOptForEditor(e, name))
    self.layoutChanged.emit()

  def addOptForEditor(self, editor: FRParamEditor, name: str):
    if self.displayFormat.format(editor=editor, stateName=name) in self.displayedData:
      return
    self.layoutAboutToBeChanged.emit()
    self.paramStatesLst.append(name)
    self.editorList.append(editor)
    self.layoutChanged.emit()

  def updateEditorOpts(self, editor: FRParamEditor):
    self.layoutAboutToBeChanged.emit()
    for ii in range(len(self.paramStatesLst) - 1, -1, -1):
      if self.editorList[ii] is editor:
        del self.paramStatesLst[ii]
        del self.editorList[ii]
    self.addEditors([editor])
    self.layoutChanged.emit()

  def data(self, index: QtCore.QModelIndex, role: int=QtCore.Qt.DisplayRole):
    row = index.row()
    paramState = self.paramStatesLst[row]
    editor = self.editorList[row]
    if role == QtCore.Qt.DisplayRole:
      return self.displayFormat.format(stateName=paramState, editor=editor)
    elif role == QtCore.Qt.EditRole:
      return paramState, editor
    else:
      return

  @property
  def displayedData(self):
    return [self.displayFormat.format(stateName=stng, editor=edtr)
            for stng, edtr in zip(self.paramStatesLst, self.editorList)]

  def rowCount(self, paren=QtCore.QModelIndex()) -> int:
    return len(self.paramStatesLst)

  def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
    return 'Parameter State List'

  @staticmethod
  def getParamStateFiles(stateDir: str, fileExt: str) -> List[str]:
    files = Path(stateDir).glob(f'*.{fileExt}')
    return [file.stem for file in files]

class FRQuickLoaderEditor(FRParamEditor):
  def __init__(self, parent=None, editorList: List[FRParamEditor]=None, onlyOneStatePerEditor=False):
    super().__init__(parent, paramList=[],
                     saveDir=QUICK_LOAD_DIR, fileType='loader')
    if editorList is None:
      editorList = []
    self.listModel = FREditorListModel(editorList, self)

    self.addNewParamState = FRPopupLineEditor(self, self.listModel)
    self.addNewParamState.setPlaceholderText('Press Tab or type...')
    self.centralLayout.insertWidget(0, self.addNewParamState)

    self.addNewParamState.returnPressed.connect(self.addFromLineEdit)

    self.onlyOneStatePerEditor = onlyOneStatePerEditor

  def applyBtnClicked(self):
    super().applyBtnClicked()
    for grp in self.params: # type: GroupParameter
      if grp.hasChildren():
        act: ActionParameter = next(iter(grp))
        act.sigActivated.emit(act)

  @Slot()
  def addFromLineEdit(self):
    completer = self.addNewParamState.completer()
    selection = completer.completionModel()
    if self.addNewParamState.text() not in self.listModel.displayedData:
      return
    selectionIdx = completer.popup().currentIndex()
    if not selectionIdx.isValid():
      selectionIdx = completer.currentIndex()
    paramState, editor = selection.data(selectionIdx, QtCore.Qt.EditRole)
    if editor.name not in self.params.names:
      curGroup = self.params.addChild(dict(name=editor.name, type='group', removable=True))
    else:
      curGroup = self.params.names[editor.name]
    if self.onlyOneStatePerEditor:
      curGroup.clearChildren()
    newChild = ActionParameter(name=paramState, removable=True)
    curGroup.addChild(newChild)
    newChild.sigActivated.connect(lambda act: editor.loadParamState(paramState))