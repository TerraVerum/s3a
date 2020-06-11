from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter, Parameter

from s3a.projectvars import QUICK_LOAD_DIR
from .genericeditor import FRParamEditor
from ..graphicsutils import FRPopupLineEditor, raiseErrorLater
from ...structures import FRIllRegisteredPropError

Slot = QtCore.pyqtSlot

class FREditorListModel(QtCore.QAbstractListModel):
  def __init__(self, editorList: List[FRParamEditor], parent: QtWidgets.QWidget=None):
    super().__init__(parent)
    self.displayFormat = '{stateName} | {editor.name}'
    self.paramStatesLst: List[str] = []
    self.editorList: List[FRParamEditor] = []
    self.uniqueEditors: List[FRParamEditor] = []

    self.addEditors(editorList)

  def addEditors(self, editorList: List[FRParamEditor]):
    self.uniqueEditors.extend(editorList)
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
  def getParamStateFiles(stateDir: Path, fileExt: str) -> List[str]:
    files = stateDir.glob(f'*.{fileExt}')
    return [file.stem for file in files]


def _addRmOption(param: Parameter):
  item = next(iter(param.items.keys()))

  try:
    menu: QtWidgets.QMenu = item.contextMenu
  except AttributeError: # If the menu doesn't exist yet
    menu = QtWidgets.QMenu()
    item.contextMenu = menu
  existingActs = [act.text() for act in menu.actions()]
  if 'Remove' not in existingActs:
    item.contextMenu.addAction('Remove').triggered.connect(item.requestRemove)


class FRQuickLoaderEditor(FRParamEditor):
  def __init__(self, parent=None, editorList: List[FRParamEditor]=None):
    super().__init__(parent, paramList=[],
                     saveDir=QUICK_LOAD_DIR, fileType='loader')
    if editorList is None:
      editorList = []
    self.listModel = FREditorListModel(editorList, self)

    self.addNewParamState = FRPopupLineEditor(self, self.listModel)
    self.addNewParamState.setPlaceholderText('Press Tab or type...')
    self.centralLayout.insertWidget(0, self.addNewParamState)

    self.addNewParamState.completer().activated.connect(self.addFromLineEdit)

  def loadParamState(self, stateName: str, stateDict: dict=None, addChildren=False, removeChildren=False):
    ret = super().loadParamState(stateName, stateDict, addChildren=True, removeChildren=True)
    invalidGrps = []
    editorNames = [e.name for e in self.listModel.uniqueEditors]
    hasInvalidEntries = False
    for grp in self.params: # type: GroupParameter
      name = grp.name()
      # Get corresponding editor
      try:
        idx = editorNames.index(name)
        editor = self.listModel.uniqueEditors[idx]
      except ValueError:
        invalidGrps.append(grp)
        hasInvalidEntries = True
        continue
      for act in grp: # type: ActionParameter
        self.addActForEditor(editor, act.name(), act)
    for grp in invalidGrps:
      grp.remove()
    if hasInvalidEntries:
      errMsg = f"The following editors were not recognized:\n" \
               f"{[grp.name() for grp in invalidGrps]}\n" \
               f"Must be one of:\n" \
               f"{[e.name for e in self.listModel.uniqueEditors]}"
      raiseErrorLater(FRIllRegisteredPropError(errMsg))
    self.applyBtnClicked()
    return ret

  def buildFromUserProfile(self, profileSrc: dict):
    # If quick loader is given along with other params, use the quick loader as the
    # base and apply other settings on top of it
    selfStateName = profileSrc.get(self.name, None)
    if selfStateName is not None:
      self.loadParamState(selfStateName)

    for editor in self.listModel.uniqueEditors:
      paramStateName = profileSrc.get(editor.name, None)
      if paramStateName is not None:
        editor.loadParamState(paramStateName)
    return profileSrc


  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False):
    stateDict = self.paramDictWithOpts(['type'], [ActionParameter, GroupParameter])
    super().saveParamState(saveName, stateDict, allowOverwriteDefault)


  def applyBtnClicked(self):
    super().applyBtnClicked()
    for grp in self.params.childs: # type: GroupParameter
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
    self.addActForEditor(editor, paramState)


  def addActForEditor(self, editor: FRParamEditor, paramState: str, act: ActionParameter=None):
    if editor.name not in self.params.names:
      curGroup = self.params.addChild(dict(name=editor.name, type='group', removable=True))
    else:
      # Bug: If 'removable' is not specified on construction of the parameter item,
      # It is not made possible through the context menu. Fix this
      curGroup = self.params.names[editor.name]
      _addRmOption(curGroup)

    curGroup.opts['removable'] = True
    if act is None and paramState not in curGroup.names:
      act = ActionParameter(name=paramState, removable=True, type='action')
      curGroup.addChild(act)
    elif act is not None:
      act.opts['removable'] = True
      _addRmOption(act)
    act.sigActivated.connect(
      lambda _act: self._safeLoadParamState(_act, editor,paramState))

  def _safeLoadParamState(self, action: ActionParameter, editor: FRParamEditor,
                          paramState: str):
    """
    It is possible for the quick loader to refer to a param state that no longer
    exists. When this happens, failure should be graceful and the action should be
    deleted
    """
    try:
      editor.loadParamState(paramState)
    except FileNotFoundError:
      action.remove()
      # Wait until end of process cycle to raise error
      formattedState = self.listModel.displayFormat.format(editor=editor, stateName=paramState)
      raiseErrorLater(FRIllRegisteredPropError(
        f'Attempted to load {formattedState} but the setting was not found.'
      ))