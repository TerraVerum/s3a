from pathlib import Path
from typing import List, Union, Sequence
from warnings import warn

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree.parameterTypes import GroupParameter, Parameter

from s3a.graphicsutils import PopupLineEditor, raiseErrorLater
from s3a.constants import QUICK_LOAD_DIR
from . import ParamEditorDockGrouping
from .genericeditor import ParamEditor
from .pgregistered import ActionWithShortcutParameter as ActWithShc
from s3a.structures import ParamEditorError, S3AWarning


class EditorListModel(QtCore.QAbstractListModel):
  def __init__(self, editorList: List[ParamEditor], parent: QtWidgets.QWidget=None):
    super().__init__(parent)
    self.displayFormat = '{stateName} | {editor.name}'
    self.paramStatesLst: List[str] = []
    self.editorList: List[ParamEditor] = []
    self.uniqueEditors: List[ParamEditor] = []

    self.addEditors(editorList)

  def addEditors(self, editorList: List[ParamEditor]):
    editorList = [e for e in editorList if e not in self.uniqueEditors and e.saveDir is not None]
    self.uniqueEditors.extend(editorList)
    self.layoutAboutToBeChanged.emit()
    for editor in editorList:
      for stateName in self.getParamStateFiles(editor.saveDir, editor.fileType):
        self.paramStatesLst.append(stateName)
        self.editorList.append(editor)
      editor.sigParamStateCreated.connect(lambda name, e=editor:
                                          self.addOptForEditor(e, name))
    self.layoutChanged.emit()

  def addOptForEditor(self, editor: ParamEditor, name: str):
    if self.displayFormat.format(editor=editor, stateName=name) in self.displayedData:
      return
    self.layoutAboutToBeChanged.emit()
    self.paramStatesLst.append(name)
    self.editorList.append(editor)
    self.layoutChanged.emit()

  # def updateEditorOpts(self, editor: FRParamEditor):
  #   self.layoutAboutToBeChanged.emit()
  #   for ii in range(len(self.paramStatesLst) - 1, -1, -1):
  #     if self.editorList[ii] is editor:
  #       del self.paramStatesLst[ii]
  #       del self.editorList[ii]
  #   self.addEditors([editor])
  #   self.layoutChanged.emit()

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

  def stringList(self):
    return self.displayedData

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
    if stateDir is None: return []
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


class QuickLoaderEditor(ParamEditor):
  def __init__(self, parent=None, editorList: List[ParamEditor]=None):
    super().__init__(parent, paramList=[],
                     saveDir=QUICK_LOAD_DIR, fileType='loader', name='Editor State Shortcuts')
    if editorList is None:
      editorList = []
    self.listModel = EditorListModel(editorList, self)

    self.addNewParamState = PopupLineEditor(self, self.listModel, clearOnComplete=False)
    self.centralLayout.insertWidget(0, self.addNewParamState)

    # self.addNewParamState.completer().activated.connect(self.addFromLineEdit)
    self.addNewParamState.editingFinished.connect(self.addFromLineEdit)

  def loadParamState(self, stateName: Union[str, Path], stateDict: dict=None,
                     addChildren=False, removeChildren=False, applyChanges=True):
    ret = super().loadParamState(stateName, stateDict, addChildren=True, removeChildren=True,
                                 applyChanges=False)
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
      for act in grp: # type: ActWithShc
        self.addActForEditor(editor, act.name(), act)
    for grp in invalidGrps:
      grp.remove()
    if hasInvalidEntries:
      errMsg = f"The following editors were not recognized:\n" \
               f"{[grp.name() for grp in invalidGrps]}\n" \
               f"Must be one of:\n" \
               f"{[e.name for e in self.listModel.uniqueEditors]}"
      warn(errMsg, S3AWarning)
    if applyChanges:
      self.applyChanges()
    return ret

  def buildFromUserProfile(self, profileSrc: dict):
    # If quick loader is given along with other params, use the quick loader as the
    # base and apply other settings on top of it
    errSettings = []

    for editor in [self] + self.listModel.uniqueEditors:
      paramStateName = profileSrc.get(editor.name.replace(' ', '').lower(), None)
      if paramStateName is not None:
        try:
          editor.loadParamState(paramStateName)
        except Exception as ex:
          errSettings.append(f'{editor.name}: {ex}')
    if len(errSettings) > 0:
      warn('The following settings could not be loaded (shown as <setting>: <exception>)\n'
           + "\n\n".join(errSettings), S3AWarning)
    return profileSrc

  def addDock(self, dock: Union[ParamEditor, ParamEditorDockGrouping]):
    if isinstance(dock, ParamEditorDockGrouping):
      self.listModel.addEditors(dock.editors)
    else:
      self.listModel.addEditors([dock])

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False, blockWrite=False):
    stateDict = self.paramDictWithOpts(['type'], [ActWithShc, GroupParameter])
    super().saveParamState(saveName, stateDict, allowOverwriteDefault, blockWrite)


  def applyChanges(self):
    super().applyChanges()
    for grp in self.params.childs: # type: GroupParameter
      if grp.hasChildren():
        act: ActWithShc = next(iter(grp))
        act.activate()

  def addFromLineEdit(self):
    completer = self.addNewParamState.completer()
    selection = completer.completionModel()
    try:
      selectionIdx = self.listModel.displayedData.index(self.addNewParamState.text())
    except ValueError:
      selectionIdx = None
    if selectionIdx is None:
      return
    qtSelectionIdx = self.listModel.index(selectionIdx)
    # selectionIdx = completer.popup().currentIndex()
    # if not selectionIdx.isValid():
    #   selectionIdx = completer.currentIndex()
    paramState, editor = qtSelectionIdx.data(QtCore.Qt.EditRole)
    self.addActForEditor(editor, paramState)
    self.addNewParamState.clear()


  def addActForEditor(self, editor: ParamEditor, paramState: str, act: ActWithShc=None):
    if editor.name not in self.params.names:
      curGroup = self.params.addChild(dict(name=editor.name, type='group', removable=True))
    else:
      # Bug: If 'removable' is not specified on construction of the parameter item,
      # It is not made possible through the context menu. Fix this
      curGroup = self.params.names[editor.name]
      _addRmOption(curGroup)
      if act is None and paramState in curGroup.names:
        act = curGroup.child(paramState)

    if paramState in curGroup.names and act is not None and act.isActivateConnected:
      # Duplicate option, no reason to add
      return
    curGroup.opts['removable'] = True
    if act is None:
      act = ActWithShc(name=paramState, removable=True, type='actionwithshortcut')
    curGroup.addChild(act)

    act.opts['removable'] = True
    _addRmOption(act)
    act.sigActivated.connect(
      lambda _act: self._safeLoadParamState(_act, editor, paramState))
    act.isActivateConnected = True

  def _safeLoadParamState(self, action: ActWithShc, editor: ParamEditor,
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
      raiseErrorLater(ParamEditorError(
        f'Attempted to load {formattedState} but the setting was not found.'
      ))