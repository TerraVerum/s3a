from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree.parameterTypes import ActionParameter

from cdef.projectvars import USER_PROFILES_DIR
from .genericeditor import FRParamEditor
from ..graphicsutils import FRPopupLineEditor

Slot = QtCore.pyqtSlot

@dataclass
class FRSettingListEntry:
  settingName: str
  editor: FRParamEditor

class FREditorListModel(QtCore.QAbstractListModel):
  def __init__(self, editorList: List[FRParamEditor], parent: QtWidgets.QWidget=None):
    super().__init__(parent)
    self.displayFormat = '{setting} | {editor.name}'
    self.settingsList: List[str] = []
    self.editorList: List[FRParamEditor] = []
    self.addEditors(editorList)

  def addEditors(self, editorList: List[FRParamEditor]):
    self.layoutAboutToBeChanged.emit()
    for editor in editorList:
      for settingName in self.getSettingsFiles(editor.saveDir, editor.fileType):
        self.settingsList.append(settingName)
        self.editorList.append(editor)
      editor.sigParamStateCreated.connect(lambda name, e=editor:
                                          self.addOptForEditor(e, name))
    self.layoutChanged.emit()

  def addOptForEditor(self, editor: FRParamEditor, name: str):
    if self.displayFormat.format(editor=editor, setting=name) in self.displayedData:
      return
    self.layoutAboutToBeChanged.emit()
    self.settingsList.append(name)
    self.editorList.append(editor)
    self.layoutChanged.emit()

  def updateEditorOpts(self, editor: FRParamEditor):
    self.layoutAboutToBeChanged.emit()
    for ii in range(len(self.settingsList)-1, -1, -1):
      if self.editorList[ii] is editor:
        del self.settingsList[ii]
        del self.editorList[ii]
    self.addEditors([editor])
    self.layoutChanged.emit()

  def data(self, index: QtCore.QModelIndex, role: int=QtCore.Qt.DisplayRole):
    row = index.row()
    setting = self.settingsList[row]
    editor = self.editorList[row]
    if role == QtCore.Qt.DisplayRole:
      return self.displayFormat.format(setting=setting, editor=editor)
    elif role == QtCore.Qt.EditRole:
      return setting, editor
    else:
      return

  @property
  def displayedData(self):
    return [self.displayFormat.format(setting=stng, editor=edtr)
            for stng, edtr in zip(self.settingsList, self.editorList)]

  def rowCount(self, paren=QtCore.QModelIndex()) -> int:
    return len(self.settingsList)

  def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
    return 'Settings List'

  @staticmethod
  def getSettingsFiles(settingsDir: str, ext: str) -> List[str]:
    files = Path(settingsDir).glob(f'*.{ext}')
    return [file.stem for file in files]

class FRQuickLoaderEditor(FRParamEditor):
  def __init__(self, parent=None, editorList: List[FRParamEditor]=None, oneSettingPerEditor=False):
    super().__init__(parent, paramList=[],
                     saveDir=USER_PROFILES_DIR, fileType='cdefprofile')
    if editorList is None:
      editorList = []
    self.listModel = FREditorListModel(editorList, self)

    self.addNewSetting = FRPopupLineEditor(self, self.listModel)
    self.addNewSetting.setPlaceholderText('Press Tab or type...')
    self.centralLayout.insertWidget(0, self.addNewSetting)

    self.addNewSetting.returnPressed.connect(self.addFromLineEdit)

    self.oneSettingPerEditor = oneSettingPerEditor

  @Slot()
  def addFromLineEdit(self):
    completer = self.addNewSetting.completer()
    selection = completer.completionModel()
    if self.addNewSetting.text() not in self.listModel.displayedData:
      return
    selectionIdx = completer.popup().currentIndex()
    if not selectionIdx.isValid():
      selectionIdx = completer.currentIndex()
    setting, editor = selection.data(selectionIdx, QtCore.Qt.EditRole)
    if editor.name not in self.params.names:
      curGroup = self.params.addChild(dict(name=editor.name, type='group', removable=True))
    else:
      curGroup = self.params.names[editor.name]
    if self.oneSettingPerEditor:
      curGroup.clearChildren()
    newChild = ActionParameter(name=setting, removable=True)
    curGroup.addChild(newChild)
    newChild.sigActivated.connect(lambda: editor.loadSettings(setting))