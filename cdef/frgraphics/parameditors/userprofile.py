from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ActionParameter

from .genericeditor import FRParamEditor
from cdef.projectvars import USER_PROFILES_DIR

Slot = QtCore.pyqtSlot

@dataclass
class FRSettingListEntry:
  settingName: str
  editor: FRParamEditor

class FREditorListModel(QtCore.QAbstractListModel):
  def __init__(self, editorList: List[FRParamEditor], parent: QtWidgets.QWidget=None):
    super().__init__(parent)
    self.settingsList: List[FRSettingListEntry] = []
    self.addEditors(editorList)

  def addEditors(self, editorList: List[FRParamEditor]):
    self.layoutAboutToBeChanged.emit()
    for editor in editorList:
      for settingName in self.getSettingsFiles(editor.saveDir, editor.fileType):
        self.settingsList.append(FRSettingListEntry(settingName, editor))
      # editor.sigParamStateCreated.connect(lambda name, e=editor:
      #                                     self.updateEditorOpts(e))
    self.layoutChanged.emit()

  def addOptForEditor(self, editor: FRParamEditor, name: str):
    self.layoutAboutToBeChanged.emit()
    self.settingsList.append(FRSettingListEntry(name, editor))
    self.layoutChanged.emit()

  def updateEditorOpts(self, editor: FRParamEditor):
    self.layoutAboutToBeChanged.emit()
    for ii in range(len(self.settingsList)-1, -1, -1):
      if self.settingsList[ii].editor is editor:
        del self.settingsList[ii]
    self.addEditors([editor])
    self.layoutChanged.emit()

  def data(self, index: QtCore.QModelIndex, role: int=QtCore.Qt.DisplayRole):
    settingPair = self.settingsList[index.row()]
    if role == QtCore.Qt.DisplayRole:
      return f'{settingPair.settingName} | {settingPair.editor.name}'
    elif role == QtCore.Qt.EditRole:
      return settingPair
    else:
      return

  def rowCount(self, paren=QtCore.QModelIndex()) -> int:
    return len(self.settingsList)

  def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
    return 'Settings List'

  @staticmethod
  def getSettingsFiles(settingsDir: str, ext: str) -> List[str]:
    files = Path(settingsDir).glob(f'*.{ext}')
    return [file.stem for file in files]

class FRPopupLineEditor(QtWidgets.QLineEdit):
  def __init__(self, parent: QtWidgets.QWidget=None, model: QtCore.QAbstractListModel=None):
    super().__init__(parent)

    if model is not None:
      self.setModel(model)

  def setModel(self, model: QtCore.QAbstractListModel):
    completer = QtWidgets.QCompleter(model, self)
    completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    completer.setCompletionRole(QtCore.Qt.DisplayRole)
    completer.setFilterMode(QtCore.Qt.MatchContains)
    completer.activated.connect(lambda: QtCore.QTimer.singleShot(0, self.clear))
    self.textChanged.connect(self.resetCompleterPrefix)

    self.setCompleter(completer)

  def focusOutEvent(self, ev: QtGui.QFocusEvent):
    reason = ev.reason()
    if reason == QtCore.Qt.TabFocusReason or reason == QtCore.Qt.BacktabFocusReason:
      # Simulate tabbing through completer options instead of losing focus
      self.setFocus()
      completer = self.completer()
      if completer.popup().isVisible():
        incAmt = 1 if reason == QtCore.Qt.TabFocusReason else -1
        nextIdx = (completer.currentRow()+incAmt)%completer.completionCount()
        completer.setCurrentRow(nextIdx)
      else:
        completer.complete()
      completer.popup().setCurrentIndex(completer.currentIndex())
      return
    else:
      super().focusOutEvent(ev)

  def clear(self):
    super().clear()

  def resetCompleterPrefix(self):
    if self.text() == '':
      self.completer().setCompletionPrefix('')

class FRUserProfileEditor(FRParamEditor):
  def __init__(self, parent=None, editorList: List[FRParamEditor]=None, oneSettingPerEditor=True):
    super().__init__(parent, paramList=[],
                     saveDir=USER_PROFILES_DIR, fileType='cdefprofile')
    if editorList is None:
      editorList = []
    self.listModel = FREditorListModel(editorList, self)

    self.addNewSetting = FRPopupLineEditor(self, self.listModel)
    self.centralLayout.insertWidget(0, self.addNewSetting)

    self.addNewSetting.returnPressed.connect(self.addFromLineEdit)

    self.oneSettingPerEditor = oneSettingPerEditor

  @Slot()
  def addFromLineEdit(self):
    completer = self.addNewSetting.completer()
    selection = completer.completionModel()
    if self.addNewSetting.text() != completer.currentCompletion():
      return
    setting_editor: FRSettingListEntry = selection.data(completer.currentIndex(), QtCore.Qt.EditRole)
    editor = setting_editor.editor
    setting = setting_editor.settingName
    if editor.name not in self.params.names:
      curGroup = self.params.addChild(dict(name=editor.name, type='group', removable=True))
    else:
      curGroup = self.params.names[editor.name]
    if self.oneSettingPerEditor:
      curGroup.clearChildren()
    newChild = ActionParameter(name=setting, removable=True)
    curGroup.addChild(newChild)
    newChild.sigActivated.connect(lambda: editor.loadSettings(setting))