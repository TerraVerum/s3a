import sys
from functools import partial
from functools import wraps
from glob import glob
from os.path import basename
from pathlib import Path
from traceback import format_exception, format_exception_only
from typing import Optional, Union

from ruamel.yaml import YAML

from cdef import appInst
from cdef.structures import FRAppIOError, FRCdefException

yaml = YAML()
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from cdef.projectvars import ANN_AUTH_DIR

Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

def disableAppDuringFunc(func):
  @wraps(func)
  def disableApp(*args, **kwargs):
    # Captures 'self' instance
    mainWin = args[0]
    try:
      mainWin.setEnabled(False)
      return func(*args, **kwargs)
    finally:
      mainWin.setEnabled(True)
  return disableApp

def popupFilePicker(parent, winTitle: str, fileFilter: str) -> Optional[str]:
  retVal = None
  fileDlg = QtWidgets.QFileDialog()
  fname, _ = fileDlg.getOpenFileName(parent, winTitle, '', fileFilter)

  if len(fname) > 0:
    retVal = fname
  return retVal

def dialogGetSaveFileName(parent, winTitle, defaultTxt: str=None)-> Optional[str]:
  failedSave = True
  returnVal: Optional[str] = None
  while failedSave:
    saveName, ok = QtWidgets.QInputDialog().getText(
      parent, winTitle, winTitle + ':', QtWidgets.QLineEdit.Normal, defaultTxt)
    # TODO: Make this more robust. At the moment just very basic sanitation
    for disallowedChar in ['/', '\\']:
      saveName = saveName.replace(disallowedChar, '')
    if ok and not saveName:
      # User presses 'ok' without typing anything except disallowed characters
      # Keep asking for a name
      continue
    elif not ok:
      # User pressed 'cancel' -- Doesn't matter whether they entered a name or not
      # Stop asking for name
      break
    else:
      # User pressed 'ok' and entered a valid name
      return saveName
  return returnVal

def saveToFile(saveObj, saveDir, saveName, fileType, allowOverwriteDefault=False):
  if not allowOverwriteDefault and saveName.lower() == 'default':
    errMsg = 'Cannot overwrite default setting.\n\'Default\' is automatically' \
             ' generated, so it should not be modified.'
    raise FRAppIOError(errMsg)
  else:
    with open(f'{saveDir}{saveName}.{fileType}', 'w') as saveFile:
      yaml.dump(saveObj, saveFile)

def dialogGetAuthorName(parent: QtWidgets.QMainWindow) -> str:
  """
  Attempts to load the username from a default file if found on the system. Otherwise,
  requests the user name.
  :param parent:
  :return:
  """
  annPath = Path(ANN_AUTH_DIR)
  annFile = annPath.joinpath('defaultAuthor.txt')
  msgDlg = QtWidgets.QMessageBox(parent)
  msgDlg.setModal(True)
  if annFile.exists():
    with open(str(annFile), 'r') as ifile:
      lines = ifile.readlines()
      if not lines:
        reply = msgDlg.No
      else:
        name = lines[0]
        reply = msgDlg.question(parent, 'Default Author',
                  f'The default author for this application is\n{name}.\n'
                     f'Is this you?', msgDlg.Yes, msgDlg.No)
      if reply == msgDlg.Yes:
        return name

  dlg = QtWidgets.QInputDialog(parent)
  dlg.setCancelButtonText('Quit')
  dlg.setModal(True)
  name = ''
  ok = False
  quitApp = False
  while len(name) < 1 or not ok:
    name, ok = dlg.getText(parent, 'Enter Username', 'Please enter your username: ',
                           QtWidgets.QLineEdit.Normal)
    if not ok:
      reply = msgDlg.question(parent, 'Quit Application',
                              f'Quit the application?', msgDlg.Yes, msgDlg.No)
      if reply == msgDlg.Yes:
        quitApp = True
        break
  if quitApp:
    sys.exit(0)
  return name

def raiseErrorLater(err: Exception):
  def _raise():
    raise err
  QtCore.QTimer.singleShot(0, _raise)

def attemptFileLoad(fpath, openMode='r') -> Union[dict, bytes]:
  with open(fpath, openMode) as ifile:
    loadObj = yaml.load(ifile)
  return loadObj

def addDirItemsToMenu(parentMenu, dirRegex, triggerFunc, removeExistingChildren=True):
  """Helper function for populating menu from directory contents"""
  # We don't want all menu children to be removed, since this would also remove the 'edit' and
  # separator options. So, do this step manually. Remove all actions after the separator
  if removeExistingChildren:
    encounteredSep = False
    for ii, action in enumerate(parentMenu.children()):
      if encounteredSep:
        parentMenu.removeAction(action)
      elif action.isSeparator():
        encounteredSep = True
  # TODO: At the moment param files that start with '.' aren't getting included in the
  #  glob
  itemNames = glob(dirRegex)
  for name in itemNames:
    # glob returns entire filepath, so keep only filename as layout name
    name = basename(name)
    # Also strip file extension
    name = name[0:name.rfind('.')]
    curAction = parentMenu.addAction(name)
    curAction.triggered.connect(partial(triggerFunc, name))

def create_addMenuAct(mainWin: QtWidgets.QWidget, parentMenu: QtWidgets.QMenu, title: str, asMenu=False) \
    -> Union[QtWidgets.QMenu, QtWidgets.QAction]:
  menu = None
  if asMenu:
    menu = QtWidgets.QMenu(title, mainWin)
    act = menu.menuAction()
  else:
    act = QtWidgets.QAction(title)
  parentMenu.addAction(act)
  if asMenu:
    return menu
  else:
    return act


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
      if completer is None:
        return
      popup = completer.popup()
      if popup.isVisible() and popup.currentIndex().isValid():
        incAmt = 1 if reason == QtCore.Qt.TabFocusReason else -1
        nextIdx = (completer.currentRow()+incAmt)%completer.completionCount()
        completer.setCurrentRow(nextIdx)
      else:
        completer.complete()
      popup.show()
      popup.setCurrentIndex(completer.currentIndex())
      popup.setFocus()
      return
    else:
      super().focusOutEvent(ev)

  def clear(self):
    super().clear()

  def resetCompleterPrefix(self):
    if self.text() == '':
      self.completer().setCompletionPrefix('')

def makeExceptionsShowDialogs(win: QtWidgets.QMainWindow):
  """
  When a qt application encounters an error, it will generally crash the entire
  application even if this is undesirable behavior. This will make qt applications
  show a dialog rather than crashing.
  Use with caution! Maybe the application *should* crash on an error, but this will
  prevent that from happening.
  """
  # Procedure taken from https://stackoverflow.com/a/40674244/9463643
  def new_except_hook(etype, evalue, tb):
    msgWithTrace = ''.join(format_exception(etype, evalue, tb))
    msgWithoutTrace = str(evalue)
    dlg = FRScrollableErrorDialog(win, notCritical=issubclass(etype, FRCdefException),
                                  msgWithTrace=msgWithTrace, msgWithoutTrace=msgWithoutTrace)
    dlg.show()
    dlg.exec()
  def patch_excepthook():
    sys.excepthook = new_except_hook

  QtCore.QTimer.singleShot(0, patch_excepthook)


class FRScrollableErrorDialog(QtWidgets.QDialog):
  def __init__(self, parent: QtWidgets.QWidget=None, notCritical=False,
               msgWithTrace='', msgWithoutTrace=''):
    super().__init__(parent)
    style = self.style()

    if notCritical:
      icon = style.standardIcon(style.SP_MessageBoxInformation)
      self.setWindowTitle('Information')
    else:
      icon = style.standardIcon(style.SP_MessageBoxCritical)
      self.setWindowTitle('Error')

    self.setWindowIcon(icon)
    verticalLayout = QtWidgets.QVBoxLayout(self)


    scrollArea = QtWidgets.QScrollArea(self)
    scrollArea.setWidgetResizable(True)
    scrollAreaWidgetContents = QtWidgets.QWidget()

    scrollLayout = QtWidgets.QVBoxLayout(scrollAreaWidgetContents)

    # Set to message with trace first so sizing is correct
    scrollMsg = QtWidgets.QLabel(msgWithTrace, scrollAreaWidgetContents)
    scrollMsg.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse
                                      | QtCore.Qt.TextSelectableByKeyboard)
    scrollLayout.addWidget(scrollMsg, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
    scrollArea.setWidget(scrollAreaWidgetContents)
    verticalLayout.addWidget(scrollArea)

    btnLayout = QtWidgets.QHBoxLayout()
    ok = QtWidgets.QPushButton('Ok', self)
    toggleTrace = QtWidgets.QPushButton('Toggle Stack Trace', self)
    btnLayout.addWidget(ok)
    btnLayout.addWidget(toggleTrace)
    spacerItem = QtWidgets.QSpacerItem(ok.width(), ok.height(),
                                       QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Minimum)

    ok.clicked.connect(self.close)
    self.resize(self.sizeHint())

    showTrace = False
    def updateTxt():
      nonlocal showTrace
      if showTrace:
        newText = msgWithTrace
      else:
        newText = msgWithoutTrace
      showTrace = not showTrace
      scrollMsg.setText(newText)

    toggleTrace.clicked.connect(lambda: updateTxt())

    btnLayout.addItem(spacerItem)
    verticalLayout.addLayout(btnLayout)
    ok.setFocus()
    updateTxt()