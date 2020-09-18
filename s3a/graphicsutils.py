from __future__ import annotations
import html
import sys
from functools import partial
from functools import wraps
from os.path import basename
from pathlib import Path
from traceback import format_exception
from typing import Optional, Union, Callable, Generator, Sequence

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import Parameter
from ruamel.yaml import YAML

import s3a
from s3a.constants import ANN_AUTH_DIR
from s3a.structures import FRIOError, FRS3AException, FilePath, FRS3AWarning

yaml = YAML()

Signal = QtCore.Signal
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

def saveToFile(saveObj, savePath: Path, allowOverwriteDefault=False):
  if not allowOverwriteDefault and savePath.stem.lower() == 'default':
    errMsg = 'Cannot overwrite default setting.\n\'Default\' is automatically' \
             ' generated, so it should not be modified.'
    raise FRIOError(errMsg)
  else:
    # Known pycharm bug
    # noinspection PyTypeChecker
    with open(savePath, 'w') as saveFile:
      yaml.dump(saveObj, saveFile)

def dialogGetAuthorName(parent: QtWidgets.QMainWindow) -> str:
  """
  Attempts to load the username from a default file if found on the system. Otherwise,
  requests the user name.
  :param parent:
  :return:
  """
  annPath = Path(ANN_AUTH_DIR)
  authorFname = annPath.joinpath('defaultAuthor.txt')
  msgDlg = QtWidgets.QMessageBox(parent)
  msgDlg.setModal(True)
  if authorFname.exists():
    with open(str(authorFname), 'r') as ifile:
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

def attemptFileLoad(fpath: FilePath , openMode='r') -> Union[dict, bytes]:
  with open(fpath, openMode) as ifile:
    loadObj = yaml.load(ifile)
  return loadObj

def addDirItemsToMenu(parentMenu: QtWidgets.QMenu, dirGlob: Generator,
                      triggerFunc: Callable, removeExistingChildren=True):
  """Helper function for populating menu from directory contents"""
  # We don't want all menu children to be removed, since this would also remove the 'edit' and
  # separator options. So, do this step manually. Remove all actions after the separator
  if removeExistingChildren:
    encounteredSep = False
    for ii, action in enumerate(parentMenu.children()):
      action: QtWidgets.QAction
      if encounteredSep:
        parentMenu.removeAction(action)
      elif action.isSeparator():
        encounteredSep = True
  # TODO: At the moment param files that start with '.' aren't getting included in the
  #  glob
  for name in dirGlob:
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
  def __init__(self, parent: QtWidgets.QWidget=None, model: QtCore.QAbstractListModel=None,
               placeholderText='Press Tab or type...', clearOnComplete=True,
               forceMatch=True):
    super().__init__(parent)
    self.setPlaceholderText(placeholderText)
    self.clearOnComplete = clearOnComplete
    self.forceMatch = forceMatch

    if model is not None:
      self.setModel(model)

  def setModel(self, model: QtCore.QAbstractListModel):
    completer = QtWidgets.QCompleter(model, self)
    completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
    completer.setCompletionRole(QtCore.Qt.DisplayRole)
    completer.setFilterMode(QtCore.Qt.MatchContains)
    if self.clearOnComplete:
      completer.activated.connect(lambda: QtCore.QTimer.singleShot(0, self.clear))

    self.textChanged.connect(lambda: self.resetCompleterPrefix())

    self.setCompleter(completer)

  # TODO: Get working with next prev focusing for smoother logic
  # def focusNextPrevChild(self, nextChild: bool):
  #   if self.forceMatch and self.text() not in self.completer().model().stringList():
  #     dummyFocusEv = QtGui.QFocusEvent(QtCore.QEvent.FocusOut)
  #     self.focusOutEvent(dummyFocusEv)
  #     return False
  #   return super().focusNextPrevChild(nextChild)

  def focusOutEvent(self, ev: QtGui.QFocusEvent):
    reason = ev.reason()
    if reason in [QtCore.Qt.TabFocusReason, QtCore.Qt.BacktabFocusReason,
                  QtCore.Qt.OtherFocusReason]:
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
      ev.accept()
      return
    else:
      super().focusOutEvent(ev)

  def clear(self):
    super().clear()

  def resetCompleterPrefix(self):
    if self.text() == '':
      self.completer().setCompletionPrefix('')

old_sys_except_hook = sys.excepthook
usingPostponedErrors = False
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
    # Allow sigabort to kill the app
    if etype in [KeyboardInterrupt, SystemExit]:
      s3a.appInst.exit(1)
      s3a.appInst.processEvents()
      raise
    msgWithTrace = ''.join(format_exception(etype, evalue, tb))
    msgWithoutTrace = str(evalue)
    dlg = FRScrollableErrorDialog(win, notCritical=issubclass(etype, (FRS3AException,
                                                                      FRS3AWarning)),
                                  msgWithTrace=msgWithTrace, msgWithoutTrace=msgWithoutTrace)
    dlg.show()
    dlg.exec_()
  def patch_excepthook():
    global usingPostponedErrors
    sys.excepthook = new_except_hook
    usingPostponedErrors = True
  QtCore.QTimer.singleShot(0, patch_excepthook)
  s3a.appInst.processEvents()

def restoreExceptionBehavior():
  def patch_excepthook():
    global usingPostponedErrors
    sys.excepthook = old_sys_except_hook
    usingPostponedErrors = False
  QtCore.QTimer.singleShot(0, patch_excepthook)
  s3a.appInst.processEvents()

def raiseErrorLater(err: Exception):
  # Fire immediately if not in gui mode
  if not usingPostponedErrors:
    raise err
  # else
  def _raise():
    raise err
  QtCore.QTimer.singleShot(0, _raise)


class FRScrollableErrorDialog(QtWidgets.QDialog):
  def __init__(self, parent: QtWidgets.QWidget=None, notCritical=False,
               msgWithTrace='', msgWithoutTrace=''):
    super().__init__(parent)
    style = self.style()
    self.setModal(True)

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
    msgLbl = QtWidgets.QLabel(msgWithTrace, scrollAreaWidgetContents)
    msgLbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse
                                      | QtCore.Qt.TextSelectableByKeyboard)
    scrollLayout.addWidget(msgLbl, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
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
      msgLbl.setText(newText)
    self.msgLbl = msgLbl
    toggleTrace.clicked.connect(lambda: updateTxt())

    btnLayout.addItem(spacerItem)
    verticalLayout.addLayout(btnLayout)
    self.toggleTrace = toggleTrace
    ok.setFocus()
    updateTxt()

def autosaveOptsDialog(parent):
  dlg = QtWidgets.QDialog(parent)
  layout = QtWidgets.QGridLayout(dlg)
  dlg.setLayout(layout)

  fileBtn = QtWidgets.QPushButton('Select Output Folder')
  folderNameDisplay = QtWidgets.QLabel('', dlg)
  def retrieveFolderName():
    folderDlg = QtWidgets.QFileDialog(dlg)
    folderDlg.setModal(True)
    folderName = folderDlg.getExistingDirectory(dlg, 'Select Output Folder')

    dlg.folderName = folderName
    folderNameDisplay.setText(folderName)
  fileBtn.clicked.connect(retrieveFolderName)

  layout.addWidget(folderNameDisplay, 0, 0)
  layout.addWidget(fileBtn, 0, 1)

  saveLbl = QtWidgets.QLabel('Save Name:', dlg)
  baseFileNameEdit = QtWidgets.QLineEdit(dlg)
  dlg.baseFileNameEdit = baseFileNameEdit

  layout.addWidget(saveLbl, 1, 0)
  layout.addWidget(baseFileNameEdit, 1, 1)

  intervalLbl = QtWidgets.QLabel('Save Interval (mins):', dlg)
  intervalEdit = QtWidgets.QSpinBox(dlg)
  intervalEdit.setMinimum(1)
  dlg.intervalEdit = intervalEdit

  layout.addWidget(intervalLbl, 2, 0)
  layout.addWidget(intervalEdit, 2, 1)

  saveDescr = QtWidgets.QLabel('Every <em>interval</em> minutes, a new autosave is'
                               ' created from the component list. Its name is'
                               ' [Parent Folder]/[base name]_[counter].csv, where'
                               ' counter is the current save file number.', dlg)
  saveDescr.setWordWrap(True)
  layout.addWidget(saveDescr, 3, 0, 1, 2)


  okBtn = QtWidgets.QPushButton('Ok')
  cancelBtn = QtWidgets.QPushButton('Cancel')
  cancelBtn.clicked.connect(dlg.reject)
  okBtn.clicked.connect(dlg.accept)
  dlg.okBtn = okBtn

  layout.addWidget(okBtn, 4, 0)
  layout.addWidget(cancelBtn, 4, 1)
  return dlg

# Taken directly from https://stackoverflow.com/a/46212292/9463643
class QAwesomeTooltipEventFilter(QtCore.QObject):
  """
  Tooltip-specific event filter dramatically improving the tooltips of all
  widgets for which this filter is installed.

  Motivation
  ----------
  **Rich text tooltips** (i.e., tooltips containing one or more HTML-like
  tags) are implicitly wrapped by Qt to the width of their parent windows and
  hence typically behave as expected.

  **Plaintext tooltips** (i.e., tooltips containing no such tags), however,
  are not. For unclear reasons, plaintext tooltips are implicitly truncated to
  the width of their parent windows. The only means of circumventing this
  obscure constraint is to manually inject newlines at the appropriate
  80-character boundaries of such tooltips -- which has the distinct
  disadvantage of failing to scale to edge-case display and device
  environments (e.g., high-DPI). Such tooltips *cannot* be guaranteed to be
  legible in the general case and hence are blatantly broken under *all* Qt
  versions to date. This is a `well-known long-standing issue <issue_>`__ for
  which no official resolution exists.

  This filter globally addresses this issue by implicitly converting *all*
  intercepted plaintext tooltips into rich text tooltips in a general-purpose
  manner, thus wrapping the former exactly like the latter. To do so, this
  filter (in order):

  #. Auto-detects whether the:

     * Current event is a :class:`QEvent.ToolTipChange` event.
     * Current widget has a **non-empty plaintext tooltip**.

  #. When these conditions are satisfied:

     #. Escapes all HTML syntax in this tooltip (e.g., converting all ``&``
        characters to ``&amp;`` substrings).
     #. Embeds this tooltip in the Qt-specific ``<qt>..</qt>`` tag, thus
        implicitly converting this plaintext tooltip into a rich text tooltip.

  . _issue:
      https://bugreports.qt.io/browse/QTBUG-41051
  """


  def eventFilter(self, widget: QtCore.QObject, event: QtCore.QEvent) -> bool:
    """
    Tooltip-specific event filter handling the passed Qt object and event.
    """

    # If this is a tooltip event...
    if event.type() == QtCore.QEvent.ToolTipChange:
      # If the target Qt object containing this tooltip is *NOT* a widget,
      # raise a human-readable exception. While this should *NEVER* be the
      # case, edge cases are edge cases because they sometimes happen.
      if not isinstance(widget, QtWidgets.QWidget):
        raise ValueError('QObject "{}" not a widget.'.format(widget))

      # Tooltip for this widget if any *OR* the empty string otherwise.
      tooltip = widget.toolTip()

      # If this tooltip is both non-empty and not already rich text...
      if tooltip and not QtCore.Qt.mightBeRichText(tooltip):
        # Convert this plaintext tooltip into a rich text tooltip by:
        #
        #* Escaping all HTML syntax in this tooltip.
        #* Embedding this tooltip in the Qt-specific "<qt>...</qt>" tag.
        tooltip = '<qt>{}</qt>'.format(html.escape(tooltip))

        # Replace this widget's non-working plaintext tooltip with this
        # working rich text tooltip.
        widget.setToolTip(tooltip)

        # Notify the parent event handler this event has been handled.
        return True

    # Else, defer to the default superclass handling of this event.
    return super().eventFilter(widget, event)


def contextMenuFromEditorActions(editors: Union[s3a.FRParamEditor, Sequence[s3a.FRParamEditor]],
                                 title: str=None, menuParent: QtWidgets.QWidget=None):
  if not isinstance(editors, Sequence):
    editors = [editors]
  if title is None:
    title = editors[0].name

  menu = QtWidgets.QMenu(title, menuParent)
  for editor in editors:
    actions = []
    paramNames = []
    def findActions(paramRoot: Parameter):
      for child in paramRoot.childs:
        findActions(child)
      if 'action' in paramRoot.opts['type']:
        actions.append(paramRoot)
        paramNames.append(paramRoot.name())
    findActions(editor.params)
    for action, name in zip(actions, paramNames):
      menu.addAction(name, action.activate)

  return menu