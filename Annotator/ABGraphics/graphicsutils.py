from typing import Optional

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from functools import wraps

import pickle as pkl

from os.path import basename
from pathlib import Path
from glob import glob
from functools import partial

from .. import appInst
from .. import Annotator

Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

def applyWaitCursor(func):
  @wraps(func)
  def wrapWithWaitCursor(*args, **kwargs):
    try:
      appInst.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
      return func(*args, **kwargs)
    finally:
      appInst.restoreOverrideCursor()
  return wrapWithWaitCursor

def disableAppDuringFunc(func):
  @wraps(func)
  def disableApp(*args, **kwargs):
    # Captures 'self' instance
    mainWin: Annotator = args[0]
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

def dialogSaveToFile(parent, saveObj, winTitle, saveDir, saveExt, allowOverwriteDefault=False):
  failedSave = True
  returnVal = None
  while failedSave:
    saveName, ok = QtWidgets.QInputDialog() \
      .getText(parent, winTitle, winTitle + ':', QtWidgets.QLineEdit.Normal)
    if ok:
      returnVal = saveName
      # Prevent overwriting default layout
      if not allowOverwriteDefault and saveName.lower() == 'default':
        QtWidgets.QMessageBox().information(parent, f'Error During Save',
                                            'Cannot overwrite default setting.\n\'Default\' is automatically'
                                            ' generated, so it should not be modified.', QtWidgets.QMessageBox.Ok)
        return None
      else:
        try:
          # TODO: Make this more robust. At the moment just very basic sanitation
          for disallowedChar in ['/', '\\']:
            saveName = saveName.replace(disallowedChar, '')
          with open(f'{saveDir}{saveName}.{saveExt}', 'wb') as saveFile:
            pkl.dump(saveObj, saveFile)
          failedSave = False
        except FileNotFoundError as e:
          QtWidgets.QMessageBox().information(parent, 'Invalid Name',
                                              'Invalid save name. Please rename the parameter state.',
                                              QtWidgets.QMessageBox.Ok)
  return returnVal

def dialogGetAuthorName(parent: QtWidgets.QMainWindow, defaultAuthFilename: Path) -> (bool, str):
  """
  Attempts to load the username from a default file if found on the system. Otherwise,
  requests the user name. Used before the start of the :class:`Annotator` application
  :param parent:
  :param defaultAuthFilename:
  :return:
  """
  msgDlg = QtWidgets.QMessageBox(parent)
  msgDlg.setModal(True)
  if defaultAuthFilename.exists():
    with open(defaultAuthFilename, 'r') as ifile:
      lines = ifile.readlines()
      if not lines:
        reply = msgDlg.No
      else:
        name = lines[0]
        reply = msgDlg.question(parent, 'Default Author',
                  f'The default author for this application is\n{name}.\n'
                     f'Is this you?', msgDlg.Yes, msgDlg.No)
      if reply == msgDlg.Yes:
        return False, name

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
  return quitApp, name

def attemptLoadSettings(fpath, openMode='rb', showErrorOnFail=True):
  """
  I/O helper function that, when given a file path, either returns the pickle object
  associated with that file or displays an error message and returns nothing.
  """
  pklObj = None
  try:
    curFile = open(fpath, openMode)
    pklObj = pkl.load(curFile)
    curFile.close()
  except IOError as err:
    if showErrorOnFail:
      QtWidgets.QErrorMessage().showMessage(f'Settings could not be loaded.\n'
                                      f'Error: {err}')
  finally:
    return pklObj

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