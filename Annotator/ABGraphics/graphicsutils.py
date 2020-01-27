from typing import Optional

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from functools import wraps

import pickle as pkl

from os import path
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
  saveName, ok = QtWidgets.QInputDialog() \
  .getText(parent, winTitle, winTitle + ':', QtWidgets.QLineEdit.Normal)
  returnVal = None
  if ok:
    returnVal = saveName
    # Prevent overwriting default layout
    if not allowOverwriteDefault and saveName.lower() == 'default':
      QtWidgets.QMessageBox().information(parent, f'Error During Save',
                  'Cannot overwrite default setting.\n\'Default\' is automatically'
                  ' generated, so it should not be modified.', QtWidgets.QMessageBox.Ok)
      return None
    else:
      with open(f'{saveDir}{saveName}.{saveExt}', 'wb') as saveFile:
        pkl.dump(saveObj, saveFile)
  return returnVal

def attemptLoadSettings(fpath, openMode='rb'):
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
  itemNames = glob(dirRegex)
  for name in itemNames:
    # glob returns entire filepath, so keep only filename as layout name
    name = path.basename(name)
    # Also strip file extension
    name = name[0:name.rfind('.')]
    curAction = parentMenu.addAction(name)
    curAction.triggered.connect(partial(triggerFunc, name))