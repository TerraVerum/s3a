import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from functools import wraps

import pickle as pkl

from os import path
from glob import glob
from functools import partial

def applyWaitCursor(func):
  @wraps(func)
  def wrapWithWaitCursor(*args, **kwargs):
    try:
      pg.QAPP.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
      return func(*args, **kwargs)
    finally:
      pg.QAPP.restoreOverrideCursor()
  return wrapWithWaitCursor

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
  # Remove existing menus so only the current file system setup is in place
  if removeExistingChildren:
    for action in parentMenu.children():
      parentMenu.removeAction(action)
  itemNames = glob(dirRegex)
  for name in itemNames:
    # glob returns entire filepath, so keep only filename as layout name
    name = path.basename(name)
    # Also strip file extension
    name = name[0:name.rfind('.')]
    curAction = parentMenu.addAction(name)
    curAction.triggered.connect(partial(triggerFunc, name))
  pass