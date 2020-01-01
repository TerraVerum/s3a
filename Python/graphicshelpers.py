import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from functools import wraps

import pickle as pkl

from os import path
from glob import glob
from functools import partial

import numpy as np

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
      QtGui.QMessageBox().information(parent, 'Error During Save',
                  'Cannot overwrite default setting.', QtGui.QMessageBox.Ok)
    else:
      with open(f'{saveDir}{saveName}.{saveExt}', 'wb') as saveFile:
        pkl.dump(saveObj, saveFile)
  return returnVal

def attemptLoadSettings(fpath, openMode='rb'):
  '''
  I/O helper function that, when given a file path, either returns the pickle object
  associated with that file or displays an error message and returns nothing.
  '''
  pklObj = None
  try:
    curFile = open(fpath, openMode)
    pklObj = pkl.load(curFile)
    curFile.close()
  except IOError as err:
    QtGui.QErrorMessage().showMessage(f'Settings could not be loaded.\n'
                                      f'Error: {err}')
  finally:
    return pklObj
      
def addDirItemsToMenu(parentMenu, dirRegex, triggerFunc, removeExistingChildren=True):
  '''Helper function for populating menu from directory contents'''
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

class ClickableTextItem(pg.TextItem):
  sigClicked = Signal()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.origCursor = self.cursor()
    self.hoverCursor = QtCore.Qt.PointingHandCursor
    self.setAnchor((0.5,0.5))
    self.setAcceptHoverEvents(True)

  def hoverEnterEvent(self, ev):
    self.setCursor(self.hoverCursor)

  def hoverLeaveEvent(self, ev):
    #self.setCursor(self.origCursor)
    self.unsetCursor()

  def mousePressEvent(self, ev):
    self.sigClicked.emit()

class ClickableImageItem(pg.ImageItem):
  sigClicked = Signal(object)

  def mouseClickEvent(self, ev):
    if ev.button() == QtCore.Qt.LeftButton:
      self.sigClicked.emit(ev)
      return

class SaveablePolyROI(pg.PolyLineROI):
  def __init__(self, *args, **kwargs):
    # Since this won't execute until after module import, it doesn't cause
    # a dependency
    super().__init__(*args, **kwargs)
    # Force new menu options
    self.getMenu()

  def getMenu(self, *args, **kwargs):
    '''
    Adds context menu option to add current ROI area to existing region
    '''
    if self.menu is None:
      menu = super().getMenu()
      addAct = QtGui.QAction("Add to Region", menu)
      menu.addAction(addAct)
      self.addAct = addAct
      self.menu = menu
    return self.menu

  def getImgMask(self, imgItem: pg.ImageItem):
    imgMask = np.zeros(imgItem.image.shape[0:2], dtype='bool')
    roiSlices,_ = self.getArraySlice(imgMask, imgItem)
    # TODO: Clip regions that extend beyond image dimensions
    roiSz = [curslice.stop - curslice.start for curslice in roiSlices]
    # renderShapeMask takes width, height args. roiSlices has row/col sizes,
    # so switch this order when passing to renderShapeMask
    roiSz = roiSz[::-1]
    roiMask = self.renderShapeMask(*roiSz).astype('uint8')
    # Also, the return value for renderShapeMask is given in col-major form.
    # Transpose this, since all other data is in row-major.
    roiMask = roiMask.T
    imgMask[roiSlices[0], roiSlices[1]] = roiMask
    return imgMask