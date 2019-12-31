import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from functools import wraps

import pickle as pkl

import numpy as np

# Must import * to avoid circular dependency
# If just Component was imported, the namespace must be resolved during import.
# 'import *' doesn't require this check, so the code doesn't fail
from component import *

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
  if ok:
    # Prevent overwriting default layout
    if not allowOverwriteDefault and saveName.lower() == 'default':
      QtGui.QMessageBox().information(parent, 'Error During Save',
                  'Cannot overwrite default layout.', QtGui.QMessageBox.Ok)
      return
    with open(f'{saveDir}{saveName}.{saveExt}', 'wb') as saveFile:
      pkl.dump(saveObj, saveFile)  

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